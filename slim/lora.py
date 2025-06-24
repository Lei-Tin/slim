import torch
import torch.nn.functional as F
from slim.quantization.quantization import Quantizer as AutoQuantizer
import tqdm.auto as tqdm
from .utils import prune_nm, get_layers_list, find_layers
from typing import Optional, Dict, Any
import math
import multiprocessing
import numpy as np
import logging
import pandas as pd
import os

# Configure logging for calibration errors
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global list to collect calibration errors during pipeline execution
CALIBRATION_ERRORS = []


def save_calibration_errors_to_csv(calibration_errors, csv_path):
    """
    Save calibration errors to CSV file
    
    Args:
        calibration_errors: List of calibration error dictionaries
        csv_path: Path to save CSV file
    """
    if not calibration_errors:
        logger.warning("No calibration errors to save")
        return
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.', exist_ok=True)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(calibration_errors)
    df.to_csv(csv_path, index=False)
    
    # Log summary
    logger.info(f"Saved {len(calibration_errors)} calibration error measurements to {csv_path}")
    logger.info(f"Overall mean relative error: {df['mean_relative_error'].mean():.6f}")
    logger.info(f"Overall max relative error: {df['mean_relative_error'].max():.6f}")


def sqrtm_scipy(A: np.ndarray):
    if not isinstance(A, np.ndarray):
        raise RuntimeError("input matrix must be a numpy array")
    import scipy.linalg as spla
    A_sqrt, errest = spla.sqrtm(A, disp=False)
    return dict(A_sqrt=A_sqrt, errest=errest)


def prune_and_optimize_lora(
        L,
        R,
        num_iters=1000,
        lr_end_factor=1e-4
):
    """
    Prune L in LoRA and optimizer L and R to compensate for the pruning loss.

    Args:
        L: torch.Tensor, The left matrix in LoRA
        R: torch.Tensor, The right matrix in LoRA
        num_iters: int, The number of optimization iterations
        lr_end_factor: float, The factor to scale the learning rate by at the end of optimization

    Returns:
        torch.Tensor, The mask of the pruned elements in L
    """
    target = torch.matmul(L, R).float()
    target_norm = torch.norm(target).item()
    L_mask = prune_nm(L.t(), 2, 4).t()
    L[L_mask] = 0
    L_param = torch.nn.Parameter(L.float(), requires_grad=True)
    R_param = torch.nn.Parameter(R.float(), requires_grad=True)
    optimizer = torch.optim.Adam([L_param, R_param], lr=1e6 / min(L.shape[0], R.shape[1]) ** 2)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=lr_end_factor, total_iters=num_iters)
    progress_bar = tqdm.tqdm(range(num_iters))
    initial_error = torch.norm(torch.matmul(L, R).float() - target.float()) / target_norm
    for iter in progress_bar:
        output = torch.matmul(L_param, R_param)
        loss = torch.norm(output - target) / target_norm
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        L_param.data[L_mask] = 0
        progress_bar.set_description(
            'Iteration {} - Initial Loss: {:.2f} - Current Loss: {:.2f}, LR: {:.2e}'.format(
                iter + 1,
                initial_error.item(),
                loss.item(),
                scheduler.get_lr()[0]
            )
        )
    L.data = L_param.data.to(torch.bfloat16)
    R.data = R_param.data.to(torch.bfloat16)
    return L_mask


def quantize_lora(
        model,
        bitwidth=8,
        lora_tile_size=256
):
    """
    Quantize the LoRA matrices in a model.

    Args:
        model: nn.Module, The model to quantize
        bitwidth: int, The number of bits to quantize the LoRA matrices to
        lora_tile_size: int, The size of the

    Returns:
        None
    """

    quantizer = AutoQuantizer(
        "weight",
        num_bits=bitwidth,
        block_quantization=True,
        block_dim=lora_tile_size,
    )
    layers = get_layers_list(model)

    progress_bar = tqdm.tqdm(range(len(layers)))

    for i in progress_bar:
        layer = layers[i]

        subset = find_layers(layer)

        for name in subset:
            progress_bar.set_description(f"Layer {i} - Quantizing LoRA for {name}")

            quantized_lora_left = quantizer.dequantize_absmax(
                quantizer.quantize_weight(subset[name].lora_left.data)
            )

            quantized_lora_right = quantizer.dequantize_absmax(
                quantizer.quantize_weight(subset[name].lora_right.data)
            )

            subset[name].lora_left.data = quantized_lora_left.to(subset[name].weight.dtype)
            subset[name].lora_right.data = quantized_lora_right.to(subset[name].weight.dtype)
            subset[name].lora_quantizer = quantizer


class ScaleHookFactoryDiagonal:
    """
    QERA Diagonal Scale Hook Factory
    scale = diag( sqrt( E[ x_1^2]), sqrt( E[ x_2^2]), ..., sqrt( E[ x_n^2] ) )
    """
    def __init__(self, torch_dtype=None, store_on_cpu=True):
        self.scales = {}
        self.n_samples = {}
        self.compute_devices = {}
        self.torch_dtype = torch_dtype
        self.handles = []
        self.store_on_cpu = store_on_cpu  # New parameter to control CPU storage

    def get_scale_hook(self, name: str) -> callable:
        self.scales[name] = None
        self.n_samples[name] = 0

        @torch.no_grad()
        def scale_hook(
            module: torch.nn.Linear,
            input: tuple[torch.Tensor],
            output: torch.Tensor,
        ) -> None:
            x = input[0]
            x = x.view(-1, x.shape[-1])
            num_samples, _ = x.shape
            x = x.pow(2).sum(0)

            self.n_samples[name] += num_samples
            if self.scales[name] is None:
                self.compute_devices[name] = x.device
                if self.torch_dtype is None:
                    self.torch_dtype = x.dtype
                scale = x.to(self.torch_dtype)
            else:
                # Move existing scale back to compute device for accumulation
                if self.store_on_cpu and self.scales[name].device.type == 'cpu':
                    scale = self.scales[name].to(self.compute_devices[name])
                else:
                    scale = self.scales[name].to(self.compute_devices[name])
                scale = scale + x.to(self.torch_dtype)

            # Store on CPU to save GPU memory if enabled
            if self.store_on_cpu:
                self.scales[name] = scale.cpu()
            else:
                self.scales[name] = scale

        return scale_hook

    @torch.no_grad()
    def get_scale_dict(self, progress_bar=False) -> dict[str, torch.Tensor]:
        scale_names_prog_bar = tqdm.tqdm(
            self.scales, desc="Computing scale", disable=not progress_bar, total=len(self.scales)
        )

        for name in scale_names_prog_bar:
            if self.scales[name] is not None:  # Only compute for layers that actually collected data
                # Move scale to compute device for processing
                if self.store_on_cpu and self.scales[name].device.type == 'cpu':
                    scale = self.scales[name].to(self.compute_devices[name])
                else:
                    scale = self.scales[name].to(self.compute_devices[name])
                scale = torch.sqrt(scale) * (1 / math.sqrt(self.n_samples[name]))
                
                # Store final scale on CPU to save memory
                if self.store_on_cpu:
                    self.scales[name] = scale.cpu()
                else:
                    self.scales[name] = scale

        return self.scales
    
    def get_scale_for_layer(self, layer_name: str, device: torch.device) -> torch.Tensor:
        """Get scale for a specific layer, moving it to the requested device only when needed."""
        if layer_name in self.scales and self.scales[layer_name] is not None:
            return self.scales[layer_name].to(device)
        return None

    def remove_all_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


class ScaleHookFactoryRxx:
    """
    QERA RXX Scale Hook Factory (Exact method)
    For row vector x, scale = E[ x^T x ] ^ 0.5, where Rxx = E[ x^T x ] is the auto-correlation matrix
    """
    def __init__(self, torch_dtype=None, store_on_cpu=True):
        self.scales = {}
        self.n_samples = {}
        self.compute_devices = {}
        self.torch_dtype = torch_dtype
        self.handles = []
        self.store_on_cpu = store_on_cpu  # New parameter to control CPU storage

    def get_scale_hook(self, name: str) -> callable:
        self.scales[name] = None
        self.n_samples[name] = 0

        @torch.no_grad()
        def scale_hook(
            module: torch.nn.Linear,
            input: tuple[torch.Tensor],
            output: torch.Tensor,
        ) -> None:
            x = input[0]
            x = x.reshape(-1, x.shape[-1])
            n_samples, in_features = x.shape
            
            if self.scales[name] is None:
                if self.torch_dtype is None:
                    self.torch_dtype = x.dtype
                self.compute_devices[name] = x.device
                if self.store_on_cpu:
                    self.scales[name] = torch.zeros(
                        in_features, in_features, dtype=torch.float64, device='cpu'
                    )  # Use float64 for accumulation, store on CPU
                else:
                    self.scales[name] = torch.zeros(
                        in_features, in_features, dtype=torch.float64, device=x.device
                    )

            compute_device = self.compute_devices[name]
            
            # Move scale to compute device for accumulation
            if self.store_on_cpu and self.scales[name].device.type == 'cpu':
                scales = self.scales[name].to(compute_device)
            else:
                scales = self.scales[name].to(compute_device)
            
            x = x.to(self.torch_dtype)
            x = x.to(compute_device)
            
            # Batched outer product: sum over batch dimension
            delta = torch.einsum("bi,bj->ij", x, x).to(torch.float64)
            scales += delta
            
            # Store back on CPU to save GPU memory if enabled
            if self.store_on_cpu:
                self.scales[name] = scales.cpu()
            else:
                self.scales[name] = scales
            self.n_samples[name] += n_samples

        return scale_hook

    @torch.no_grad()
    def get_scale_dict(self, progress_bar=False, sqrtm_implementation: str = "scipy", sqrtm_num_iters: int = 200) -> dict[str, torch.Tensor]:
        if sqrtm_implementation == "scipy":
            # Use multiprocessing for scipy implementation to speed up computation
            return self._get_scale_dict_scipy_multiprocessing(progress_bar)
        else:
            # Use iterative method (single-threaded, GPU-accelerated)
            return self._get_scale_dict_iterative(progress_bar, sqrtm_num_iters)
    
    def _get_scale_dict_scipy_multiprocessing(self, progress_bar=False) -> dict[str, torch.Tensor]:
        """Compute scales using scipy with multiprocessing for speed."""
        # convert to numpy
        for name in self.scales:
            if self.scales[name] is not None:
                # Normalize by number of samples first
                if self.store_on_cpu and self.scales[name].device.type == 'cpu':
                    scale = self.scales[name]
                else:
                    scale = self.scales[name].cpu()
                scale = scale / self.n_samples[name]
                self.scales[name] = scale.numpy()
        
        num_cores = multiprocessing.cpu_count()
        num_processes = max(1, num_cores // 64)

        with multiprocessing.Pool(num_processes) as pool:
            with tqdm.tqdm(
                total=len(self.scales), desc="Computing scale", disable=not progress_bar
            ) as pbar:
                for name, scale_and_err in zip(
                    self.scales.keys(), pool.imap(sqrtm_scipy, self.scales.values())
                ):
                    if self.scales[name] is not None:
                        scale = scale_and_err["A_sqrt"]
                        self.scales[name] = scale
                    pbar.update()

        # convert to torch tensor
        for name in self.scales:
            if self.scales[name] is not None:
                scale = self.scales[name]
                n_samples = self.n_samples[name]
                scale = (
                    torch.from_numpy(scale)
                    .to(torch.float32)
                    .to(self.compute_devices[name])
                )
                scale = scale * (1 / math.sqrt(n_samples))
                
                # Store final scale on CPU to save memory
                if self.store_on_cpu:
                    self.scales[name] = scale.cpu()
                else:
                    self.scales[name] = scale

        return self.scales
    
    def _get_scale_dict_iterative(self, progress_bar=False, sqrtm_num_iters: int = 200) -> dict[str, torch.Tensor]:
        """Compute scales using iterative Newton-Schulz method (single-threaded, GPU-accelerated)."""
        scale_names_prog_bar = tqdm.tqdm(
            self.scales, desc="Computing RXX scale (iterative)", disable=not progress_bar, total=len(self.scales)
        )

        for name in scale_names_prog_bar:
            if self.scales[name] is not None:  # Only compute for layers that actually collected data
                compute_device = self.compute_devices[name]
                
                # Move scale to compute device for processing
                if self.store_on_cpu and self.scales[name].device.type == 'cpu':
                    scale = self.scales[name].to(compute_device)
                else:
                    scale = self.scales[name].to(compute_device)
                
                # Normalize by number of samples
                scale = scale / self.n_samples[name]
                
                # Use iterative Newton-Schulz method
                scale_sqrt = sqrtm_newton_schulz(scale.unsqueeze(0), numIters=sqrtm_num_iters).squeeze(0)
                scale_sqrt = scale_sqrt.to(torch.float32)
                
                # Store final scale on CPU to save memory
                if self.store_on_cpu:
                    self.scales[name] = scale_sqrt.cpu()
                else:
                    self.scales[name] = scale_sqrt

        return self.scales
    
    def get_scale_for_layer(self, layer_name: str, device: torch.device) -> torch.Tensor:
        """Get scale for a specific layer, moving it to the requested device only when needed."""
        if layer_name in self.scales and self.scales[layer_name] is not None:
            return self.scales[layer_name].to(device)
        return None

    def remove_all_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def sqrtm_newton_schulz(A, numIters=200):
    """Newton-Schulz iterations method to get matrix square root."""
    normA = torch.linalg.matrix_norm(A, keepdim=True)
    err = normA + 1.0
    I = torch.eye(*A.shape[-2:], dtype=A.dtype, device=A.device)
    Z = torch.eye(*A.shape[-2:], dtype=A.dtype, device=A.device).expand_as(A)
    Y = A / normA
    
    for i in range(numIters):
        T = 0.5 * (3.0 * I - Z.bmm(Y))
        Y_new = Y.bmm(T)
        Z_new = T.bmm(Z)

        # Check for convergence
        mat_a_approx = torch.bmm(Y_new, Y_new) * normA
        residual = A - mat_a_approx
        current_err = torch.linalg.matrix_norm(residual, keepdim=True) / normA
        if torch.all(current_err > err):
            break

        err = current_err
        Y = Y_new
        Z = Z_new

    sA = Y * torch.sqrt(normA)
    return sA


def get_layer_name(model, layer):
    """Get the full name of a layer in the model"""
    for name, module in model.named_modules():
        if module is layer:
            return name
    return None


def find_layers_to_register_scale_hook(model):
    """Find layers to register scale hooks for - simplified version for SLiM"""
    layers_to_register = []
    
    # Find all linear layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Skip embedding and output layers
            if any(skip in name.lower() for skip in ['embed', 'lm_head', 'head']):
                continue
            layers_to_register.append({
                "target_layer": name,
                "layers_sharing_scale": []
            })
    
    return layers_to_register


def register_scale_hooks(model, layers_to_register: list, mode: str = "diag", store_on_cpu: bool = True) -> 'ScaleHookFactoryDiagonal':
    """Register QERA scale hooks following the original QERA implementation"""
    if mode in ["diagonal", "diag"]:
        hook_factory = ScaleHookFactoryDiagonal(torch_dtype=torch.float32, store_on_cpu=store_on_cpu)
    elif mode == "rxx":
        hook_factory = ScaleHookFactoryRxx(torch_dtype=torch.float32, store_on_cpu=store_on_cpu)
    else:
        raise ValueError(f"Mode {mode} not supported. Use 'diag' or 'rxx'")
    
    # Get layers to register if not provided
    if not layers_to_register:
        layers_to_register = find_layers_to_register_scale_hook(model)
    
    for target_and_share in layers_to_register:
        target_layer_name = target_and_share["target_layer"]
        # Get the actual layer from the model
        target_layer = None
        for name, module in model.named_modules():
            if name == target_layer_name and isinstance(module, torch.nn.Linear):
                target_layer = module
                break
        
        if target_layer is not None:
            handle = target_layer.register_forward_hook(hook_factory.get_scale_hook(target_layer_name))
            hook_factory.handles.append(handle)
    
    return hook_factory


def add_lora(
        module,
        W_mask,
        rank_ratio=0.01,
        slim_lora=False,
        activations=None,
        quantizer=None,
        prune_lora=False,
        separate_lora=True,
        lora_tile_size=None,
        quantize_first=False,
        scale_important_weights=False,
        use_qera=False,
        qera_mode="diag",
        qera_scale_dict=None,
        calibration_inputs=None,
        log_calibration_error=False,
        layer_name=""
):
    """
    Add low-rank adapters to compensate for the compression loss.

    Args:
        module: nn.Module, The module to add the low-rank adapters to
        W_mask: torch.Tensor, The mask of the pruned weights
        rank_ratio: float, The ratio of the rank of the low-rank approximation to the number of rows in the weight matrix
        slim_lora: bool, Whether to use slim LoRA
        activations: torch.Tensor, The activations of the layer
        quantizer: Quantizer, The quantizer to use
        prune_lora: bool, Whether to prune the LoRA matrices
        separate_lora: bool, Whether to use separate LoRA matrices
        lora_tile_size: int, The size of the LoRA tiles
        use_qera: bool, Whether to use QERA's L and R matrices
        qera_mode: str, The mode for QERA scaling ("diag" or "rxx")
        qera_scale_dict: Dict[str, torch.Tensor], Dictionary of scales from QERA
        calibration_inputs: List[torch.Tensor], Input tensors for calibration error computation
        log_calibration_error: bool, Whether to compute and log calibration error
        layer_name: str, Name of the layer for logging purposes
    """
    
    if scale_important_weights:
        # Get 1% of largest activations
        metric = activations.scaler_row * module.weight.data.abs().sum(dim=0)
        important_weights = metric.topk(
            int(0.01 * metric.numel()), largest=True, sorted=False)[1]
    else:
        important_weights = None

    # Original weight
    original_weight = module.weight.data.clone()
    
    # Compute the compressed weight (pruned + optionally quantized)
    if slim_lora and not any(activations.scaler_row == 0):
        if quantizer is None:
            W_metric = module.weight.data * (torch.sqrt(activations.scaler_row.reshape((1, -1))))
            new_weight = W_metric.clone().detach()
            new_weight[W_mask] = 0
        else:
            W_metric = module.weight.data * (torch.sqrt(activations.scaler_row.reshape((1, -1))))
            new_weight = module.weight.data
            if quantize_first:
                new_weight = quantizer.quantize_weight(new_weight, important_weights)
                new_weight = quantizer.dequantize_absmax(new_weight) * (torch.sqrt(activations.scaler_row.reshape((1, -1))))
                new_weight[W_mask] = 0            
            else:
                new_weight[W_mask] = 0
                new_weight = quantizer.quantize_weight(new_weight, important_weights)
                new_weight = quantizer.dequantize_absmax(new_weight) * (torch.sqrt(activations.scaler_row.reshape((1, -1))))
            
            # For SLiM-LoRA, we need to unscale to get the actual compressed weight
            denom = (torch.sqrt(activations.scaler_row.reshape((1, -1))))
            compressed_weight = new_weight / denom
    else:
        compressed_weight = module.weight.data.clone().detach()
        if quantize_first:
            if quantizer is not None:
                compressed_weight = quantizer.quantize_weight(compressed_weight, important_weights)
                compressed_weight = quantizer.dequantize_absmax(compressed_weight)     
            compressed_weight[W_mask] = 0
        else:
            compressed_weight[W_mask] = 0
            if quantizer is not None:
                compressed_weight = quantizer.quantize_weight(compressed_weight, important_weights)
                compressed_weight = quantizer.dequantize_absmax(compressed_weight)

    # Choose which algorithm to use for LoRA computation
    if use_qera and qera_scale_dict is not None:
        # Use QERA algorithm
        qera_scales = None
        if hasattr(module, '_module_name') and module._module_name in qera_scale_dict:
            qera_scales = qera_scale_dict[module._module_name]
        
        if qera_scales is not None:
            print(f"Using QERA scales for {module._module_name}")
            lora_left, lora_right = compute_qera_lora(
                original_weight, compressed_weight, qera_scales, rank_ratio
            )
        else:
            print(f"No QERA scales found for {module._module_name}")
            # Fallback to regular SVD if no QERA scales found
            residual = original_weight - compressed_weight
            lora_left, lora_right = compute_svd_lora(residual, rank_ratio)
    else:
        # Use SLiM's algorithm (potentially with activation scaling)
        if slim_lora and not any(activations.scaler_row == 0):
            # SLiM's activation-aware approach
            W_metric = original_weight * (torch.sqrt(activations.scaler_row.reshape((1, -1))))
            compressed_scaled = compressed_weight * (torch.sqrt(activations.scaler_row.reshape((1, -1))))
            error_mat = W_metric - compressed_scaled
            
            # SVD on scaled error
            lora_left, lora_right = compute_svd_lora(error_mat, rank_ratio)
            
            # Unscale the LoRA matrices
            denom = (torch.sqrt(activations.scaler_row.reshape((1, -1)))).to(torch.bfloat16)
            lora_left = lora_left / (denom.t())
        else:
            # Standard SVD approach
            residual = original_weight - compressed_weight
            lora_left, lora_right = compute_svd_lora(residual, rank_ratio)

    # Apply LoRA tile size constraints if needed
    if lora_tile_size is not None:
        rank = lora_left.shape[1]
        tile_dim = lora_tile_size
        residue = rank % tile_dim
        if residue != 0:
            new_rank = rank + (tile_dim - residue)
            # Pad LoRA matrices to meet tile constraints
            lora_left = torch.cat([lora_left, torch.zeros(lora_left.shape[0], new_rank - rank, dtype=lora_left.dtype, device=lora_left.device)], dim=1)
            lora_right = torch.cat([lora_right, torch.zeros(new_rank - rank, lora_right.shape[1], dtype=lora_right.dtype, device=lora_right.device)], dim=0)

    # Apply pruning to LoRA matrices if requested
    if prune_lora and separate_lora:
        lora_left_mask = prune_and_optimize_lora(lora_left, lora_right)

    # Store LoRA matrices in the module
    if separate_lora:
        module.lora_left = torch.nn.Parameter(lora_left.to(torch.bfloat16).contiguous())
        module.lora_right = torch.nn.Parameter(lora_right.to(torch.bfloat16).contiguous())
        module.weight.data = compressed_weight.to(torch.bfloat16).contiguous()
        if prune_lora:
            module.lora_left_mask = lora_left_mask
    else:
        # Merge LoRA back into weights
        low_rank_weight = lora_right.t() @ lora_left.t()
        module.weight.data = (compressed_weight + low_rank_weight).to(torch.bfloat16)

    # Compute and log calibration error if requested
    if log_calibration_error and calibration_inputs is not None:
        calibration_error_stats = compute_calibration_error(
            module=module,
            original_weight=original_weight,
            compressed_weight=compressed_weight.to(torch.bfloat16),
            lora_left=lora_left.to(torch.bfloat16),
            lora_right=lora_right.to(torch.bfloat16),
            calibration_inputs=calibration_inputs,
            layer_name=layer_name or "unknown",
            separate_lora=separate_lora
        )
        # Store calibration error stats in the module for later access
        module.calibration_error_stats = calibration_error_stats
        
        # Add to global list for CSV export
        if calibration_error_stats:
            CALIBRATION_ERRORS.append(calibration_error_stats)
        
        return calibration_error_stats
    
    return None


def clear_calibration_errors():
    """Clear the global calibration errors list"""
    global CALIBRATION_ERRORS
    CALIBRATION_ERRORS = []


def get_calibration_errors():
    """Get the current calibration errors"""
    return CALIBRATION_ERRORS.copy()


def export_calibration_errors_to_csv(csv_path):
    """Export accumulated calibration errors to CSV"""
    if CALIBRATION_ERRORS:
        save_calibration_errors_to_csv(CALIBRATION_ERRORS, csv_path)
    else:
        logger.warning("No calibration errors to export")


def compute_qera_lora(original_weight, compressed_weight, qera_scales, rank_ratio):
    """
    Compute LoRA matrices using QERA algorithm
    
    Args:
        original_weight: Original uncompressed weight matrix [out_dim, in_dim]
        compressed_weight: Compressed (pruned/quantized) weight matrix [out_dim, in_dim] 
        qera_scales: QERA scales - either diagonal vector [in_dim] or full matrix [in_dim, in_dim]
        rank_ratio: Rank ratio for low-rank approximation
        
    Returns:
        lora_left: Left LoRA matrix [out_dim, rank]
        lora_right: Right LoRA matrix [rank, in_dim]
    """
    device = original_weight.device
    dtype = torch.float32
    
    # Convert to float32 for computation and move to same device
    original_weight = original_weight.to(device=device, dtype=dtype)
    compressed_weight = compressed_weight.to(device=device, dtype=dtype)
    # Move scales from CPU to compute device if needed
    qera_scales = qera_scales.to(device=device, dtype=dtype)

    # QERA algorithm:
    # 1. Compute residual
    residual = original_weight - compressed_weight
    
    # 2. Scale the residual
    if qera_scales.ndim == 1:
        # Diagonal case: use element-wise scaling
        scale_matrix = torch.diag(qera_scales)
        residual_scaled = residual @ scale_matrix
        scale_inv = torch.diag(1.0 / qera_scales)
    else:
        # RXX case: full matrix scaling
        residual_scaled = residual @ qera_scales
        scale_inv = torch.inverse(qera_scales)
    
    # 3. SVD on scaled residual
    L, R_ = _low_rank_decomposition_qera(residual_scaled, int(rank_ratio * min(residual.shape)))
    
    # 4. Unscale R
    R = (scale_inv @ R_.T).T
    
    # 5. Return as LoRA matrices in the correct format for the hook
    # Hook expects: input @ lora_left @ lora_right
    # So lora_left should be [in_features, rank] and lora_right should be [rank, out_features]
    lora_left = R.T  # [rank, in_dim] -> [in_dim, rank]
    lora_right = L.T  # [out_dim, rank] -> [rank, out_dim]
    
    return lora_left.to(torch.bfloat16), lora_right.to(torch.bfloat16)


def compute_svd_lora(error_matrix, rank_ratio):
    """
    Compute LoRA matrices using standard SVD
    
    Args:
        error_matrix: Error matrix to decompose [out_dim, in_dim]
        rank_ratio: Rank ratio for low-rank approximation
        
    Returns:
        lora_left: Left LoRA matrix [out_dim, rank] (corresponds to LoRA B)
        lora_right: Right LoRA matrix [rank, in_dim] (corresponds to LoRA A)
    """
    # Use SVD on the error matrix to find the best low-rank approximation
    U, S, V = torch.svd(error_matrix.float())
    
    rank = int(rank_ratio * min(error_matrix.shape))
    
    # Standard LoRA decomposition: error_matrix ≈ lora_left @ lora_right
    # Hook expects: input @ lora_left @ lora_right
    # So lora_left should be [in_features, rank] and lora_right should be [rank, out_features]
    # Where error_matrix = U @ diag(S) @ V^T
    U_rank = U[:, :rank] @ torch.diag(S[:rank])  # [out_dim, rank]
    V_rank = V[:, :rank].T  # [rank, in_dim]
    
    # Transpose to match hook convention
    lora_left = V_rank.T  # [in_dim, rank]
    lora_right = U_rank.T  # [rank, out_dim]
    
    return lora_left.to(torch.bfloat16), lora_right.to(torch.bfloat16)


@torch.no_grad()
def _low_rank_decomposition_qera(x: torch.Tensor, reduced_rank: int):
    """QERA's SVD decomposition"""
    assert x.ndim == 2
    # Use full_matrices=False for better accuracy according to QERA
    U, S, Vh = torch.linalg.svd(x, full_matrices=False)
    L = U @ torch.diag(S)[:, :reduced_rank]
    R = Vh[:reduced_rank, :]
    return L, R


def compute_calibration_error(module, original_weight, compressed_weight, lora_left, lora_right, 
                            calibration_inputs, layer_name="", separate_lora=True):
    """
    Compute calibration error: output difference between original weights and compressed weights + LoRA
    
    Args:
        module: The linear module
        original_weight: Original uncompressed weight matrix [out_dim, in_dim]
        compressed_weight: Compressed (pruned/quantized) weight matrix [out_dim, in_dim]
        lora_left: Left LoRA matrix [in_dim, rank]
        lora_right: Right LoRA matrix [rank, out_dim] 
        calibration_inputs: List of input tensors to use for calibration [batch_size, ..., in_dim]
        layer_name: Name of the layer for logging
        separate_lora: Whether LoRA is stored separately or merged
        
    Returns:
        dict: Dictionary containing error metrics
    """
    if calibration_inputs is None or len(calibration_inputs) == 0:
        logger.warning(f"No calibration inputs provided for layer {layer_name}")
        return {}
    
    device = original_weight.device
    dtype = original_weight.dtype
    
    # Convert inputs to tensors if needed and move to device
    if not isinstance(calibration_inputs, list):
        calibration_inputs = [calibration_inputs]
    
    errors = []
    relative_errors = []
    
    with torch.no_grad():
        for i, input_tensor in enumerate(calibration_inputs):
            # Ensure input is on correct device and reshape for matrix multiplication
            input_tensor = input_tensor.to(device=device, dtype=dtype)
            original_shape = input_tensor.shape
            
            # Original output
            original_output = F.linear(input_tensor, original_weight, None) 
            
            # Compressed + LoRA output
            if separate_lora:

                compressed_output = F.linear(input_tensor, compressed_weight, None)

                lora_intermediate = F.linear(input_tensor, lora_left.t(), None)
                lora_output = F.linear(lora_intermediate, lora_right.t(), None)
                combined_output = compressed_output + lora_output
            else:
                # Merged case: LoRA already added to compressed_weight
                combined_output = F.linear(input_tensor, compressed_weight.t(), None)
            
            # Compute error metrics
            error = torch.norm(original_output - combined_output, p='fro')
            original_norm = torch.norm(original_output, p='fro')
            
            if original_norm > 0:
                relative_error = error / original_norm
            else:
                relative_error = error
                
            errors.append(error.item())
            relative_errors.append(relative_error.item())
    
    # Compute statistics
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    mean_relative_error = np.mean(relative_errors)
    max_relative_error = np.max(relative_errors)
    
    # Log the results
    logger.info(f"Calibration Error for {layer_name}: Mean Rel Error: {mean_relative_error:.6f}")
    
    return {
        'layer_name': layer_name,
        'mean_absolute_error': mean_error,
        'max_absolute_error': max_error,
        'mean_relative_error': mean_relative_error,
        'max_relative_error': max_relative_error,
        'num_calibration_samples': len(calibration_inputs)
    }


