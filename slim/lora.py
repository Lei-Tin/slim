import torch
from slim.quantization.quantization import Quantizer as AutoQuantizer
import tqdm.auto as tqdm
from .utils import prune_nm, get_layers_list, find_layers
from typing import Optional, Dict, Any
import math


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
    def get_scale_dict(self, progress_bar=False, scale_sharing_map=None) -> dict[str, torch.Tensor]:
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

        # Apply scale sharing if provided
        if scale_sharing_map:
            for target_layer, layers_sharing_scale in scale_sharing_map.items():
                if target_layer in self.scales and self.scales[target_layer] is not None:
                    target_scale = self.scales[target_layer]
                    for shared_layer in layers_sharing_scale:
                        if shared_layer in self.scales:
                            self.scales[shared_layer] = target_scale.clone()

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
    def get_scale_dict(self, progress_bar=False, sqrtm_implementation: str = "scipy", sqrtm_num_iters: int = 200, scale_sharing_map=None) -> dict[str, torch.Tensor]:
        scale_names_prog_bar = tqdm.tqdm(
            self.scales, desc="Computing RXX scale", disable=not progress_bar, total=len(self.scales)
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
                
                # Compute matrix square root
                if sqrtm_implementation == "scipy":
                    import scipy.linalg as spla
                    scale_np = scale.cpu().numpy()
                    scale_sqrt_np = spla.sqrtm(scale_np).real
                    scale_sqrt = torch.from_numpy(scale_sqrt_np).to(device=compute_device, dtype=torch.float32)
                else:
                    # Use iterative Newton-Schulz method
                    scale_sqrt = sqrtm_newton_schulz(scale.unsqueeze(0), numIters=sqrtm_num_iters).squeeze(0)
                    scale_sqrt = scale_sqrt.to(torch.float32)
                
                # Store final scale on CPU to save memory
                if self.store_on_cpu:
                    self.scales[name] = scale_sqrt.cpu()
                else:
                    self.scales[name] = scale_sqrt

        # Apply scale sharing if provided
        if scale_sharing_map:
            for target_layer, layers_sharing_scale in scale_sharing_map.items():
                if target_layer in self.scales and self.scales[target_layer] is not None:
                    target_scale = self.scales[target_layer]
                    for shared_layer in layers_sharing_scale:
                        if shared_layer in self.scales:
                            self.scales[shared_layer] = target_scale.clone()

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


# Global configuration for scale sharing by model type
OPT_SCALE_SHARING_CONFIG = {
    # For OPT models, k_proj shares scales with q_proj and v_proj
    "scale_sharing_patterns": [
        {
            "target_layer_pattern": "self_attn.k_proj",
            "shared_layers_patterns": ["self_attn.q_proj", "self_attn.v_proj"]
        }
    ]
}

# Model type configurations
MODEL_SCALE_SHARING_CONFIGS = {
    "opt": OPT_SCALE_SHARING_CONFIG,
    # Add other model types here as needed
}

def create_scale_sharing_map(model, model_type=None):
    """Create scale sharing map based on model type configuration"""
    if model_type is None or model_type not in MODEL_SCALE_SHARING_CONFIGS:
        return {}
    
    config = MODEL_SCALE_SHARING_CONFIGS[model_type]
    scale_sharing_map = {}
    
    # Find all decoder layers
    layers = get_layers_list(model)
    
    # Determine the correct layer prefix based on model structure
    layer_prefix = "model.model.decoder.layers" if hasattr(model, 'model') and hasattr(model.model, 'decoder') else "model.layers"
    
    for i, layer in enumerate(layers):
        subset = find_layers(layer)
        
        # Apply scale sharing patterns
        for pattern in config["scale_sharing_patterns"]:
            target_pattern = pattern["target_layer_pattern"]
            shared_patterns = pattern["shared_layers_patterns"]
            
            # Check if target layer exists in this layer
            if target_pattern in subset:
                target_layer_name = f"{layer_prefix}.{i}.{target_pattern}"
                shared_layer_names = []
                
                # Find all shared layers that exist
                for shared_pattern in shared_patterns:
                    if shared_pattern in subset:
                        shared_layer_names.append(f"{layer_prefix}.{i}.{shared_pattern}")
                
                if shared_layer_names:
                    scale_sharing_map[target_layer_name] = shared_layer_names
    
    return scale_sharing_map


def register_qera_hooks_with_sharing(model, qera_hook_factory, model_type=None):
    """Register QERA hooks with proper scale sharing based on model type"""
    layers = get_layers_list(model)
    layer_prefix = "model.model.decoder.layers" if hasattr(model, 'model') and hasattr(model.model, 'decoder') else "model.layers"
    
    # Create scale sharing map
    scale_sharing_map = create_scale_sharing_map(model, model_type)
    
    # Get all target layers (layers that will actually collect data)
    target_layers = set()
    shared_layers = set()
    
    for target_layer, shared_list in scale_sharing_map.items():
        target_layers.add(target_layer)
        shared_layers.update(shared_list)
    
    # Register hooks for all layers
    for i, layer in enumerate(layers):
        subset = find_layers(layer)
        
        for name in subset:
            full_name = f"{layer_prefix}.{i}.{name}"
            
            # Only register hook if this layer collects its own data
            # (either it's not in shared_layers, or it's a target_layer)
            if full_name not in shared_layers or full_name in target_layers:
                hook = qera_hook_factory.get_scale_hook(full_name)
                handle = subset[name].register_forward_hook(hook)
                qera_hook_factory.handles.append(handle)
            else:
                # Initialize placeholder for shared layers
                qera_hook_factory.scales[full_name] = None
    
    return scale_sharing_map

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
        qera_scale_dict=None
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