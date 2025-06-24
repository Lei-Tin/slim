import torch
import torch.nn as nn
from .sparsegpt import SparseGPT
from .sparsegpt import Quantizer as SparseGPTQuantizer
from .layerwrapper import WrappedGPT
from .data import get_loaders
from .utils import get_layers_list, shift_zeros, find_layers, prune_nm
from .lora import add_lora, register_scale_hooks, compute_calibration_error, CALIBRATION_ERRORS
from slim.quantization.quantization import Quantizer as AutoQuantizer, QuantizedMatmul
import tqdm.auto as tqdm
from .jsq_utils import clip_matrix, generate_ss
from .smooth import smooth_layer
from huggingface_hub import hf_hub_download
import numpy as np


def prepare_calibration_input(
        model,
        dataloader,
        nsamples=128
):
    """
    Prepare inputs for calibration.

    Args:
        model: torch.nn.Module - The model to calibrate
        dataloader: torch.utils.data.DataLoader - The dataloader to use for calibration

    Returns:
        inps: torch.Tensor - The input tensor for calibration
        outs: torch.Tensor - The output tensor for calibration
        attention_mask: torch.Tensor - The attention mask for calibration
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = get_layers_list(model)


    dtype = next(iter(model.parameters())).dtype
    torch.cuda.empty_cache()
    inps = torch.zeros((nsamples, model.config.max_position_embeddings, model.config.hidden_size), dtype=dtype, device="cpu")
    input_device = "cpu"
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.to(input_device)
            cache['i'] += 1
            for key in kwargs:
                cache[key] = kwargs[key]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0])
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    model.config.use_cache = use_cache
    del cache['i']
    return inps, outs, cache


def prune_magnitude(
        model,
        sparsity_ratio,
        prune_n=0,
        prune_m=0,
        quantize_weight=False,
        bitwidth=4,
        slim_quant=False,
        tiled_weight_quantization=False,
        weight_tile_size=256,
):
    """
    Prune a model using magnitude pruning and quantize weights using SLiM-Quant or AbsMax.

    Args:
        model: torch.nn.Module - The model to prune
        sparsity_ratio: float - The ratio of weights to prune
        prune_n: int - The number N in N:M pruning
        prune_m: int - The number M in N:M pruning
        quantize_weight: bool - Whether to quantize weights
        bitwidth: int - The bitwidth to use for quantization
        slim_quant: bool - Whether to use SLiM-Quant
        tiled_weight_quantization: bool - Whether to use block quantization
        weight_tile_size: int - The size of the blocks for block quantization

    Returns:
        None
    """
    layers = get_layers_list(model)
    progress_bar = tqdm.tqdm(range(len(layers)))

    if quantize_weight:
        quantizer = AutoQuantizer(
            "weight",
            num_bits=bitwidth,
            slim_quant=slim_quant,
            block_quantization=tiled_weight_quantization,
            block_dim=weight_tile_size,
        )
    else:
        quantizer = None

    for i in progress_bar:
        progress_bar.set_description(f"Layer {i}")
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = prune_nm(W_metric, prune_n, prune_m)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel() * sparsity_ratio)].cpu()
                W_mask = (W_metric <= thresh)

            W[W_mask] = 0
            subset[name].weight.data = W
            if quantizer is not None:
                quantized_weight = quantizer.quantize_weight(subset[name].weight.data)
                subset[name].weight.data = quantizer.dequantize_absmax(quantized_weight).to(torch.bfloat16)
                if not tiled_weight_quantization:
                    subset[name].scaling_factor = quantizer.scaling_factor
                else:
                    subset[name].scaling_factor = None


def prune_wanda(
        model,
        tokenizer,
        sparsity_ratio=0.5,
        prune_n=0,
        prune_m=0,
        quantize_weight=False,
        bitwidth=4,
        slim_quant=False,
        tiled_weight_quantization=False,
        weight_tile_size=256,
        shift_zero_metrics=True,
        lora_rank=0.,
        slim_lora=True,
        prune_lora=False,
        quantize_lora=False,
        lora_tile_size=256,
        separate_lora=True,
        nsamples=128,
        seed=0,
        calibration_dataset="c4",
        pad_lora=False,
        quantize_first=True,
        scale_important_weights=False,
        use_qera=False,
        qera_mode="diag",
        qera_sqrtm_implementation="scipy",
        model_type=None,
        log_calibration_error=False,
        calibration_error_samples=10,
):
    """
    Prune a model using WANDA and quantize weights using SLiM-Quant or AbsMax and add low-rank adapter using SLiM or SVD.

    Args:
        model: torch.nn.Module - The model to prune
        tokenizer: transformers.Tokenizer - The tokenizer for the model
        sparsity_ratio: float - The ratio of weights to prune
        prune_n: int - The number N in N:M pruning
        prune_m: int - The number M in N:M pruning
        quantize_weight: bool - Whether to quantize weights
        bitwidth: int - The bitwidth to use for quantization
        slim_quant: bool - Whether to use slim quantization
        tiled_weight_quantization: bool - Whether to use block quantization
        weight_tile_size: int - The size of the blocks for block quantization
        shift_zero_metrics: bool - Whether to shift zero metrics
        lora_rank: float - The rank ratio for LoRA
        slim_lora: bool - Whether to use slim LoRA
        prune_lora: bool - Whether to prune LoRA matrices
        quantize_lora: bool - Whether to quantize LoRA matrices
        lora_tile_size: int - The size of the blocks for LoRA quantization
        separate_lora: bool - Whether to use separate LoRA matrices
        nsamples: int - The number of samples to use for calibration
        seed: int - The seed to use for calibration
        calibration_dataset: str - The dataset to use for calibration
        pad_lora: bool - Whether to pad LoRA matrices
        quantize_first: bool - Whether to quantize before pruning
        scale_important_weights: bool - Whether to scale important weights
        use_qera: bool - Whether to use QERA's L and R matrices
        qera_mode: str - The mode for QERA scaling ("diag" or "rxx")
        qera_sqrtm_implementation: str - The implementation for matrix square root computation ("scipy" or "iterative")
        model_type: str - The type of the model
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    dataloader, _ = get_loaders(
        calibration_dataset,
        nsamples=nsamples,
        seed=seed,
        seqlen=model.config.max_position_embeddings,
        tokenizer=tokenizer
    )

    with torch.no_grad():
        inps, outs, kwargs = prepare_calibration_input(model, dataloader, nsamples)

    if quantize_weight:
        quantizer = AutoQuantizer(
            "weight",
            num_bits=bitwidth,
            slim_quant=slim_quant,
            block_quantization=tiled_weight_quantization,
            block_dim=weight_tile_size,
        )
    else:
        quantizer = None

    layers = get_layers_list(model)

    # Capture module inputs for calibration error computation if requested
    module_inputs_for_calibration = {}
    if log_calibration_error:
        print("Capturing module inputs for calibration error computation...")
        module_inputs_for_calibration = capture_module_inputs_for_calibration(
            model, dataloader, nsamples, calibration_error_samples
        )

    # Initialize QERA scale collection if using QERA with CPU storage for memory efficiency
    qera_hook_factory = None
    qera_scale_dict = None
    
    if use_qera:
        if qera_mode in ["diagonal", "diag"]:
            from .lora import ScaleHookFactoryDiagonal
            # Enable CPU storage to save GPU memory during scale collection
            qera_hook_factory = ScaleHookFactoryDiagonal(torch_dtype=torch.float32, store_on_cpu=True)
        elif qera_mode == "rxx":
            from .lora import ScaleHookFactoryRxx
            # Enable CPU storage to save GPU memory during scale collection
            qera_hook_factory = ScaleHookFactoryRxx(torch_dtype=torch.float32, store_on_cpu=True)
        else:
            raise ValueError(f"Unknown QERA mode: {qera_mode}")
        
        print(f"Registering QERA hooks for all layers individually (no scale sharing)...")
        
        # Register hooks for all linear layers individually (no scale sharing)
        from .lora import find_layers_to_register_scale_hook
        layers_to_register = find_layers_to_register_scale_hook(model)
        
        for layer_info in layers_to_register:
            layer_name = layer_info["target_layer"]
            # Get the actual layer from the model
            target_layer = None
            for name, module in model.named_modules():
                if name == layer_name and isinstance(module, torch.nn.Linear):
                    target_layer = module
                    break
            
            if target_layer is not None:
                handle = target_layer.register_forward_hook(qera_hook_factory.get_scale_hook(layer_name))
                qera_hook_factory.handles.append(handle)
        
        # Run calibration through the entire model to collect QERA scales
        print("Running QERA calibration through entire model (scales stored on CPU)...")
        model = model.cuda()
        for batch in tqdm.tqdm(dataloader, desc="Running QERA calibration through entire model"):
            with torch.no_grad():
                _ = model(batch[0].cuda())
        model = model.cpu()
        torch.cuda.empty_cache()  # Free GPU memory after calibration
        
        # Compute QERA scales for all layers individually (no scale sharing)
        print("Computing QERA scales (stored on CPU)...")
        if qera_mode == "rxx":
            qera_scale_dict = qera_hook_factory.get_scale_dict(
                progress_bar=False, 
                sqrtm_implementation=qera_sqrtm_implementation
            )
        else:
            qera_scale_dict = qera_hook_factory.get_scale_dict(progress_bar=False)
        qera_hook_factory.remove_all_hooks()
        print(f"Computed QERA scales for {len(qera_scale_dict)} layers (stored on CPU for memory efficiency)")
        
        # Clean up calibration data and QERA factory to free memory
        del qera_hook_factory
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    progress_bar = tqdm.tqdm(range(len(layers)))

    for i in progress_bar:
        progress_bar.set_description(f"Layer {i} - Gathering data")
        layer = layers[i].cuda()

        subset = find_layers(layer)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(nsamples):
            for key in kwargs:
                if isinstance(kwargs[key], torch.Tensor):
                    kwargs[key] = kwargs[key].cuda()
                if isinstance(kwargs[key], tuple):
                    kwargs[key] = tuple([k.cuda() for k in kwargs[key]])

            # Process one sample at a time with immediate cleanup
            with torch.no_grad():
                inp_cuda = inps[j].unsqueeze(0).cuda()
                out_cuda = layer(inp_cuda, **kwargs)[0]
                outs[j] = out_cuda.to(outs.device)
            
                # Clean up immediately to prevent accumulation
                del inp_cuda, out_cuda

        for h in handles:
            h.remove()

        progress_bar.set_description(f"Layer {i} - Computing metrics")

        for name in subset:
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            W_mask = torch.zeros_like(W_metric) == 1

            if prune_n != 0:
                W_mask = prune_nm(W_metric, prune_n, prune_m)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                indices = sort_res[1][:, :int(W_metric.shape[1] * sparsity_ratio)]
                W_mask.scatter_(1, indices, True)

            if lora_rank > 0.:
                lora_tile_size = lora_tile_size if (quantize_lora or pad_lora) else None
                
                # Store the module name for QERA matching - use correct prefix for model type
                layer_prefix = "model.model.decoder.layers" if (hasattr(model, 'model') and hasattr(model.model, 'decoder')) else "model.layers"
                full_name = f"{layer_prefix}.{i}.{name}"
                if hasattr(subset[name], '_module_name'):
                    subset[name]._module_name = full_name
                else:
                    setattr(subset[name], '_module_name', full_name)
                
                # Prepare calibration inputs for this module if logging is enabled
                layer_calibration_inputs = None
                if log_calibration_error:
                    # Use the captured module inputs for this specific module
                    if full_name in module_inputs_for_calibration:
                        layer_calibration_inputs = module_inputs_for_calibration[full_name]
                    else:
                        # Fallback: use a subset of layer inputs (should rarely happen)
                        layer_calibration_inputs = []
                        for j in range(min(nsamples, calibration_error_samples)):
                            layer_calibration_inputs.append(inps[j])
                        print(f"Warning: No captured inputs found for {full_name}, using layer inputs as fallback")
                
                add_lora(subset[name],
                         W_mask=W_mask,
                         rank_ratio=lora_rank,
                         slim_lora=slim_lora,
                         activations=wrapped_layers[name],
                         quantizer=quantizer,
                         prune_lora=prune_lora,
                         separate_lora=separate_lora,
                         lora_tile_size=lora_tile_size,
                         quantize_first=quantize_first,
                         scale_important_weights=scale_important_weights,
                         use_qera=use_qera,
                         qera_mode=qera_mode,
                         qera_scale_dict=qera_scale_dict,
                         calibration_inputs=layer_calibration_inputs,
                         log_calibration_error=log_calibration_error,
                         layer_name=full_name
                         )

                if quantizer is not None:
                    if not tiled_weight_quantization:
                        subset[name].scaling_factor = quantizer.scaling_factor
                    else:
                        subset[name].scaling_factor = None

                if separate_lora:
                    def add_lora_hook(module, input, output):
                        if hasattr(module, "lora_quantizer"):
                            xl = QuantizedMatmul.apply(
                                input[0].to(module.lora_left.dtype) / torch.sqrt(module.lora_rank),
                                module.lora_left,
                                module.lora_quantizer
                            )
                            xlr = QuantizedMatmul.apply(
                                xl / torch.sqrt(module.lora_rank),
                                module.lora_right,
                                module.lora_quantizer
                            )
                            output += xlr
                        else:
                            output += torch.matmul(
                                torch.matmul(input[0].to(module.lora_left.dtype),
                                             module.lora_left / torch.sqrt(module.lora_rank)),
                                module.lora_right / torch.sqrt(module.lora_rank))

                    subset[name].lora_rank = torch.tensor(subset[name].lora_left.shape[1])
                    subset[name].lora_left = torch.nn.Parameter(subset[name].lora_left * torch.sqrt(subset[name].lora_rank))
                    subset[name].lora_right = torch.nn.Parameter(subset[name].lora_right * torch.sqrt(subset[name].lora_rank))
                    subset[name].register_forward_hook(add_lora_hook)
            else:
                if scale_important_weights:
                    # Get 1% of largest activations
                    metric = subset[name].scaler_row * subset[name].weight.data.abs().sum(dim=0)
                    important_weights = metric.topk(
                        int(0.01 * metric.numel()), largest=True, sorted=False)[1]
                else:
                    important_weights = None
                if quantize_first:
                    if quantizer is not None:
                        quantized_weight = quantizer.quantize_weight(subset[name].weight.data, important_weights)
                        subset[name].weight.data = quantizer.dequantize_absmax(quantized_weight).to(torch.bfloat16)
                        if quantizer is not None:
                            if not tiled_weight_quantization:
                                subset[name].scaling_factor = quantizer.scaling_factor
                            else:
                                subset[name].scaling_factor = None
                    subset[name].weight.data[W_mask] = 0  ## set weights to zero
                else:
                    subset[name].weight.data[W_mask] = 0  ## set weights to zero
                    if quantizer is not None:
                        quantized_weight = quantizer.quantize_weight(subset[name].weight.data, important_weights)
                        subset[name].weight.data = quantizer.dequantize_absmax(quantized_weight).to(torch.bfloat16)
                        if quantizer is not None:
                            if not tiled_weight_quantization:
                                subset[name].scaling_factor = quantizer.scaling_factor
                            else:
                                subset[name].scaling_factor = None


        progress_bar.set_description(f"Layer {i} - Evaluating Output")

        for j in range(nsamples):
            with torch.no_grad():
                # Move input to GPU, process, then immediately move output to CPU
                inp_cuda = inps[j].unsqueeze(0).cuda()
                out_cuda = layer(inp_cuda, **kwargs)[0]
                outs[j] = out_cuda.to(outs[j].device)
                
                # Clean up GPU tensors immediately
                del inp_cuda, out_cuda
                
        inps, outs = outs, inps

        # Critical memory cleanup after each layer to prevent accumulation
        layers[i] = layer.cpu()
        del layer
        
        # Clean up wrapped layers and their accumulated activations
        wrapped_layers.clear()  # Clear all references
        del wrapped_layers
        
        # Clean up subset reference
        del subset
        
        # Force garbage collection and clear GPU cache
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        progress_bar.set_description(f"Layer {i} - Memory cleaned")

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_sparsegpt(
        model, 
        tokenizer,
        sparsity_ratio=0.5,
        prune_n=0, 
        prune_m=0,
        nsamples=128,
        seed=0,
        quantize_weight=False,
        bitwidth=4,
        tiled_weight_quantization=False,
        weight_tile_size=256,
        calibration_dataset="c4"
):
    """
    Prune a model using SparseGPT and quantize weights using OPTQ (GPTQ).
    SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa

    Args:
        model: torch.nn.Module - The model to prune
        tokenizer: transformers.Tokenizer - The tokenizer for the model
        device: torch.device - The device to use for pruning
        sparsity_ratio: float - The ratio of weights to prune
        prune_n: int - The number N in N:M pruning
        prune_m: int - The number M in N:M pruning
        nsamples: int - The number of samples to use for calibration
        seed: int - The seed to use for calibration
        quantize_weight: bool - Whether to quantize weights
        bitwidth: int - The bitwidth to use for quantization
        tiled_weight_quantization: bool - Whether to use block quantization
        weight_tile_size: int - The size of the blocks for block
        calibration_dataset: str - The dataset to use for calibration

    Returns:
        None
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    dataloader, _ = get_loaders(
        calibration_dataset,
        nsamples=nsamples,
        seed=seed,
        seqlen=model.config.max_position_embeddings,
        tokenizer=tokenizer
    )

    with torch.no_grad():
        inps, outs, kwargs = prepare_calibration_input(model, dataloader, nsamples)

    layers = get_layers_list(model)

    progress_bar = tqdm.tqdm(range(len(layers)))

    for i in progress_bar:
        progress_bar.set_description(f"Layer {i} - Gathering data")
        layer = layers[i].cuda()

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])
            if quantize_weight:
                gpts[name].quantizer = SparseGPTQuantizer()
                gpts[name].quantizer.configure(
                    bitwidth,
                    perchannel=tiled_weight_quantization,
                    sym=True,
                    mse=False,
                )

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(nsamples):
            for key in kwargs:
                if isinstance(kwargs[key], torch.Tensor):
                    kwargs[key] = kwargs[key].cuda()
                if isinstance(kwargs[key], tuple):
                    kwargs[key] = tuple([k.cuda() for k in kwargs[key]])

            outs[j] = layer(inps[j].unsqueeze(0).cuda(), **kwargs)[0].to(outs.device)

        for h in handles:
            h.remove()

        for name in gpts:
            progress_bar.set_description(f"Layer {i} - Pruning and Quantizing {name}")
            gpts[name].fasterprune(sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=weight_tile_size)
            if quantize_weight:
                if not tiled_weight_quantization:
                    subset[name].scaling_factor = 1. / gpts[name].quantizer.scale[0]
                else:
                    subset[name].scaling_factor = None
            gpts[name].free()

        progress_bar.set_description(f"Layer {i} - Evaluating Output")
        for j in range(nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0).cuda(), **kwargs)[0].to(outs[j].device)

        layers[i] = layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def quantize_model(
       model,
       bitwidth=4,
       slim_quant=False,
       weight_tiled_quantization=False,
       weight_tile_size=256,
):
    """
    Quantize the model using the AutoQuantizer class.

    Args:
        model: torch.nn.Module - The model to quantize
        bitwidth: int - The bitwidth to quantize the model to
        slim_quant: bool - Use SLiM-Quant
        weight_tiled_quantization: bool - Use block quantization
        weight_tile_size: int - The size of the block for block quantization

    Returns:
        None
    """
    quantizer = AutoQuantizer(
        "weight",
        num_bits=bitwidth,
        slim_quant=slim_quant,
        block_quantization=weight_tiled_quantization,
        block_dim=weight_tile_size,
    )
    layers = get_layers_list(model)

    progress_bar = tqdm.tqdm(range(len(layers)))
    
    for i in progress_bar:
        layer = layers[i]

        subset = find_layers(layer)
        
        for name in subset:
            progress_bar.set_description(f"Layer {i} - Quantizing {name}")

            quantized_weight = quantizer.dequantize_absmax(
                quantizer.quantize_weight(subset[name].weight.data)
            )
            
            subset[name].weight.data = quantized_weight.to(subset[name].weight.dtype)


def joint_pq(
        model,
        tokenizer,
        prune_n=0,
        prune_m=0,
        nsamples=128,
        bitwidth=4,
        sparsity_ratio=0.5,
        weight_tile_size=256,
        mixing_factor=2.1,
        seed=0,
        calibration_dataset="c4",
        lora_rank=0.,
        slim_lora=True,
        prune_lora=False,
        quantize_lora=False,
        lora_tile_size=256,
        separate_lora=True,
        quantize_first=True, 
        pad_lora=False,
        scale_important_weights=False,
        use_qera=False,
        qera_mode="diag",
        qera_sqrtm_implementation="scipy",
        model_type=None,
        log_calibration_error=False,
        calibration_error_samples=10,
):
    """
    Prune and quantize a model using joint pruning and quantization.
    
    Args:
        model: torch.nn.Module - The model to prune and quantize
        tokenizer: transformers.Tokenizer - The tokenizer for the model
        prune_n: int - The number N in N:M pruning
        prune_m: int - The number M in N:M pruning
        nsamples: int - The number of samples to use for calibration
        bitwidth: int - The bitwidth to use for quantization
        sparsity_ratio: float - The ratio of weights to prune
        weight_tile_size: int - The size of the blocks for block quantization
        mixing_factor: float - The mixing factor for joint pruning and quantization
        seed: int - The seed to use for calibration
        calibration_dataset: str - The dataset to use for calibration
        lora_rank: float - The rank ratio for LoRA
        slim_lora: bool - Whether to use slim LoRA
        prune_lora: bool - Whether to prune LoRA matrices
        quantize_lora: bool - Whether to quantize LoRA matrices
        lora_tile_size: int - The size of the blocks for LoRA quantization
        separate_lora: bool - Whether to use separate LoRA matrices
        quantize_first: bool - Whether to quantize before pruning
        pad_lora: bool - Whether to pad LoRA matrices
        scale_important_weights: bool - Whether to scale important weights
        use_qera: bool - Whether to use QERA's L and R matrices
        qera_mode: str - The mode for QERA scaling ("diag" or "rxx")
        model_type: str - The type of the model
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    dataloader, _ = get_loaders(
        calibration_dataset,
        nsamples=nsamples,
        seed=seed,
        seqlen=model.config.max_position_embeddings,
        tokenizer=tokenizer
    )

    with torch.no_grad():
        inps, outs, kwargs = prepare_calibration_input(model, dataloader, nsamples)

    quantizer = AutoQuantizer(
        "weight",
        num_bits=bitwidth,
        slim_quant=False,
        block_quantization=True,
        block_dim=weight_tile_size,
    )

    layers = get_layers_list(model)

    # Capture module inputs for calibration error computation if requested
    module_inputs_for_calibration = {}
    if log_calibration_error:
        print("Capturing module inputs for calibration error computation...")
        module_inputs_for_calibration = capture_module_inputs_for_calibration(
            model, dataloader, nsamples, calibration_error_samples
        )

    # Auto-detect model type if not specified
    if model_type is None:
        model_type = model.config.model_type if hasattr(model.config, 'model_type') else 'unknown'
    
    print(f"Registering QERA hooks for {model_type} model with proper scale sharing...")
    
    # Use the simplified global configuration approach
    from .lora import register_qera_hooks_with_sharing
    scale_sharing_map = register_qera_hooks_with_sharing(model, quantizer, model_type)

    progress_bar = tqdm.tqdm(range(len(layers)))

    for i in progress_bar:
        progress_bar.set_description(f"Layer {i} - Gathering data")
        layer = layers[i].cuda()

        subset = find_layers(layer)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        progress_bar.set_description(f"Layer {i} - Gathering data")
        act_scales = {}

        layer_name = f'model.layers.{i}'

        def stat_tensor(name, tensor):
            hidden_dim = tensor.shape[-1]
            tensor = tensor.view(-1, hidden_dim).abs().detach()
            comming_max = torch.max(tensor, dim=0)[0].float().cpu()
            
            layer_name = f"model.layers.{i}"
            full_name = layer_name + '.' + name

            if full_name in act_scales:
                act_scales[full_name] = torch.max(act_scales[full_name], comming_max)
            else:
                act_scales[full_name] = comming_max

        def add_batch(name):
            def tmp(_, inp, out):
                inp = clip_matrix(inp[0].data, True, 0, 1e-2)
                stat_tensor(name, inp)
                wrapped_layers[name].add_batch(inp, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(nsamples):
            for key in kwargs:
                if isinstance(kwargs[key], torch.Tensor):
                    kwargs[key] = kwargs[key].cuda()
                if isinstance(kwargs[key], tuple):
                    kwargs[key] = tuple([k.cuda() for k in kwargs[key]])

            outs[j] = layer(inps[j].unsqueeze(0).cuda(), **kwargs)[0].to(outs.device)

        for h in handles:
            h.remove()

        progress_bar.set_description(f"Layer {i} - Computing metrics")

        for name in subset:
            ss = generate_ss(wrapped_layers[name].inp_sum / wrapped_layers[name].inp_num, subset[name].weight.data)
            weight = torch.abs(subset[name].weight.data)
            activation = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            W_metric = weight * activation
            W_metric = W_metric + mixing_factor * ss
            W_mask = torch.zeros_like(W_metric) == 1

            if prune_n != 0:
                W_mask = prune_nm(W_metric, prune_n, prune_m)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                # unstructured pruning
                indices = sort_res[1][:, :int(W_metric.shape[1] * sparsity_ratio)]
                W_mask.scatter_(1, indices, True)

            if lora_rank > 0.:
                lora_tile_size = lora_tile_size if (quantize_lora or pad_lora) else None
                
                # Store the module name for QERA matching - use correct prefix for model type
                layer_prefix = "model.model.decoder.layers" if (hasattr(model, 'model') and hasattr(model.model, 'decoder')) else "model.layers"
                full_name = f"{layer_prefix}.{i}.{name}"
                if hasattr(subset[name], '_module_name'):
                    subset[name]._module_name = full_name
                else:
                    setattr(subset[name], '_module_name', full_name)
                
                # Prepare calibration inputs for this module if logging is enabled
                layer_calibration_inputs = None
                if log_calibration_error:
                    # Use the captured module inputs for this specific module
                    if full_name in module_inputs_for_calibration:
                        layer_calibration_inputs = module_inputs_for_calibration[full_name]
                    else:
                        # Fallback: use a subset of layer inputs (should rarely happen)
                        layer_calibration_inputs = []
                        for j in range(min(nsamples, calibration_error_samples)):
                            layer_calibration_inputs.append(inps[j])
                        print(f"Warning: No captured inputs found for {full_name}, using layer inputs as fallback")
                
                add_lora(subset[name],
                         W_mask=W_mask,
                         rank_ratio=lora_rank,
                         slim_lora=slim_lora,
                         activations=wrapped_layers[name],
                         quantizer=quantizer,
                         prune_lora=prune_lora,
                         separate_lora=separate_lora,
                         lora_tile_size=lora_tile_size,
                         quantize_first=quantize_first,
                         scale_important_weights=scale_important_weights,
                         use_qera=use_qera,
                         qera_mode=qera_mode,
                         qera_scale_dict=scale_sharing_map,
                         calibration_inputs=layer_calibration_inputs,
                         log_calibration_error=log_calibration_error,
                         layer_name=full_name
                         )

                if quantizer is not None:
                    subset[name].scaling_factor = None

                if separate_lora:
                    def add_lora_hook(module, input, output):
                        if hasattr(module, "lora_quantizer"):
                            xl = QuantizedMatmul.apply(
                                input[0].to(module.lora_left.dtype) / torch.sqrt(module.lora_rank),
                                module.lora_left,
                                module.lora_quantizer
                            )
                            xlr = QuantizedMatmul.apply(
                                xl / torch.sqrt(module.lora_rank),
                                module.lora_right,
                                module.lora_quantizer
                            )
                            output += xlr
                        else:
                            output += torch.matmul(
                                torch.matmul(input[0].to(module.lora_left.dtype),
                                             module.lora_left / torch.sqrt(module.lora_rank)),
                                module.lora_right / torch.sqrt(module.lora_rank))

                    subset[name].lora_rank = torch.tensor(subset[name].lora_left.shape[1])
                    subset[name].lora_left = torch.nn.Parameter(subset[name].lora_left * torch.sqrt(subset[name].lora_rank))
                    subset[name].lora_right = torch.nn.Parameter(subset[name].lora_right * torch.sqrt(subset[name].lora_rank))
                    subset[name].register_forward_hook(add_lora_hook)
                
                # Zero out pruned weights after LoRA decomposition
                subset[name].weight.data[W_mask] = 0
            else:
                if scale_important_weights:
                    # Get 1% of largest activations
                    metric = subset[name].scaler_row * subset[name].weight.data.abs().sum(dim=0)
                    important_weights = metric.topk(
                        int(0.01 * metric.numel()), largest=True, sorted=False)[1]
                else:
                    important_weights = None
                if quantize_first:
                    if quantizer is not None:
                        quantized_weight = quantizer.quantize_weight(subset[name].weight.data, important_weights)
                        subset[name].weight.data = quantizer.dequantize_absmax(quantized_weight).to(torch.bfloat16)
                        if quantizer is not None:
                            subset[name].scaling_factor = None
                    subset[name].weight.data[W_mask] = 0
                else:
                    subset[name].weight.data[W_mask] = 0
                    if quantizer is not None:
                        quantized_weight = quantizer.quantize_weight(subset[name].weight.data, important_weights)
                        subset[name].weight.data = quantizer.dequantize_absmax(quantized_weight).to(torch.bfloat16)
                        if quantizer is not None:
                            subset[name].scaling_factor = None

        # Apply smoothing after pruning for all layers
        for j in range(nsamples):
            with torch.no_grad():
                for key in kwargs:
                    if isinstance(kwargs[key], torch.Tensor):
                        kwargs[key] = kwargs[key].cuda()
                    if isinstance(kwargs[key], tuple):
                        kwargs[key] = tuple([k.cuda() for k in kwargs[key]])
                outs[j] = layer(inps[j].unsqueeze(0).cuda(), **kwargs)[0].to(outs.device)

        progress_bar.set_description(f"Layer {i} - Smoothing")
        smooth_layer(layer_name, layer, act_scales, 0.5)

        progress_bar.set_description(f"Layer {i} - Evaluating Output")

        for j in range(nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0).cuda(), **kwargs)[0].to(outs[j].device)
        inps, outs = outs, inps

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def prune_and_quantize(
        model,
        tokenizer,
        bitwidth=4,
        slim_quant=True,
        weight_tiled_quantization=False,
        weight_tile_size=256,
        prune_method="wanda",
        sparsity_ratio=0.5,
        sparsity_type="2:4",
        quantize_weight=False,
        nsamples=128,
        shift_zero_metrics=True,
        lora_rank=0.,
        slim_lora=True,
        prune_lora=False,
        quantize_lora=False,
        lora_tile_size=256,
        separate_lora=True,
        seed=0,
        joint_pq_mixing_factor=2.1,
        calibration_dataset="c4",
        pad_lora=False,
        scale_important_weights=False,
        mask_checkpoint=None,
        use_qera=False,
        qera_mode="diag",
        qera_sqrtm_implementation="scipy",
        model_type=None,
        log_calibration_error=False,
        calibration_error_samples=10,
):
    """
    Prune and quantize a model using various methods.

    Args:
        model: torch.nn.Module - The model to prune and quantize
        tokenizer: transformers.Tokenizer - The tokenizer for the model
        bitwidth: int - The bitwidth to use for quantization
        slim_quant: bool - Whether to use slim quantization
        weight_tiled_quantization: bool - Whether to use block quantization
        weight_tile_size: int - The size of the blocks for block quantization
        prune_method: str - The pruning method to use ("wanda", "magnitude", "sparsegpt", "joint_pq")
        sparsity_ratio: float - The ratio of weights to prune
        sparsity_type: str - The type of sparsity ("2:4", "4:8", etc.)
        quantize_weight: bool - Whether to quantize weights
        nsamples: int - The number of samples to use for calibration
        shift_zero_metrics: bool - Whether to shift zero metrics
        lora_rank: float - The rank ratio for LoRA
        slim_lora: bool - Whether to use slim LoRA
        prune_lora: bool - Whether to prune LoRA matrices
        quantize_lora: bool - Whether to quantize LoRA matrices
        lora_tile_size: int - The size of the blocks for LoRA quantization
        separate_lora: bool - Whether to use separate LoRA matrices
        seed: int - The seed to use for calibration
        joint_pq_mixing_factor: float - The mixing factor for joint pruning and quantization
        calibration_dataset: str - The dataset to use for calibration
        pad_lora: bool - Whether to pad LoRA matrices
        scale_important_weights: bool - Whether to scale important weights
        mask_checkpoint: str - Path to a checkpoint containing masks
        use_qera: bool - Whether to use QERA's L and R matrices
        qera_mode: str - The mode for QERA scaling ("diag" or "rxx")
        qera_sqrtm_implementation: str - The implementation for matrix square root computation ("scipy" or "iterative")
        model_type: str - The type of the model
    """
    if sparsity_type != "unstructured":
        prune_n, prune_m = map(int, sparsity_type.split(":"))
    else:
        prune_n, prune_m = 0, 0

    if prune_method == "wanda":
        prune_wanda(
            model,
            tokenizer,
            sparsity_ratio=sparsity_ratio,
            prune_n=prune_n,
            prune_m=prune_m,
            quantize_weight=quantize_weight,
            bitwidth=bitwidth,
            slim_quant=slim_quant,
            tiled_weight_quantization=weight_tiled_quantization,
            weight_tile_size=weight_tile_size,
            shift_zero_metrics=shift_zero_metrics,
            lora_rank=lora_rank,
            slim_lora=slim_lora,
            prune_lora=prune_lora,
            quantize_lora=quantize_lora,
            lora_tile_size=lora_tile_size,
            separate_lora=separate_lora,
            nsamples=nsamples,
            seed=seed,
            calibration_dataset=calibration_dataset,
            pad_lora=pad_lora,
            scale_important_weights=scale_important_weights,
            use_qera=use_qera,
            qera_mode=qera_mode,
            qera_sqrtm_implementation=qera_sqrtm_implementation,
            model_type=model_type,
            log_calibration_error=log_calibration_error,
            calibration_error_samples=calibration_error_samples,
        )
    elif prune_method == "magnitude":
        prune_magnitude(
            model,
            sparsity_ratio=sparsity_ratio,
            prune_n=prune_n,
            prune_m=prune_m,
            quantize_weight=quantize_weight,
            bitwidth=bitwidth,
            slim_quant=slim_quant,
            tiled_weight_quantization=weight_tiled_quantization,
            weight_tile_size=weight_tile_size,
        )
    elif prune_method == "sparsegpt":
        prune_sparsegpt(
            model,
            tokenizer,
            sparsity_ratio=sparsity_ratio,
            prune_n=prune_n,
            prune_m=prune_m,
            nsamples=nsamples,
            seed=seed,
            quantize_weight=quantize_weight,
            bitwidth=bitwidth,
            tiled_weight_quantization=weight_tiled_quantization,
            weight_tile_size=weight_tile_size,
            calibration_dataset=calibration_dataset,
        )
    elif prune_method == "joint_pq":
        joint_pq(
            model,
            tokenizer,
            prune_n=prune_n,
            prune_m=prune_m,
            nsamples=nsamples,
            bitwidth=bitwidth,
            sparsity_ratio=sparsity_ratio,
            weight_tile_size=weight_tile_size,
            mixing_factor=joint_pq_mixing_factor,
            seed=seed,
            calibration_dataset=calibration_dataset,
            lora_rank=lora_rank,
            slim_lora=slim_lora,
            prune_lora=prune_lora,
            quantize_lora=quantize_lora,
            lora_tile_size=lora_tile_size,
            separate_lora=separate_lora,
            quantize_first=True,
            pad_lora=pad_lora,
            scale_important_weights=scale_important_weights,
            use_qera=use_qera,
            qera_mode=qera_mode,
            qera_sqrtm_implementation=qera_sqrtm_implementation,
            model_type=model_type,
            log_calibration_error=log_calibration_error,
            calibration_error_samples=calibration_error_samples,
        )
    else:
        raise ValueError(f"Invalid prune method: {prune_method}")


def capture_module_inputs_for_calibration(
        model,
        dataloader,
        nsamples=128,
        max_calibration_samples=10
):
    """
    Capture inputs to individual linear modules during calibration forward pass.
    This captures the actual inputs that each linear module receives during the forward pass.
    
    Args:
        model: torch.nn.Module - The model
        dataloader: torch.utils.data.DataLoader - The dataloader
        nsamples: int - Number of samples to use for calibration
        max_calibration_samples: int - Maximum number of samples to store for calibration error computation
        
    Returns:
        Dict[str, List[torch.Tensor]] - Dictionary mapping module names to lists of input tensors
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    # Dictionary to store inputs for each module
    module_inputs = {}
    module_handles = []
    
    def create_input_capture_hook(module_name):
        def hook(module, input, output):
            if module_name not in module_inputs:
                module_inputs[module_name] = []
            
            # Only store up to max_calibration_samples to save memory
            if len(module_inputs[module_name]) < max_calibration_samples:
                # Store the input tensor on CPU to save GPU memory
                input_tensor = input[0].detach().cpu()
                module_inputs[module_name].append(input_tensor)
        return hook
    
    # Register hooks on all linear modules
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Skip embedding and output layers
            if any(skip in name.lower() for skip in ['embed', 'lm_head', 'head']):
                continue
            handle = module.register_forward_hook(create_input_capture_hook(name))
            module_handles.append(handle)
    
    # Run forward pass to capture inputs
    model = model.cuda()
    sample_count = 0
    for batch in dataloader:
        if sample_count >= nsamples:
            break
        with torch.no_grad():
            _ = model(batch[0].cuda())
        sample_count += batch[0].shape[0]
    
    # Remove all hooks
    for handle in module_handles:
        handle.remove()
    
    model.config.use_cache = use_cache
    model = model.cpu()
    torch.cuda.empty_cache()
    
    print(f"Captured inputs for {len(module_inputs)} modules with up to {max_calibration_samples} samples each")
    
    return module_inputs