import torch
import time
import numpy as np
import os
import yaml
from pathlib import Path

# Load config first
config_path = Path("config/base.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Set global config
import mast3r_slam.config as cfg_module
cfg_module.config = config

from mast3r_slam.mast3r_utils import load_mast3r
from mast3r_slam.frame import create_frame
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
import lietorch

def create_test_frame(h=224, w=224, device="cuda"):
    """Create a dummy frame for testing"""
    # Create dummy image data in float format [0, 1] expected by resize_img
    img = np.random.rand(h, w, 3).astype(np.float32)
    # Create an identity pose (expected T_WC argument)
    T_WC = lietorch.Sim3.Identity(1, device=device)
    # create_frame signature: create_frame(i, img, T_WC, img_size=512, device="cuda:0")
    frame = create_frame(0, img, T_WC, img_size=h, device=device)
    return frame

def benchmark_model_inference(model, frame, use_fp8=False, num_runs=10, batch_size=1):
    """Benchmark the actual model forward pass"""
    
    # Create FP8 recipe
    fp8_recipe = te.fp8.DelayedScaling(margin=0, fp8_format=recipe.Format.E4M3)
    
    # For FP8, we need batch size divisible by 8 due to tensor alignment requirements
    if use_fp8 and batch_size == 1:
        # Replicate the frame to create a batch of 8
        img_batch = frame.img.repeat(8, 1, 1, 1)
        true_shape_batch = frame.img_true_shape.repeat(8, 1)
    else:
        img_batch = frame.img
        true_shape_batch = frame.img_true_shape
    
    # Warmup
    for _ in range(3):
        if use_fp8:
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                with torch.no_grad():
                    feat, pos, _ = model._encode_image(img_batch, true_shape_batch)
        else:
            with torch.no_grad():
                feat, pos, _ = model._encode_image(img_batch, true_shape_batch)
        torch.cuda.synchronize()
    
    # Timing
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        if use_fp8:
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                with torch.no_grad():
                    feat, pos, _ = model._encode_image(img_batch, true_shape_batch)
        else:
            with torch.no_grad():
                feat, pos, _ = model._encode_image(img_batch, true_shape_batch)
                
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        # If we used batch of 8, divide by 8 to get per-image time
        if use_fp8 and batch_size == 1:
            elapsed = elapsed / 8
        times.append(elapsed)
    
    return np.mean(times), np.std(times)

def replace_linear_layers_with_te(model, verbose=False):
    """Replace torch.nn.Linear layers with transformer_engine.Linear for FP8 support"""
    import transformer_engine.pytorch as te
    
    replaced_count = 0
    
    def replace_module(module, prefix=''):
        nonlocal replaced_count
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, torch.nn.Linear):
                # Create TE Linear layer with same dimensions
                # Note: TE.Linear doesn't accept dtype parameter
                te_linear = te.Linear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    device=child.weight.device
                )
                # Convert to same dtype as original and copy weights
                te_linear = te_linear.to(dtype=child.weight.dtype)
                with torch.no_grad():
                    te_linear.weight.copy_(child.weight)
                    if child.bias is not None:
                        te_linear.bias.copy_(child.bias)
                
                # Replace the module
                setattr(module, name, te_linear)
                replaced_count += 1
                if verbose:
                    print(f"Replaced {full_name}: Linear({child.in_features}, {child.out_features})")
            else:
                # Recursively replace in child modules
                replace_module(child, full_name)
    
    replace_module(model)
    print(f"✓ Replaced {replaced_count} torch.nn.Linear layers with transformer_engine.Linear")
    return model

if __name__ == "__main__":
    device = "cuda:0"
    
    print("=== FP8 Acceleration Test for MASt3R ===\n")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Compute Capability: {torch.cuda.get_device_capability()}")
    
    # Load model
    print("\nLoading MASt3R model...")
    model = load_mast3r(device=device)
    model.eval()
    
    # Create test frame
    print("Creating test frame...")
    frame = create_test_frame(224, 224, device)
    
    # Test 1: Original model
    print("\n=== Test 1: Original Model ===")
    mean_time_orig, std_time_orig = benchmark_model_inference(model, frame, use_fp8=False)
    print(f"Baseline (FP32): {mean_time_orig:.2f} ± {std_time_orig:.2f} ms")
    
    # Test 2: Original model with FP8 autocast
    print("\n=== Test 2: Original Model + FP8 Autocast ===")
    mean_time_fp8, std_time_fp8 = benchmark_model_inference(model, frame, use_fp8=True)
    print(f"With FP8 autocast: {mean_time_fp8:.2f} ± {std_time_fp8:.2f} ms")
    print(f"Speedup: {mean_time_orig/mean_time_fp8:.2f}x")
    
    # Test 3: Model with FP16
    print("\n=== Test 3: Model in FP16 ===")
    model_fp16 = model.half()
    frame.img = frame.img.half()
    mean_time_fp16, std_time_fp16 = benchmark_model_inference(model_fp16, frame, use_fp8=False)
    print(f"FP16: {mean_time_fp16:.2f} ± {std_time_fp16:.2f} ms")
    print(f"Speedup vs FP32: {mean_time_orig/mean_time_fp16:.2f}x")
    
    # Test 4: FP16 model with FP8 autocast
    print("\n=== Test 4: FP16 Model + FP8 Autocast ===")
    mean_time_fp16_fp8, std_time_fp16_fp8 = benchmark_model_inference(model_fp16, frame, use_fp8=True)
    print(f"FP16 + FP8: {mean_time_fp16_fp8:.2f} ± {std_time_fp16_fp8:.2f} ms")
    print(f"Speedup vs FP32: {mean_time_orig/mean_time_fp16_fp8:.2f}x")
    
    # Test 5: Replace Linear layers with TE.Linear and use FP8
    print("\n=== Test 5: Native TE Linear Layers + FP8 ===")
    print("Loading fresh model and replacing all Linear layers with TE.Linear...")
    model_te = load_mast3r(device=device)
    model_te.eval()
    model_te = replace_linear_layers_with_te(model_te, verbose=False)
    
    # Create fresh frame for TE model
    frame_te = create_test_frame(224, 224, device)
    
    print("Benchmarking TE model with FP8...")
    mean_time_te_fp8, std_time_te_fp8 = benchmark_model_inference(model_te, frame_te, use_fp8=True)
    print(f"TE + FP8: {mean_time_te_fp8:.2f} ± {std_time_te_fp8:.2f} ms")
    print(f"Speedup vs FP32: {mean_time_orig/mean_time_te_fp8:.2f}x")
    print(f"Speedup vs FP16: {mean_time_fp16/mean_time_te_fp8:.2f}x")
    
    # Summary
    print("\n=== Summary ===")
    print(f"FP32 Baseline:        {mean_time_orig:.2f} ms (1.00x)")
    print(f"FP32 + FP8 autocast:  {mean_time_fp8:.2f} ms ({mean_time_orig/mean_time_fp8:.2f}x)")
    print(f"FP16:                 {mean_time_fp16:.2f} ms ({mean_time_orig/mean_time_fp16:.2f}x)")
    print(f"FP16 + FP8 autocast:  {mean_time_fp16_fp8:.2f} ms ({mean_time_orig/mean_time_fp16_fp8:.2f}x)")
    print(f"TE Linear + FP8:      {mean_time_te_fp8:.2f} ms ({mean_time_orig/mean_time_te_fp8:.2f}x)")
    
    print(f"\nBest configuration: ", end="")
    
    print(f"\nBest configuration: ", end="")
    
    best_time = min(mean_time_orig, mean_time_fp8, mean_time_fp16, mean_time_fp16_fp8, mean_time_te_fp8)
    if best_time == mean_time_te_fp8:
        print(f"TE Linear + FP8 ({mean_time_te_fp8:.2f} ms, {mean_time_orig/mean_time_te_fp8:.2f}x speedup)")
    elif best_time == mean_time_fp16_fp8:
        print(f"FP16 + FP8 ({mean_time_fp16_fp8:.2f} ms, {mean_time_orig/mean_time_fp16_fp8:.2f}x speedup)")
    elif best_time == mean_time_fp16:
        print(f"FP16 ({mean_time_fp16:.2f} ms, {mean_time_orig/mean_time_fp16:.2f}x speedup)")
    elif best_time == mean_time_fp8:
        print(f"FP8 ({mean_time_fp8:.2f} ms, {mean_time_orig/mean_time_fp8:.2f}x speedup)")
    else:
        print("No speedup achieved")
    
    print(f"\nMemory usage: {torch.cuda.memory_allocated()/1e9:.2f} GB")