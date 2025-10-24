"""
Test 3: Image Encoding (Encoder Forward Pass)
Tests model._encode_image() - the ViT encoder
"""
import torch
import time
import numpy as np
from pathlib import Path
import yaml
import lietorch

config_path = Path("../config/base.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

import sys
sys.path.insert(0, str(Path("..").resolve()))

from mast3r_slam.config import set_global_config
set_global_config(config)

from mast3r_slam.mast3r_utils import load_mast3r
from mast3r_slam.frame import create_frame
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

def benchmark_encoding(num_warmup=10, num_runs=50):
    """Benchmark image encoding"""
    print("=" * 70)
    print("TEST 3: Image Encoding (ViT Encoder)")
    print("=" * 70)
    
    # Check if checkpoint exists
    checkpoint_path = Path("../checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
    if not checkpoint_path.exists():
        print(f"\nError: Checkpoint not found at {checkpoint_path}")
        print("Please download the model checkpoint first.")
        return None
    
    # Create dummy frame
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    T_WC = lietorch.SE3.Identity(1, device="cuda")
    
    # Test FP32 model
    print("\nLoading FP32 model...")
    model_fp32 = load_mast3r(device="cuda", use_fp8=False)
    model_fp32.eval()
    
    frame_fp32 = create_frame(0, img, T_WC, img_size=512, use_fp16=False)
    
    # Warmup
    print("Warming up...")
    for _ in range(num_warmup):
        with torch.no_grad():
            feat, pos, _ = model_fp32._encode_image(frame_fp32.img, frame_fp32.img_true_shape)
        torch.cuda.synchronize()
    
    # Benchmark
    print("Benchmarking FP32 encoding...")
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            feat, pos, _ = model_fp32._encode_image(frame_fp32.img, frame_fp32.img_true_shape)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    
    mean_fp32 = np.mean(times)
    std_fp32 = np.std(times)
    print(f"  FP32: {mean_fp32:.2f} ± {std_fp32:.2f} ms")
    print(f"  Feature shape: {feat.shape}")
    
    # Test FP16 model
    print("\nLoading FP16 model...")
    del model_fp32
    torch.cuda.empty_cache()
    
    model_fp16 = load_mast3r(device="cuda", use_fp8=True)
    model_fp16.eval()
    
    frame_fp16 = create_frame(0, img, T_WC, img_size=512, use_fp16=True)
    
    # Warmup
    print("Warming up...")
    for _ in range(num_warmup):
        with torch.no_grad():
            feat, pos, _ = model_fp16._encode_image(frame_fp16.img, frame_fp16.img_true_shape)
        torch.cuda.synchronize()
    
    # Benchmark
    print("Benchmarking FP16 encoding...")
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            feat, pos, _ = model_fp16._encode_image(frame_fp16.img, frame_fp16.img_true_shape)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    
    mean_fp16 = np.mean(times)
    std_fp16 = np.std(times)
    print(f"  FP16: {mean_fp16:.2f} ± {std_fp16:.2f} ms")
    print(f"  Speedup: {mean_fp32/mean_fp16:.2f}x")
    
    # Test FP16 + FP8 autocast
    print("\nBenchmarking FP16 + FP8 autocast...")
    fp8_recipe = te.fp8.DelayedScaling(margin=0, fp8_format=recipe.Format.E4M3)
    
    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                feat, pos, _ = model_fp16._encode_image(frame_fp16.img, frame_fp16.img_true_shape)
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                feat, pos, _ = model_fp16._encode_image(frame_fp16.img, frame_fp16.img_true_shape)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    
    mean_fp8 = np.mean(times)
    std_fp8 = np.std(times)
    print(f"  FP16+FP8: {mean_fp8:.2f} ± {std_fp8:.2f} ms")
    print(f"  Speedup vs FP32: {mean_fp32/mean_fp8:.2f}x")
    print(f"  Speedup vs FP16: {mean_fp16/mean_fp8:.2f}x")
    
    return {
        "encoding_fp32": mean_fp32,
        "encoding_fp16": mean_fp16,
        "encoding_fp8": mean_fp8
    }

if __name__ == "__main__":
    results = benchmark_encoding()
    
    if results is None:
        print("\nSkipped: Model checkpoint not available")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print("SUMMARY:")
    print(f"  Encoding is done ONCE per frame (cached)")
    print(f"  FP32: {results['encoding_fp32']:.2f}ms")
    print(f"  FP16: {results['encoding_fp16']:.2f}ms ({results['encoding_fp32']/results['encoding_fp16']:.2f}x)")
    print(f"  FP8:  {results['encoding_fp8']:.2f}ms ({results['encoding_fp32']/results['encoding_fp8']:.2f}x)")
