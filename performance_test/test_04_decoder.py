"""
Test 4: Decoder (Cross-Attention)
Tests model._decoder() - the cross-attention decoder
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

def benchmark_decoder(num_warmup=10, num_runs=50):
    """Benchmark decoder (cross-attention)"""
    print("=" * 70)
    print("TEST 4: Decoder (Cross-Attention)")
    print("=" * 70)
    
    # Check if checkpoint exists
    checkpoint_path = Path("../checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
    if not checkpoint_path.exists():
        print(f"\nError: Checkpoint not found at {checkpoint_path}")
        print("Please download the model checkpoint first.")
        return None
    
    # Create dummy frames
    img1 = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    T_WC = lietorch.SE3.Identity(1, device="cuda")
    
    # Load FP16 model (for FP8 testing)
    print("\nLoading FP16 model and encoding frames...")
    model = load_mast3r(device="cuda", use_fp8=True)
    model.eval()
    
    frame1 = create_frame(0, img1, T_WC, img_size=512, use_fp16=True)
    frame2 = create_frame(1, img2, T_WC, img_size=512, use_fp16=True)
    
    # Encode frames
    fp8_recipe = te.fp8.DelayedScaling(margin=0, fp8_format=recipe.Format.E4M3)
    with torch.no_grad():
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            feat1, pos1, _ = model._encode_image(frame1.img, frame1.img_true_shape)
            feat2, pos2, _ = model._encode_image(frame2.img, frame2.img_true_shape)
    
    shape1 = frame1.img_true_shape
    shape2 = frame2.img_true_shape
    
    print(f"  Feature shape: {feat1.shape}")
    
    # Test WITHOUT FP8 (FP16 only)
    print("\nBenchmarking FP16 decoder...")
    
    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            res1, res2 = model._decoder(feat1, pos1, feat2, pos2)
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            res1, res2 = model._decoder(feat1, pos1, feat2, pos2)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    
    mean_fp16 = np.mean(times)
    std_fp16 = np.std(times)
    print(f"  FP16: {mean_fp16:.2f} ± {std_fp16:.2f} ms")
    if isinstance(res1, dict):
        print(f"  Output shape: {res1['pts3d'].shape}")
    else:
        print(f"  Output is tuple, length: {len(res1)}")
    
    # Test WITH FP8 autocast
    print("\nBenchmarking FP16 + FP8 decoder...")
    
    # Enable FP8 for decoder
    import mast3r_slam.mast3r_utils as mutils
    original_use_fp8 = mutils.USE_FP8
    mutils.USE_FP8 = True
    
    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            res1, res2 = model._decoder(feat1, pos1, feat2, pos2)
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            res1, res2 = model._decoder(feat1, pos1, feat2, pos2)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    
    mean_fp8 = np.mean(times)
    std_fp8 = np.std(times)
    print(f"  FP16+FP8: {mean_fp8:.2f} ± {std_fp8:.2f} ms")
    print(f"  Speedup: {mean_fp16/mean_fp8:.2f}x")
    
    mutils.USE_FP8 = original_use_fp8
    
    return {
        "decoder_fp16": mean_fp16,
        "decoder_fp8": mean_fp8
    }

if __name__ == "__main__":
    results = benchmark_decoder()
    
    if results is None:
        print("\nSkipped: Model checkpoint not available")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print("SUMMARY:")
    print(f"  Decoder runs for EVERY frame-keyframe pair")
    print(f"  FP16: {results['decoder_fp16']:.2f}ms")
    print(f"  FP8:  {results['decoder_fp8']:.2f}ms ({results['decoder_fp16']/results['decoder_fp8']:.2f}x speedup)")
    print(f"\n  With 3 keyframes: decoder cost = {results['decoder_fp8']*3:.1f}ms per frame")
