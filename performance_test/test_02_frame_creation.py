"""
Test 2: Frame Creation (Image Preprocessing)
Tests create_frame() function
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

from mast3r_slam.frame import create_frame

def benchmark_frame_creation(num_warmup=10, num_runs=100):
    """Benchmark frame creation"""
    print("=" * 70)
    print("TEST 2: Frame Creation (Image Preprocessing)")
    print("=" * 70)
    
        # Create test data
    H, W = 512, 512
    img_512 = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    T_WC = lietorch.SE3.Identity(1, device="cuda")
    
    # Test 512x512
    print("\nTesting 512x512 images:")
    
    # Warmup
    for _ in range(num_warmup):
        frame = create_frame(0, img_512, T_WC, img_size=512)
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        frame = create_frame(0, img_512, T_WC, img_size=512, use_fp16=False)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    
    mean_512 = np.mean(times)
    std_512 = np.std(times)
    print(f"  FP32: {mean_512:.3f} ± {std_512:.3f} ms")
    
    # Test FP16
    times_fp16 = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        frame = create_frame(0, img_512, T_WC, img_size=512, use_fp16=True)
        torch.cuda.synchronize()
        times_fp16.append((time.perf_counter() - start) * 1000)
    
    mean_512_fp16 = np.mean(times_fp16)
    std_512_fp16 = np.std(times_fp16)
    print(f"  FP16: {mean_512_fp16:.3f} ± {std_512_fp16:.3f} ms")
    
    # Note: Only test 512 since resize_img only supports 224 or 512
    print("\nNote: Skipping 1024x1024 test (resize_img only supports 224 or 512)")
    
    return {
        "frame_creation_512_fp32": mean_512,
        "frame_creation_512_fp16": mean_512_fp16
    }

if __name__ == "__main__":
    results = benchmark_frame_creation()
    print(f"\n{'='*70}")
    print("SUMMARY:")
    print(f"  Frame creation (512x512): {results['frame_creation_512_fp32']:.3f}ms")
    print(f"  FP16 overhead: {results['frame_creation_512_fp16'] - results['frame_creation_512_fp32']:.3f}ms")
    print(f"  At 30 FPS: {results['frame_creation_512_fp32'] * 30:.1f}ms/sec ({results['frame_creation_512_fp32']*30/10:.1f}%)")
