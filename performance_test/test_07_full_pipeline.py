"""
Test 7: Full Pipeline (End-to-End Frame Processing)
Tests complete frame processing pipeline from image to tracked pose
"""
import torch
import time
import numpy as np
from pathlib import Path
import yaml
import cv2

config_path = Path("../config/base.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

import sys
sys.path.insert(0, str(Path("..").resolve()))

from mast3r_slam.config import set_global_config
set_global_config(config)

from mast3r_slam.mast3r_utils import load_mast3r, mast3r_asymmetric_inference
from mast3r_slam.frame import create_frame, Frame
from mast3r_slam.matching import match_iterative_proj

def benchmark_full_pipeline(num_warmup=5, num_runs=30, use_fp8=False):
    """Benchmark full pipeline: image -> frame -> encoding -> matching -> output"""
    print("=" * 70)
    print(f"TEST 7: Full Pipeline (FP8={use_fp8})")
    print("=" * 70)
    
    # Check if checkpoint exists
    checkpoint_path = Path("../checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
    if not checkpoint_path.exists():
        print(f"\nError: Checkpoint not found at {checkpoint_path}")
        print("Please download the model checkpoint first.")
        return None
    
    # Load model
    print("Loading model...")
    model = load_mast3r(use_fp8=use_fp8)
    model = model.eval().cuda()
    print(f"  Model dtype: {next(model.parameters()).dtype}")
    
    # Create dummy images
    H, W = 512, 512
    img1 = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    
    # Create frames with proper signature: create_frame(i, img, T_WC, img_size=512, device="cuda:0", use_fp16=False)
    import lietorch
    T_WC = lietorch.SE3.Identity(1, device='cuda')
    frame1 = create_frame(0, img1, T_WC, img_size=512, device='cuda')
    frame2 = create_frame(1, img2, T_WC, img_size=512, device='cuda')
    
    print(f"\nPipeline stages:")
    print(f"  1. Frame creation (preprocessing)")
    print(f"  2. Encoder (ViT backbone)")
    print(f"  3. Decoder (cross-attention)")
    print(f"  4. Matching (CUDA backend)")
    
    # Warmup
    print("\nWarming up full pipeline...")
    for _ in range(num_warmup):
        with torch.no_grad():
            if use_fp8:
                with torch.autocast(device_type='cuda', dtype=torch.float8_e4m3fn):
                    X, C, D, Q = mast3r_asymmetric_inference(
                        model, frame1, frame2
                    )
            else:
                X, C, D, Q = mast3r_asymmetric_inference(
                    model, frame1, frame2
                )
            
            # Matching - X is already 4D from asymmetric_inference (after downsample)
            pts3d_1 = X[0].reshape(1, -1, 3)
            pts3d_2 = X[1].reshape(1, -1, 3)
            desc1 = D[0].reshape(1, -1, D.shape[-1])
            desc2 = D[1].reshape(1, -1, D.shape[-1])
            
            idx_1_to_2, valid_proj = match_iterative_proj(
                pts3d_1, pts3d_2, desc1, desc2
            )
            torch.cuda.synchronize()
    
    # Benchmark
    print("Benchmarking full pipeline...")
    times_total = []
    times_inference = []
    times_matching = []
    
    for _ in range(num_runs):
        # Total pipeline
        torch.cuda.synchronize()
        start_total = time.perf_counter()
        
        # Inference
        start_inference = time.perf_counter()
        with torch.no_grad():
            if use_fp8:
                with torch.autocast(device_type='cuda', dtype=torch.float8_e4m3fn):
                    X, C, D, Q = mast3r_asymmetric_inference(
                        model, frame1, frame2
                    )
            else:
                X, C, D, Q = mast3r_asymmetric_inference(
                    model, frame1, frame2
                )
        torch.cuda.synchronize()
        time_inference = (time.perf_counter() - start_inference) * 1000
        
        # Matching
        start_matching = time.perf_counter()
        pts3d_1 = X[0].reshape(1, -1, 3)
        pts3d_2 = X[1].reshape(1, -1, 3)
        desc1 = D[0].reshape(1, -1, D.shape[-1])
        desc2 = D[1].reshape(1, -1, D.shape[-1])
        
        idx_1_to_2, valid_proj = match_iterative_proj(
            pts3d_1, pts3d_2, desc1, desc2
        )
        torch.cuda.synchronize()
        time_matching = (time.perf_counter() - start_matching) * 1000
        
        time_total = (time.perf_counter() - start_total) * 1000
        
        times_total.append(time_total)
        times_inference.append(time_inference)
        times_matching.append(time_matching)
    
    mean_total = np.mean(times_total)
    std_total = np.std(times_total)
    mean_inference = np.mean(times_inference)
    std_inference = np.std(times_inference)
    mean_matching = np.mean(times_matching)
    std_matching = np.std(times_matching)
    
    print(f"\n{'='*70}")
    print("RESULTS:")
    print(f"  Inference:  {mean_inference:6.2f} ± {std_inference:5.2f} ms")
    print(f"  Matching:   {mean_matching:6.2f} ± {std_matching:5.2f} ms")
    print(f"  Total:      {mean_total:6.2f} ± {std_total:5.2f} ms")
    print(f"  Framerate:  {1000.0/mean_total:5.1f} FPS (single frame pair)")
    
    return {
        "inference": mean_inference,
        "matching": mean_matching,
        "total": mean_total,
        "fps": 1000.0/mean_total
    }

if __name__ == "__main__":
    print("\n" + "="*70)
    print("FULL PIPELINE BENCHMARK")
    print("="*70)
    
    # Test without FP8
    print("\n1. Testing without FP8...")
    results_fp32 = benchmark_full_pipeline(use_fp8=False)
    
    if results_fp32 is None:
        print("\nSkipped: Model checkpoint not available")
        sys.exit(1)
    
    # Test with FP8
    print("\n2. Testing with FP8...")
    results_fp8 = benchmark_full_pipeline(use_fp8=True)
    
    # Comparison
    print("\n" + "="*70)
    print("COMPARISON:")
    print("="*70)
    print(f"{'Stage':<20} {'FP16 (ms)':<15} {'FP8 (ms)':<15} {'Speedup':<10}")
    print("-"*70)
    print(f"{'Inference':<20} {results_fp32['inference']:>7.2f} {results_fp8['inference']:>14.2f} {results_fp32['inference']/results_fp8['inference']:>13.2f}x")
    print(f"{'Matching':<20} {results_fp32['matching']:>7.2f} {results_fp8['matching']:>14.2f} {results_fp32['matching']/results_fp8['matching']:>13.2f}x")
    print(f"{'Total':<20} {results_fp32['total']:>7.2f} {results_fp8['total']:>14.2f} {results_fp32['total']/results_fp8['total']:>13.2f}x")
    print(f"{'Framerate':<20} {results_fp32['fps']:>7.1f} {results_fp8['fps']:>14.1f} {results_fp8['fps']/results_fp32['fps']:>13.2f}x")
    
    print("\n" + "="*70)
    print("NOTES:")
    print("  - This tests a SINGLE frame pair (new frame vs 1 keyframe)")
    print("  - Real pipeline processes new frame vs MULTIPLE keyframes")
    print(f"  - With 3 keyframes: ~{results_fp8['total']*3:.1f}ms per frame")
    print(f"  - Target for 30 FPS: 33.33ms per frame")
    print(f"  - Backend optimization (not included): ~10-20ms additional")
