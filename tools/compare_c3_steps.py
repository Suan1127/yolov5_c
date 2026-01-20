#!/usr/bin/env python3
"""
C3 블록 단계별 비교 스크립트
PyTorch와 C 구현의 각 단계별 출력을 비교합니다.
"""

import numpy as np
import sys
from pathlib import Path

def load_tensor_bin(filepath):
    """Load tensor from binary file (dims + data)"""
    try:
        with open(filepath, 'rb') as f:
            dims = np.frombuffer(f.read(4 * 4), dtype=np.int32)
            n, c, h, w = dims
            data = np.frombuffer(f.read(n * c * h * w * 4), dtype=np.float32)
            return data.reshape(n, c, h, w)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def compare_step(step_name, pytorch_path, c_path, tolerance=1e-4):
    """Compare a single step"""
    print(f"\n{'='*60}")
    print(f"Step: {step_name}")
    print(f"{'='*60}")
    
    pytorch_tensor = load_tensor_bin(pytorch_path)
    c_tensor = load_tensor_bin(c_path)
    
    if pytorch_tensor is None:
        print(f"  [FAIL] PyTorch file not found: {pytorch_path}")
        return False
    
    if c_tensor is None:
        print(f"  [FAIL] C file not found: {c_path}")
        return False
    
    # Check shape
    if pytorch_tensor.shape != c_tensor.shape:
        print(f"  [FAIL] Shape mismatch:")
        print(f"    PyTorch: {pytorch_tensor.shape}")
        print(f"    C:       {c_tensor.shape}")
        return False
    
    # Compute differences
    diff = np.abs(pytorch_tensor - c_tensor)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    rmse = np.sqrt(np.mean(diff ** 2))
    
    # Check if within tolerance
    within_tolerance = max_diff < tolerance
    
    print(f"  Shape: {pytorch_tensor.shape}")
    print(f"  Max diff: {max_diff:.6e}")
    print(f"  Mean diff: {mean_diff:.6e}")
    print(f"  RMSE: {rmse:.6e}")
    print(f"  Within tolerance ({tolerance}): {'[OK]' if within_tolerance else '[FAIL]'}")
    
    if not within_tolerance:
        # Find locations with large differences
        large_diff_mask = diff > tolerance
        num_large = np.sum(large_diff_mask)
        print(f"  Locations with diff > {tolerance}: {num_large}")
        
        if num_large > 0 and num_large < 100:
            indices = np.where(large_diff_mask)
            print(f"  First few large diff locations:")
            for i in range(min(10, num_large)):
                idx = tuple(a[i] for a in indices)
                print(f"    {idx}: pytorch={pytorch_tensor[idx]:.6e}, c={c_tensor[idx]:.6e}, diff={diff[idx]:.6e}")
        
        # Print statistics
        print(f"  PyTorch range: [{pytorch_tensor.min():.6f}, {pytorch_tensor.max():.6f}]")
        print(f"  C range:       [{c_tensor.min():.6f}, {c_tensor.max():.6f}]")
    
    return within_tolerance

def main():
    steps = [
        ("cv1 (Conv + BN + SiLU)", 
         "debug/pytorch/c3_cv1_output.bin",
         "debug/c/c3_cv1_output.bin"),
        ("Bottleneck", 
         "debug/pytorch/c3_bottleneck_output.bin",
         "debug/c/c3_bottleneck_output.bin"),
        ("cv2 (skip path)", 
         "debug/pytorch/c3_cv2_output.bin",
         "debug/c/c3_cv2_output.bin"),
        ("Concat", 
         "debug/pytorch/c3_concat_output.bin",
         "debug/c/c3_concat_output.bin"),
        ("cv3 (final)", 
         "debug/pytorch/c3_final_output.bin",
         "debug/c/c3_final_output.bin"),
    ]
    
    print("="*60)
    print("C3 블록 단계별 비교")
    print("="*60)
    
    results = {}
    for step_name, pytorch_path, c_path in steps:
        results[step_name] = compare_step(step_name, pytorch_path, c_path)
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"  Passed: {passed}/{total}")
    print(f"  Failed: {total - passed}/{total}")
    
    print("\n단계별 결과:")
    for step_name, passed in results.items():
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} - {step_name}")
    
    if all(results.values()):
        print("\n[OK] 모든 단계가 일치합니다!")
    else:
        print("\n[FAIL] 일부 단계에서 불일치가 발견되었습니다.")
        print("첫 번째 실패한 단계를 확인하여 문제의 원인을 찾으세요.")
    
    return 0 if all(results.values()) else 1

if __name__ == '__main__':
    sys.exit(main())
