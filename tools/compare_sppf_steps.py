#!/usr/bin/env python3
"""
SPPF 블록의 각 단계별 출력을 비교하는 스크립트
PyTorch와 C 구현의 중간 출력을 비교하여 어디서 문제가 발생하는지 찾습니다.
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

def compare_tensors(pytorch_path, c_path, step_name):
    """Compare two tensor files"""
    pytorch_tensor = load_tensor_bin(pytorch_path)
    c_tensor = load_tensor_bin(c_path)
    
    if pytorch_tensor is None:
        print(f"  {step_name}: PyTorch 파일 없음: {pytorch_path}")
        return False
    
    if c_tensor is None:
        print(f"  {step_name}: C 파일 없음: {c_path}")
        return False
    
    if pytorch_tensor.shape != c_tensor.shape:
        print(f"  {step_name}: Shape 불일치 - PyTorch: {pytorch_tensor.shape}, C: {c_tensor.shape}")
        return False
    
    diff = np.abs(pytorch_tensor - c_tensor)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    rmse = np.sqrt(np.mean(diff ** 2))
    
    tolerance = 0.0001
    within_tolerance = max_diff < tolerance
    
    status = "[OK]" if within_tolerance else "[FAIL]"
    
    print(f"  {step_name}: {status}")
    print(f"    Shape: {pytorch_tensor.shape}")
    print(f"    Max diff: {max_diff:.6e}")
    print(f"    Mean diff: {mean_diff:.6e}")
    print(f"    RMSE: {rmse:.6e}")
    print(f"    PyTorch range: [{pytorch_tensor.min():.6f}, {pytorch_tensor.max():.6f}]")
    print(f"    C range: [{c_tensor.min():.6f}, {c_tensor.max():.6f}]")
    
    if not within_tolerance:
        # Find locations with large differences
        large_diff_mask = diff > tolerance
        num_large_diff = np.sum(large_diff_mask)
        print(f"    Locations with diff > {tolerance}: {num_large_diff}")
        
        # Find max diff location
        max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"    Max diff at {max_diff_idx}:")
        print(f"      PyTorch: {pytorch_tensor[max_diff_idx]:.6f}")
        print(f"      C: {c_tensor[max_diff_idx]:.6f}")
        print(f"      Diff: {diff[max_diff_idx]:.6e}")
    
    return within_tolerance

def main():
    print("=== SPPF 블록 단계별 비교 ===\n")
    
    pytorch_dir = Path("debug/pytorch")
    c_dir = Path("debug/c")
    
    steps = [
        ("cv1", "sppf_cv1_output.bin", "sppf_cv1_output.bin"),
        ("y1", "sppf_y1_output.bin", "sppf_y1_output.bin"),
        ("y2", "sppf_y2_output.bin", "sppf_y2_output.bin"),
        ("y4", "sppf_y4_output.bin", "sppf_y4_output.bin"),
        ("concat", "sppf_concat_output.bin", "sppf_concat_output.bin"),
        ("cv2", "sppf_cv2_output.bin", "sppf_cv2_output.bin"),
    ]
    
    all_ok = True
    for step_name, pytorch_file, c_file in steps:
        pytorch_path = pytorch_dir / pytorch_file
        c_path = c_dir / c_file
        
        ok = compare_tensors(str(pytorch_path), str(c_path), step_name)
        if not ok:
            all_ok = False
        print()
    
    if all_ok:
        print("=== 모든 단계 일치 ===")
    else:
        print("=== 일부 단계 불일치 ===")
        print("첫 번째 실패한 단계를 확인하세요.")

if __name__ == '__main__':
    main()
