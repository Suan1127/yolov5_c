#!/usr/bin/env python3
"""
Compare C implementation output with Python golden reference
"""

import numpy as np
import json
import argparse
from pathlib import Path


def load_tensor_bin(filepath):
    """Load tensor from binary file (C format)"""
    with open(filepath, 'rb') as f:
        dims = np.fromfile(f, dtype=np.int32, count=4)
        n, c, h, w = dims
        data = np.fromfile(f, dtype=np.float32)
        data = data.reshape(n, c, h, w)
    return data, dims


def compare_tensors(golden_path, test_path, tolerance=1e-4):
    """
    Compare two tensors and report differences
    """
    golden, golden_dims = load_tensor_bin(golden_path)
    test, test_dims = load_tensor_bin(test_path)
    
    # Check shape
    if not np.array_equal(golden_dims, test_dims):
        print(f"Shape mismatch: golden {golden_dims} vs test {test_dims}")
        return False
    
    # Compute differences
    diff = np.abs(golden - test)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    rmse = np.sqrt(np.mean(diff ** 2))
    
    # Check if within tolerance
    within_tolerance = max_diff < tolerance
    
    print(f"Comparison: {Path(golden_path).name} vs {Path(test_path).name}")
    print(f"  Shape: {golden_dims}")
    print(f"  Max diff: {max_diff:.6e}")
    print(f"  Mean diff: {mean_diff:.6e}")
    print(f"  RMSE: {rmse:.6e}")
    print(f"  Within tolerance ({tolerance}): {within_tolerance}")
    
    if not within_tolerance:
        # Find locations with large differences
        large_diff_mask = diff > tolerance
        num_large = np.sum(large_diff_mask)
        print(f"  Locations with diff > {tolerance}: {num_large}")
        
        if num_large > 0:
            indices = np.where(large_diff_mask)
            print(f"  First few large diff locations:")
            for i in range(min(10, num_large)):
                idx = tuple(a[i] for a in indices)
                print(f"    {idx}: golden={golden[idx]:.6e}, test={test[idx]:.6e}, diff={diff[idx]:.6e}")
    
    return within_tolerance


def main():
    parser = argparse.ArgumentParser(description='Compare C output with golden reference')
    parser.add_argument('golden', type=str, help='Path to golden tensor .bin file')
    parser.add_argument('test', type=str, help='Path to test tensor .bin file')
    parser.add_argument('--tolerance', type=float, default=1e-4, help='Tolerance for comparison')
    
    args = parser.parse_args()
    
    success = compare_tensors(args.golden, args.test, args.tolerance)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    import sys
    main()
