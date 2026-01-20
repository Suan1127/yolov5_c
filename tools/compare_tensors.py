#!/usr/bin/env python3
"""
Compare C implementation output with Python golden reference
Compares all matching files in two directories
"""

import numpy as np
import json
import argparse
from pathlib import Path
import sys


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
    try:
        golden, golden_dims = load_tensor_bin(golden_path)
        test, test_dims = load_tensor_bin(test_path)
    except Exception as e:
        print(f"Error loading tensors: {e}")
        return False
    
    # Check shape
    if not np.array_equal(golden_dims, test_dims):
        print(f"  ✗ Shape mismatch: golden {golden_dims} vs test {test_dims}")
        return False
    
    # Compute differences
    diff = np.abs(golden - test)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    rmse = np.sqrt(np.mean(diff ** 2))
    
    # Check if within tolerance
    within_tolerance = max_diff < tolerance
    
    print(f"  Shape: {golden_dims}")
    print(f"  Max diff: {max_diff:.6e}")
    print(f"  Mean diff: {mean_diff:.6e}")
    print(f"  RMSE: {rmse:.6e}")
    print(f"  Within tolerance ({tolerance}): {'✓' if within_tolerance else '✗'}")
    
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
                print(f"    {idx}: golden={golden[idx]:.6e}, test={test[idx]:.6e}, diff={diff[idx]:.6e}")
    
    return within_tolerance


def compare_directories(golden_dir, test_dir, tolerance=1e-4):
    """
    Compare all matching files in two directories
    """
    golden_dir = Path(golden_dir)
    test_dir = Path(test_dir)
    
    if not golden_dir.exists():
        print(f"Error: Golden directory does not exist: {golden_dir}")
        return False
    
    if not test_dir.exists():
        print(f"Error: Test directory does not exist: {test_dir}")
        return False
    
    # Find all .bin files in golden directory
    golden_files = {f.name: f for f in golden_dir.glob("*.bin")}
    test_files = {f.name: f for f in test_dir.glob("*.bin")}
    
    if not golden_files:
        print(f"Error: No .bin files found in {golden_dir}")
        return False
    
    print(f"Comparing {len(golden_files)} files...")
    print(f"Golden directory: {golden_dir}")
    print(f"Test directory: {test_dir}")
    print(f"Tolerance: {tolerance}")
    print()
    
    results = {}
    all_passed = True
    
    # Sort files by layer number for easier identification of first mismatch
    # Priority: input.bin, then layer_XXX.bin (sorted by number), then output files
    def sort_key(filename):
        if filename == "input.bin":
            return (0, 0)
        elif filename.startswith("layer_"):
            try:
                layer_num = int(filename[6:9])  # Extract number from "layer_XXX.bin"
                return (1, layer_num)
            except:
                return (2, filename)
        elif filename.startswith("output"):
            return (3, filename)
        else:
            return (4, filename)
    
    sorted_filenames = sorted(golden_files.keys(), key=sort_key)
    
    for filename in sorted_filenames:
        golden_path = golden_files[filename]
        
        if filename not in test_files:
            print(f"✗ {filename}: Missing in test directory")
            results[filename] = False
            all_passed = False
            continue
        
        test_path = test_files[filename]
        print(f"Comparing {filename}...")
        
        passed = compare_tensors(golden_path, test_path, tolerance)
        results[filename] = passed
        
        if not passed:
            all_passed = False
            # Print warning for first failure
            if all_passed or len([r for r in results.values() if not r]) == 1:
                print(f"\n⚠ First mismatch detected at: {filename}")
                print("  This is likely where the error originates.\n")
        
        print()
    
    # Summary
    print("=" * 60)
    print("Summary:")
    passed_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    print(f"  Passed: {passed_count}/{total_count}")
    print(f"  Failed: {total_count - passed_count}/{total_count}")
    
    if all_passed:
        print("\n✓ All comparisons passed!")
    else:
        print("\n✗ Some comparisons failed")
        print("\nFailed files:")
        for filename, passed in results.items():
            if not passed:
                print(f"  - {filename}")
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(description='Compare C output with golden reference')
    parser.add_argument('golden_dir', type=str, 
                        help='Directory containing golden reference .bin files (e.g., testdata/python)')
    parser.add_argument('test_dir', type=str, 
                        help='Directory containing test .bin files (e.g., testdata/c)')
    parser.add_argument('--tolerance', type=float, default=1e-4, 
                        help='Tolerance for comparison (default: 1e-4)')
    
    args = parser.parse_args()
    
    success = compare_directories(args.golden_dir, args.test_dir, args.tolerance)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
