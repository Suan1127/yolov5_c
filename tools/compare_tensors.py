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
    print(f"  Within tolerance ({tolerance}): {'OK' if within_tolerance else 'FAIL'}")
    
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
    
    print(f"Found {len(golden_files)} files in golden directory")
    print(f"Found {len(test_files)} files in test directory")
    print(f"Golden directory: {golden_dir}")
    print(f"Test directory: {test_dir}")
    print(f"Tolerance: {tolerance}")
    print()
    
    results = {}
    missing_files = []
    all_passed = True
    
    # Define layers that don't have weights (Upsample, etc.) - these may not be saved
    no_weight_layers = {11, 15}  # Upsample layers
    
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
            # Check if this is a layer that might not be saved
            is_no_weight_layer = False
            if filename.startswith("layer_"):
                try:
                    layer_num = int(filename[6:9])
                    if layer_num in no_weight_layers:
                        is_no_weight_layer = True
                except:
                    pass
            
            if is_no_weight_layer:
                print(f"SKIP {filename}: Upsample layer (no weights, may not be saved)")
                results[filename] = None  # None means skipped
            elif filename.startswith("output") or filename.endswith("_0.bin") or filename.endswith("_1.bin") or filename.endswith("_2.bin"):
                print(f"SKIP {filename}: Output file (optional)")
                results[filename] = None
            elif filename == "bus.bin" or filename.endswith("_bus.bin"):
                print(f"SKIP {filename}: Input image file (optional)")
                results[filename] = None
            else:
                print(f"MISSING {filename}: Not found in test directory")
                missing_files.append(filename)
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
            failed_count = len([r for r in results.values() if r is False])
            if failed_count == 1:
                print(f"\nWARNING: First mismatch detected at: {filename}")
                print("  This is likely where the error originates.\n")
        
        print()
    
    # Summary
    print("=" * 60)
    print("Summary:")
    
    # Count only actual comparisons (not skipped files)
    actual_results = {k: v for k, v in results.items() if v is not None}
    passed_count = sum(1 for v in actual_results.values() if v)
    total_count = len(actual_results)
    skipped_count = len([v for v in results.values() if v is None])
    
    print(f"  Compared: {total_count} files")
    print(f"  Passed: {passed_count}/{total_count}")
    if total_count - passed_count > 0:
        print(f"  Failed: {total_count - passed_count}/{total_count}")
    if skipped_count > 0:
        print(f"  Skipped: {skipped_count} files (Upsample layers, output files)")
    if missing_files:
        print(f"  Missing: {len(missing_files)} files")
    
    if all_passed and not missing_files:
        print("\n[OK] All comparisons passed!")
    else:
        if not all_passed:
            print("\n[FAIL] Some comparisons failed")
            print("\nFailed files:")
            for filename, passed in results.items():
                if passed is False:
                    print(f"  - {filename}")
        if missing_files:
            print("\nMissing files (not found in test directory):")
            for filename in missing_files:
                print(f"  - {filename}")
    
    # Return True only if all actual comparisons passed and no critical files are missing
    return all_passed and not missing_files


def main():
    parser = argparse.ArgumentParser(description='Compare C output with golden reference')
    parser.add_argument('golden', type=str, 
                        help='Golden reference file or directory (e.g., testdata/python/layer_001.bin or testdata/python)')
    parser.add_argument('test', type=str, 
                        help='Test file or directory (e.g., testdata/c/layer_001.bin or testdata/c)')
    parser.add_argument('--tolerance', type=float, default=1e-4, 
                        help='Tolerance for comparison (default: 1e-4)')
    
    args = parser.parse_args()
    
    golden_path = Path(args.golden)
    test_path = Path(args.test)
    
    # Check if both are files or both are directories
    if golden_path.is_file() and test_path.is_file():
        # Single file comparison
        print(f"Comparing single files:")
        print(f"  Golden: {golden_path}")
        print(f"  Test: {test_path}")
        print(f"  Tolerance: {args.tolerance}")
        print()
        success = compare_tensors(str(golden_path), str(test_path), args.tolerance)
    elif golden_path.is_dir() and test_path.is_dir():
        # Directory comparison
        success = compare_directories(str(golden_path), str(test_path), args.tolerance)
    else:
        print(f"Error: Both arguments must be either files or directories")
        print(f"  Golden: {golden_path} ({'file' if golden_path.is_file() else 'dir' if golden_path.exists() else 'not found'})")
        print(f"  Test: {test_path} ({'file' if test_path.is_file() else 'dir' if test_path.exists() else 'not found'})")
        sys.exit(1)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
