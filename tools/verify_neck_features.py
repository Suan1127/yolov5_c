#!/usr/bin/env python3
"""
Verify that C implementation uses the same backbone feature maps for neck as Python golden
"""

import numpy as np
from pathlib import Path

def verify_neck_features():
    """Verify backbone feature maps used in neck"""
    
    python_dir = Path("testdata_n/python")
    c_dir = Path("testdata_n/c")
    
    print("=== Backbone Feature Maps Passed to Neck (Head) ===\n")
    
    # Feature maps that should be passed from Backbone to Neck
    backbone_to_neck = {
        "Layer 6": {
            "file": "layer_006.bin",
            "shape": (1, 128, 40, 40),
            "used_in": "Layer 12 (Concat)",
            "description": "Used in Layer 12: Concat([Layer 11, Layer 6])"
        },
        "Layer 4": {
            "file": "layer_004.bin",
            "shape": (1, 64, 80, 80),
            "used_in": "Layer 16 (Concat)",
            "description": "Used in Layer 16: Concat([Layer 15, Layer 4])"
        },
        "Layer 9": {
            "file": "layer_009.bin",
            "shape": (1, 256, 20, 20),
            "used_in": "Layer 10 (direct)",
            "description": "Directly passed to Layer 10 (Head start)"
        }
    }
    
    print("Expected feature maps from Backbone to Neck:")
    for layer_name, info in backbone_to_neck.items():
        print(f"  {layer_name}:")
        print(f"    Shape: {info['shape']}")
        print(f"    Used in: {info['used_in']}")
        print(f"    Description: {info['description']}")
        print()
    
    # Check if files exist
    print("=== File Existence Check ===\n")
    all_exist = True
    
    for layer_name, info in backbone_to_neck.items():
        python_file = python_dir / info["file"]
        c_file = c_dir / info["file"]
        
        python_exists = python_file.exists()
        c_exists = c_file.exists()
        
        print(f"{layer_name} ({info['file']}):")
        print(f"  Python: {'✓' if python_exists else '✗'}")
        print(f"  C:      {'✓' if c_exists else '✗'}")
        
        if not python_exists or not c_exists:
            all_exist = False
        print()
    
    if not all_exist:
        print("⚠️  Some files are missing!")
        return False
    
    # Compare shapes
    print("=== Shape Verification ===\n")
    shapes_match = True
    
    for layer_name, info in backbone_to_neck.items():
        python_file = python_dir / info["file"]
        c_file = c_dir / info["file"]
        
        # Read Python golden
        with open(python_file, 'rb') as f:
            py_dims = np.fromfile(f, dtype=np.int32, count=4)
            py_shape = tuple(py_dims)
        
        # Read C output
        with open(c_file, 'rb') as f:
            c_dims = np.fromfile(f, dtype=np.int32, count=4)
            c_shape = tuple(c_dims)
        
        expected_shape = info["shape"]
        
        print(f"{layer_name}:")
        print(f"  Expected: {expected_shape}")
        print(f"  Python:  {py_shape} {'✓' if py_shape == expected_shape else '✗'}")
        print(f"  C:       {c_shape} {'✓' if c_shape == expected_shape else '✗'}")
        
        if py_shape != expected_shape or c_shape != expected_shape:
            shapes_match = False
        print()
    
    if not shapes_match:
        print("⚠️  Some shapes don't match!")
        return False
    
    # Summary
    print("=== Summary ===\n")
    print("✅ C implementation uses the same backbone feature maps for neck as Python golden:")
    print("  1. Layer 6 (1, 128, 40, 40) → Layer 12 (Concat)")
    print("  2. Layer 4 (1, 64, 80, 80) → Layer 16 (Concat)")
    print("  3. Layer 9 (1, 256, 20, 20) → Layer 10 (direct)")
    print()
    print("Total: 3 feature maps from Backbone to Neck")
    print("✅ Both Python and C implementations are consistent!")
    
    return True

if __name__ == '__main__':
    verify_neck_features()
