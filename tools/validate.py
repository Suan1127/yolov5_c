#!/usr/bin/env python3
"""
Validation pipeline: Generate golden reference, run C inference, and compare
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"\n✗ {description} failed with exit code {result.returncode}")
        return False
    
    print(f"✓ {description} completed successfully")
    return True


def main():
    parser = argparse.ArgumentParser(description='Validation pipeline for YOLOv5 C implementation')
    parser.add_argument('--image', type=str, default='bus', 
                        help='Image name (without extension, default: bus)')
    parser.add_argument('--model', type=str, default='path/to/yolov5s.pt',
                        help='Path to yolov5s.pt model file')
    parser.add_argument('--input-size', type=int, default=640,
                        help='Input image size (default: 640)')
    parser.add_argument('--tolerance', type=float, default=1e-4,
                        help='Tolerance for tensor comparison (default: 1e-4)')
    parser.add_argument('--skip-golden', action='store_true',
                        help='Skip golden reference generation (use existing)')
    parser.add_argument('--skip-c', action='store_true',
                        help='Skip C inference (use existing outputs)')
    
    args = parser.parse_args()
    
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    testdata_python = project_root / 'testdata' / 'python'
    testdata_c = project_root / 'testdata' / 'c'
    input_tensor = testdata_python / 'input.bin'
    
    print("="*60)
    print("YOLOv5 C Implementation Validation Pipeline")
    print("="*60)
    print(f"Image: {args.image}")
    print(f"Model: {args.model}")
    print(f"Input size: {args.input_size}")
    print(f"Tolerance: {args.tolerance}")
    print()
    
    # Step 1: Preprocess image (if input.bin doesn't exist)
    if not input_tensor.exists():
        print("Step 1: Preprocessing image...")
        preprocess_cmd = [
            sys.executable, str(script_dir / 'preprocess.py'),
            '--image', f'{args.image}.jpg',
            '--size', str(args.input_size),
            '--output-dir', str(testdata_python)
        ]
        if not run_command(preprocess_cmd, "Image preprocessing"):
            return 1
    else:
        print(f"✓ Input tensor already exists: {input_tensor}")
    
    # Step 2: Generate golden reference
    if not args.skip_golden:
        print("\nStep 2: Generating golden reference from Python model...")
        dump_cmd = [
            sys.executable, str(script_dir / 'dump_golden.py'),
            args.model,
            str(input_tensor),
            '--output', str(testdata_python)
        ]
        if not run_command(dump_cmd, "Golden reference generation"):
            return 1
    else:
        print("⏭ Skipping golden reference generation")
    
    # Step 3: Run C inference
    if not args.skip_c:
        print("\nStep 3: Running C inference...")
        # Copy input to testdata/c if needed
        testdata_c.mkdir(parents=True, exist_ok=True)
        import shutil
        if not (testdata_c / 'input.bin').exists():
            shutil.copy(input_tensor, testdata_c / 'input.bin')
        
        # Build path
        build_dir = project_root / 'build'
        if sys.platform == 'win32':
            infer_exe = build_dir / 'Release' / 'yolov5_infer.exe'
        else:
            infer_exe = build_dir / 'yolov5_infer'
        
        if not infer_exe.exists():
            print(f"Error: Inference executable not found: {infer_exe}")
            print("Please build the project first:")
            print("  mkdir build && cd build && cmake .. && make")
            return 1
        
        # Run inference with output directory
        infer_cmd = [str(infer_exe), args.image]
        # Note: We need to modify main.c to accept --output-dir argument
        # For now, we'll set it via environment variable or modify the code
        print("Note: C inference needs to be modified to save to testdata/c/")
        print("      This will be implemented in the next step")
        
        # For now, just run the inference
        if not run_command(infer_cmd, "C inference"):
            return 1
    else:
        print("⏭ Skipping C inference")
    
    # Step 4: Compare results
    print("\nStep 4: Comparing results...")
    compare_cmd = [
        sys.executable, str(script_dir / 'compare_tensors.py'),
        str(testdata_python),
        str(testdata_c),
        '--tolerance', str(args.tolerance)
    ]
    
    success = run_command(compare_cmd, "Tensor comparison")
    
    if success:
        print("\n" + "="*60)
        print("✓ Validation pipeline completed successfully!")
        print("="*60)
        return 0
    else:
        print("\n" + "="*60)
        print("✗ Validation pipeline failed")
        print("="*60)
        return 1


if __name__ == '__main__':
    sys.exit(main())
