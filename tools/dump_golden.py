#!/usr/bin/env python3
"""
Dump golden reference tensors from PyTorch YOLOv5s model
Generates intermediate layer outputs for comparison with C implementation
Saves to testdata/python/ directory with standardized naming
"""

import torch
import numpy as np
import json
import argparse
from pathlib import Path
import sys

# Add yolov5 to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'third_party' / 'yolov5'))

try:
    from models.common import DetectMultiBackend
    from utils.general import check_img_size
except ImportError:
    print("Warning: Could not import YOLOv5 modules. Using fallback method.")
    DetectMultiBackend = None


def save_tensor_bin(tensor, filepath):
    """Save tensor to binary file (C format)"""
    with open(filepath, 'wb') as f:
        shape = np.array(tensor.shape, dtype=np.int32)
        shape.tofile(f)
        tensor.astype(np.float32).tofile(f)


def dump_layer_outputs(model_path, input_tensor_path, output_dir, save_layers=None):
    """
    Run inference and dump intermediate layer outputs
    """
    if save_layers is None:
        # Save layers: [0, 1, 2, 3, 4, 5, 6, 7, 9, 17, 20, 23]
        save_layers = [0, 1, 2, 3, 4, 5, 6, 7, 9, 17, 20, 23]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load input tensor
    input_tensor_path = Path(input_tensor_path)
    if not input_tensor_path.exists():
        print(f"Error: Input tensor file not found: {input_tensor_path}")
        return None
    
    print(f"Loading input tensor from {input_tensor_path}...")
    with open(input_tensor_path, 'rb') as f:
        dims = np.fromfile(f, dtype=np.int32, count=4)
        n, c, h, w = dims
        input_data = np.fromfile(f, dtype=np.float32)
        input_data = input_data.reshape(n, c, h, w)
    
    # Save input tensor (use same filename as input, or "input.bin")
    # Extract image name from input path if it's in format "image_name.bin"
    if input_tensor_path.stem and input_tensor_path.stem != "input":
        input_save_name = input_tensor_path.stem + ".bin"
    else:
        input_save_name = "input.bin"
    
    input_path = output_dir / input_save_name
    save_tensor_bin(input_data, input_path)
    print(f"  Saved input tensor to {input_path}")
    
    # Also save as input.bin for compatibility
    input_path_compat = output_dir / "input.bin"
    if input_path != input_path_compat:
        save_tensor_bin(input_data, input_path_compat)
        print(f"  Also saved as {input_path_compat} for compatibility")
    
    # Convert to torch tensor
    input_tensor = torch.from_numpy(input_data).float()
    
    # Load model
    print(f"Loading model from {model_path}...")
    if DetectMultiBackend:
        # Use YOLOv5's model loader
        model = DetectMultiBackend(model_path, device='cpu')
        model.eval()
    else:
        # Fallback: direct torch.load
        model_dict = torch.load(model_path, map_location='cpu')
        if isinstance(model_dict, dict):
            if 'model' in model_dict:
                model = model_dict['model']
            elif 'state_dict' in model_dict:
                # Reconstruct model from state_dict (simplified)
                print("Warning: Reconstructing model from state_dict may not work correctly")
                model = model_dict
            else:
                model = model_dict
        else:
            model = model_dict
        
        if hasattr(model, 'eval'):
            model.eval()
    
    # Hook to capture intermediate outputs
    outputs = {}
    handles = []
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                outputs[layer_idx] = output.detach().cpu().numpy()
            elif isinstance(output, (list, tuple)):
                # Handle tuple outputs (e.g., from Detect layer)
                for i, out in enumerate(output):
                    if isinstance(out, torch.Tensor):
                        key = f"{layer_idx}_{i}"
                        outputs[key] = out.detach().cpu().numpy()
        return hook
    
    # Register hooks for save layers
    # Note: This is a simplified approach. For accurate layer matching,
    # you may need to traverse the model structure more carefully
    if hasattr(model, 'model'):
        # YOLOv5 model structure
        model_modules = model.model if hasattr(model, 'model') else model
        if hasattr(model_modules, 'named_modules'):
            for name, module in model_modules.named_modules():
                # Try to match layer indices
                # This is approximate - may need adjustment based on actual model structure
                try:
                    # Extract layer number from name (e.g., "model.3" -> 3)
                    if '.' in name:
                        parts = name.split('.')
                        for part in parts:
                            if part.isdigit():
                                layer_idx = int(part)
                                if layer_idx in save_layers:
                                    handle = module.register_forward_hook(make_hook(layer_idx))
                                    handles.append(handle)
                                    print(f"  Registered hook for layer {layer_idx} ({name})")
                                    break
                except:
                    pass
    
    # Run forward pass
    print("Running forward pass...")
    with torch.no_grad():
        try:
            output = model(input_tensor)
        except Exception as e:
            print(f"Error during forward pass: {e}")
            # Clean up hooks
            for handle in handles:
                handle.remove()
            return None
    
    # Clean up hooks
    for handle in handles:
        handle.remove()
    
    # Save outputs
    print(f"\nSaving layer outputs to {output_dir}...")
    saved_count = 0
    
    for layer_idx in save_layers:
        if layer_idx in outputs:
            tensor = outputs[layer_idx]
            output_path = output_dir / f"layer_{layer_idx:03d}.bin"
            save_tensor_bin(tensor, output_path)
            print(f"  Layer {layer_idx}: shape {tensor.shape} -> {output_path.name}")
            saved_count += 1
        else:
            print(f"  Warning: Layer {layer_idx} output not captured")
    
    # Save final output (Detect layer output)
    if isinstance(output, torch.Tensor):
        output_path = output_dir / "output.bin"
        output_np = output.detach().cpu().numpy()
        save_tensor_bin(output_np, output_path)
        print(f"  Final output: shape {output_np.shape} -> {output_path.name}")
        saved_count += 1
    elif isinstance(output, (list, tuple)):
        for i, out in enumerate(output):
            if isinstance(out, torch.Tensor):
                output_path = output_dir / f"output_{i}.bin"
                output_np = out.detach().cpu().numpy()
                save_tensor_bin(output_np, output_path)
                print(f"  Final output[{i}]: shape {output_np.shape} -> {output_path.name}")
                saved_count += 1
    
    # Save metadata
    metadata = {
        "save_layers": save_layers,
        "input_shape": list(input_data.shape),
        "saved_layers": list(outputs.keys()),
        "output_dir": str(output_dir)
    }
    
    meta_path = output_dir / "golden_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDumped {saved_count} tensors to {output_dir}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description='Dump golden reference tensors from PyTorch model')
    parser.add_argument('model', type=str, help='Path to yolov5s.pt')
    parser.add_argument('input', type=str, 
                        help='Path to input tensor .bin file or image name (e.g., "bus")')
    parser.add_argument('--output', type=str, default='testdata/python', 
                        help='Output directory (default: testdata/python)')
    parser.add_argument('--layers', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7, 9, 17, 20, 23],
                        help='Layer indices to save (default: 0 1 2 3 4 5 6 7 9 17 20 23)')
    
    args = parser.parse_args()
    
    # Resolve input tensor path
    input_path = Path(args.input)
    
    # If input doesn't exist, try to find it as image name
    if not input_path.exists():
        # Try as image name (without extension)
        image_name = input_path.stem if input_path.suffix else args.input
        
        # Search in multiple locations
        search_paths = [
            Path('data/inputs') / f"{image_name}.bin",
            Path('testdata/python') / f"{image_name}.bin",
            Path('testdata/c') / f"{image_name}.bin",
            Path(args.output) / f"{image_name}.bin",
        ]
        
        found = False
        for search_path in search_paths:
            if search_path.exists():
                input_path = search_path
                found = True
                print(f"Found input tensor: {input_path}")
                break
        
        if not found:
            print(f"Error: Could not find input tensor file")
            print(f"  Searched for: {image_name}.bin in:")
            for search_path in search_paths:
                print(f"    - {search_path.parent}")
            return 1
    else:
        print(f"Using input tensor: {input_path}")
    
    metadata = dump_layer_outputs(args.model, str(input_path), args.output, args.layers)
    
    if metadata:
        print("\n✓ Golden reference dump completed successfully")
        return 0
    else:
        print("\n✗ Golden reference dump failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
