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
        # Save all layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        save_layers = list(range(24))  # Layers 0-23 (layer 24 is Detect, skip it)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load input tensor
    input_tensor_path = Path(input_tensor_path)
    if not input_tensor_path.exists():
        print(f"Error: Input tensor file not found: {input_tensor_path}")
        return None
    
    if input_tensor_path.is_dir():
        print(f"Error: Input tensor path is a directory, not a file: {input_tensor_path}")
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
    detect_conv_outputs = {}  # Store detect head conv outputs before reshape
    
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
    
    # Hook for Detect head conv outputs (before reshape)
    # Detect head forward: x[i] = self.m[i](x[i]) -> (1, 255, ny, nx)
    # We need to capture this conv output before reshape
    def register_detect_hooks(model_modules):
        """Register hooks on Detect head conv layers to capture (1, 255, H, W) outputs"""
        detect_handles = []
        for name, module in model_modules.named_modules():
            if isinstance(module, torch.nn.Module) and hasattr(module, 'm') and isinstance(module.m, torch.nn.ModuleList):
                # This is a Detect module
                print(f"  Found Detect module at {name}, registering conv hooks...")
                for i, conv in enumerate(module.m):
                    def make_conv_hook(idx):
                        def conv_hook(conv_module, conv_input, conv_output):
                            # conv_output is (bs, 255, ny, nx) - this is what C saves!
                            detect_conv_outputs[f"detect_conv_{idx}"] = conv_output.detach().cpu().numpy()
                        return conv_hook
                    handle = conv.register_forward_hook(make_conv_hook(i))
                    detect_handles.append(handle)
                    print(f"    Registered hook for detect conv {i}")
        return detect_handles
    
    # Register hooks for save layers
    # YOLOv5 model structure: model.model is a Sequential container
    # We can directly access layers by index: model.model[i]
    if hasattr(model, 'model'):
        # YOLOv5 model structure
        model_modules = model.model if hasattr(model, 'model') else model
        
        # Debug: Print model structure
        print(f"\nModel structure info:")
        print(f"  Type: {type(model_modules)}")
        if hasattr(model_modules, '__len__'):
            print(f"  Length: {len(model_modules)}")
        
        # Try to access layers by index (for Sequential models)
        if hasattr(model_modules, '__len__') and hasattr(model_modules, '__getitem__'):
            # Direct index access (e.g., model.model[0], model.model[1], ...)
            print(f"\nRegistering hooks using direct index access...")
            for layer_idx in save_layers:
                try:
                    if layer_idx < len(model_modules):
                        module = model_modules[layer_idx]
                        # Check module type for debugging
                        module_type = type(module).__name__
                        handle = module.register_forward_hook(make_hook(layer_idx))
                        handles.append(handle)
                        print(f"  ✓ Registered hook for layer {layer_idx} (model.model[{layer_idx}], type={module_type})")
                    else:
                        print(f"  ✗ Layer {layer_idx} out of range (model length: {len(model_modules)})")
                except (IndexError, AttributeError, TypeError) as e:
                    print(f"  ✗ Could not register hook for layer {layer_idx}: {e}")
        
        # Fallback: named_modules approach (if direct indexing doesn't work or didn't register all)
        if len(handles) < len(save_layers) and hasattr(model_modules, 'named_modules'):
            print(f"\nRegistering hooks using named_modules (fallback)...")
            registered_indices = set()
            for name, module in model_modules.named_modules():
                # Try to match layer indices from module name
                try:
                    # Extract layer number from name (e.g., "model.3" -> 3)
                    if '.' in name:
                        parts = name.split('.')
                        for part in parts:
                            if part.isdigit():
                                layer_idx = int(part)
                                if layer_idx in save_layers and layer_idx not in registered_indices:
                                    module_type = type(module).__name__
                                    handle = module.register_forward_hook(make_hook(layer_idx))
                                    handles.append(handle)
                                    registered_indices.add(layer_idx)
                                    print(f"  ✓ Registered hook for layer {layer_idx} ({name}, type={module_type})")
                                    break
                except Exception as e:
                    pass
        
        # Register hooks for Detect head conv outputs
        detect_handles = register_detect_hooks(model_modules)
        handles.extend(detect_handles)
    
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
        # YOLOv5 Detect head returns: (torch.cat(z, 1), x) where:
        # - output[0]: concatenated tensor (1, 25200, 85)
        # - output[1]: list of 3 tensors [P3, P4, P5] before concatenation
        for i, out in enumerate(output):
            if isinstance(out, torch.Tensor):
                output_path = output_dir / f"output_{i}.bin"
                output_np = out.detach().cpu().numpy()
                save_tensor_bin(output_np, output_path)
                print(f"  Final output[{i}]: shape {output_np.shape} -> {output_path.name}")
                saved_count += 1
            elif isinstance(out, (list, tuple)):
                # output[1] is list of reshaped tensors (1, 3, ny, nx, 85) - not what we want
                # We'll use detect_conv_outputs instead
                pass
    
    # Save Detect head conv outputs (for comparison with C implementation)
    # These are (1, 255, H, W) tensors before reshape
    if detect_conv_outputs:
        print(f"\nSaving Detect head conv outputs...")
        # Map: detect_conv_0 -> P3 (80x80), detect_conv_1 -> P4 (40x40), detect_conv_2 -> P5 (20x20)
        for i in range(3):
            key = f"detect_conv_{i}"
            if key in detect_conv_outputs:
                output_np = detect_conv_outputs[key]
                # Save as output_1_0.bin, output_1_1.bin, output_1_2.bin to match C naming
                output_path = output_dir / f"output_1_{i}.bin"
                save_tensor_bin(output_np, output_path)
                print(f"  Detect conv {i}: shape {output_np.shape} -> {output_path.name}")
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
    parser.add_argument('--output', type=str, default='testdata_n/python', 
                        help='Output directory (default: testdata_n/python)')
    parser.add_argument('--layers', type=int, nargs='+', default=None,
                        help='Layer indices to save (default: all layers 0-23)')
    
    args = parser.parse_args()
    
    # Resolve input tensor path
    input_path = Path(args.input)
    
    # Check if input_path is a directory (should not be)
    if input_path.exists() and input_path.is_dir():
        print(f"Error: Input path is a directory, not a file: {input_path}")
        print(f"  Please provide a file path or image name (e.g., 'bus' or 'data/inputs/bus.bin')")
        return 1
    
    # If input doesn't exist as a file, try to find it as image name
    if not input_path.exists() or input_path.is_dir():
        # Try as image name (without extension)
        image_name = input_path.stem if input_path.suffix else args.input
        
        # Search in multiple locations (including YOLOv5n paths)
        search_paths = [
            Path('data/yolov5n/inputs') / f"{image_name}.bin",
            Path('data/yolov5s/inputs') / f"{image_name}.bin",
            Path('data/inputs') / f"{image_name}.bin",
            Path('testdata_n/python') / f"{image_name}.bin",
            Path('testdata_n/c') / f"{image_name}.bin",
            Path('testdata/python') / f"{image_name}.bin",
            Path('testdata/c') / f"{image_name}.bin",
            Path(args.output) / f"{image_name}.bin",
        ]
        
        found = False
        for search_path in search_paths:
            if search_path.exists() and search_path.is_file():
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
        # Verify it's actually a file
        if not input_path.is_file():
            print(f"Error: Input path exists but is not a file: {input_path}")
            return 1
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
