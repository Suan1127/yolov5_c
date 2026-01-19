#!/usr/bin/env python3
"""
Dump golden reference tensors from PyTorch YOLOv5s model
Generates intermediate layer outputs for comparison with C implementation
"""

import torch
import numpy as np
import json
import argparse
from pathlib import Path
import sys

# Add yolov5 to path if needed
# sys.path.insert(0, 'third_party/yolov5')


def dump_layer_outputs(model_path, input_tensor_path, output_dir, save_layers=None):
    """
    Run inference and dump intermediate layer outputs
    """
    if save_layers is None:
        # Default save layers from PROJECT_BRIEF.md
        save_layers = [3, 5, 6, 7, 9, 17, 20, 23]
    
    # Load model
    model = torch.load(model_path, map_location='cpu')
    if isinstance(model, dict):
        if 'model' in model:
            model = model['model']
    
    model.eval()
    
    # Load input tensor
    input_tensor = np.fromfile(input_tensor_path, dtype=np.int32, count=4)
    n, c, h, w = input_tensor
    input_data = np.fromfile(input_tensor_path, dtype=np.float32, offset=16)
    input_data = input_data.reshape(n, c, h, w)
    input_tensor = torch.from_numpy(input_data)
    
    # Hook to capture intermediate outputs
    outputs = {}
    
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                outputs[name] = output.detach().cpu().numpy()
        return hook
    
    # Register hooks for save layers
    handles = []
    # TODO: Register hooks based on actual model structure
    # This is a placeholder - actual implementation depends on model architecture
    
    # Run forward pass
    with torch.no_grad():
        output = model(input_tensor)
    
    # Save outputs
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "save_layers": save_layers,
        "input_shape": list(input_tensor.shape),
        "outputs": {}
    }
    
    for layer_idx, tensor in outputs.items():
        output_path = output_dir / f"layer_{layer_idx}.bin"
        
        # Save in same format as C tensor_dump
        with open(output_path, 'wb') as f:
            shape = np.array(tensor.shape, dtype=np.int32)
            shape.tofile(f)
            tensor.astype(np.float32).tofile(f)
        
        metadata["outputs"][layer_idx] = {
            "shape": list(tensor.shape),
            "path": str(output_path)
        }
    
    # Save metadata
    meta_path = output_dir / "golden_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Dumped {len(outputs)} layer outputs to {output_dir}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description='Dump golden reference tensors')
    parser.add_argument('model', type=str, help='Path to yolov5s.pt')
    parser.add_argument('input', type=str, help='Path to input tensor .bin file')
    parser.add_argument('--output', type=str, default='data/golden', help='Output directory')
    parser.add_argument('--layers', type=int, nargs='+', default=[3, 5, 6, 7, 9, 17, 20, 23],
                        help='Layer indices to save')
    
    args = parser.parse_args()
    
    dump_layer_outputs(args.model, args.input, args.output, args.layers)


if __name__ == '__main__':
    main()
