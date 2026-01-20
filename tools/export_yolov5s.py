#!/usr/bin/env python3
"""
Export YOLOv5s model weights and metadata for C inference
Converts .pt model to weights.bin and model_meta.json
Uses third_party/yolov5 for model loading
"""

import sys
import torch
import json
import argparse
import numpy as np
from pathlib import Path
from collections import OrderedDict

# Add third_party/yolov5 to path
YOLOV5_ROOT = Path(__file__).parent.parent / "third_party" / "yolov5"
if str(YOLOV5_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOV5_ROOT))

# Import YOLOv5 utilities
try:
    from models.experimental import attempt_load
    from utils.general import LOGGER
except ImportError as e:
    print(f"Error importing YOLOv5 modules: {e}")
    print(f"Make sure third_party/yolov5 is properly cloned and dependencies are installed")
    print(f"Run: cd third_party/yolov5 && pip install -r requirements.txt")
    sys.exit(1)


def make_divisible(x, divisor=8):
    """Make x divisible by divisor"""
    return int(np.ceil(x / divisor) * divisor)


def export_weights(model_path, output_dir, depth_multiple=0.33, width_multiple=0.50):
    """
    Export YOLOv5s weights to binary format using YOLOv5's attempt_load
    """
    print(f"Loading YOLOv5 model from {model_path}...")
    
    # Load model using YOLOv5's attempt_load (handles .pt format correctly)
    # Note: fuse=True to match inference behavior (Conv+BN are fused)
    try:
        model = attempt_load(model_path, device='cpu', inplace=False, fuse=True)
    except Exception as e:
        print(f"Error loading model with attempt_load: {e}")
        print("Falling back to torch.load...")
        # Fallback to direct torch.load
        ckpt = torch.load(model_path, map_location='cpu')
        if isinstance(ckpt, dict):
            if 'model' in ckpt:
                model = ckpt['model']
            elif 'ema' in ckpt:
                model = ckpt['ema']
            else:
                raise ValueError("Unknown model format in checkpoint")
        else:
            model = ckpt
    
    # Get model configuration
    if hasattr(model, 'yaml'):
        yaml_cfg = model.yaml
        if isinstance(yaml_cfg, dict):
            actual_depth = yaml_cfg.get('depth_multiple', depth_multiple)
            actual_width = yaml_cfg.get('width_multiple', width_multiple)
            if actual_depth != depth_multiple or actual_width != width_multiple:
                print(f"Warning: Model config has depth_multiple={actual_depth}, width_multiple={actual_width}")
                print(f"Using provided values: depth_multiple={depth_multiple}, width_multiple={width_multiple}")
    else:
        print(f"Model doesn't have yaml config, using provided values")
    
    # Get state dict
    state_dict = model.state_dict() if hasattr(model, 'state_dict') else model
    
    # Calculate actual channels based on width_multiple
    def get_actual_channels(base_channels):
        return make_divisible(base_channels * width_multiple, 8)
    
    # Calculate actual repeats based on depth_multiple
    def get_actual_repeats(base_repeats):
        return max(round(base_repeats * depth_multiple), 1)
    
    # Extract weights and metadata
    weights_data = []
    weights_map = OrderedDict()
    offset = 0
    
    # Process each layer (sort by name for consistent ordering)
    sorted_items = sorted(state_dict.items())
    
    for name, param in sorted_items:
        if 'weight' in name or 'bias' in name:
            param_np = param.detach().cpu().numpy()
            param_bytes = param_np.tobytes()
            
            weights_map[name] = {
                "offset": offset,
                "shape": list(param_np.shape),
                "dtype": str(param_np.dtype),
                "numel": int(param_np.size)
            }
            
            weights_data.append(param_bytes)
            offset += len(param_bytes)
    
    print(f"Found {len(weights_map)} weight/bias tensors")
    
    # Concatenate all weights
    weights_binary = b''.join(weights_data)
    
    # Save weights
    weights_path = Path(output_dir) / "weights.bin"
    with open(weights_path, 'wb') as f:
        f.write(weights_binary)
    
    print(f"Saved weights to {weights_path} ({len(weights_binary)} bytes)")
    
    # Save weights map
    weights_map_path = Path(output_dir) / "weights_map.json"
    with open(weights_map_path, 'w') as f:
        json.dump(weights_map, f, indent=2)
    
    print(f"Saved weights map to {weights_map_path}")
    
    # Get model info
    num_classes = 80
    if hasattr(model, 'nc'):
        num_classes = model.nc
    elif hasattr(model, 'yaml') and isinstance(model.yaml, dict):
        num_classes = model.yaml.get('nc', 80)
    
    # Get anchors if available
    anchors = {
        "p3": [10, 13, 16, 30, 33, 23],
        "p4": [30, 61, 62, 45, 59, 119],
        "p5": [116, 90, 156, 198, 373, 326]
    }
    if hasattr(model, 'yaml') and isinstance(model.yaml, dict):
        yaml_anchors = model.yaml.get('anchors', None)
        if yaml_anchors:
            if isinstance(yaml_anchors, list) and len(yaml_anchors) >= 3:
                anchors = {
                    "p3": yaml_anchors[0] if len(yaml_anchors[0]) == 6 else anchors["p3"],
                    "p4": yaml_anchors[1] if len(yaml_anchors[1]) == 6 else anchors["p4"],
                    "p5": yaml_anchors[2] if len(yaml_anchors[2]) == 6 else anchors["p5"]
                }
    
    # Create model metadata
    model_meta = {
        "depth_multiple": depth_multiple,
        "width_multiple": width_multiple,
        "num_classes": num_classes,
        "input_size": 640,
        "anchors": anchors,
        "channels": {
            "base": [64, 128, 256, 512, 1024],
            "actual": [get_actual_channels(c) for c in [64, 128, 256, 512, 1024]]
        },
        "total_weights_size": len(weights_binary),
        "num_parameters": len(weights_map)
    }
    
    # Save model metadata
    meta_path = Path(output_dir) / "model_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(model_meta, f, indent=2)
    
    print(f"Saved model metadata to {meta_path}")
    
    return weights_path, weights_map_path, meta_path


def main():
    parser = argparse.ArgumentParser(description='Export YOLOv5s weights for C inference')
    parser.add_argument('model', type=str, help='Path to yolov5s.pt')
    parser.add_argument('--output', type=str, default='weights', help='Output directory')
    parser.add_argument('--depth', type=float, default=0.33, help='Depth multiple')
    parser.add_argument('--width', type=float, default=0.50, help='Width multiple')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Export
    export_weights(args.model, args.output, args.depth, args.width)


if __name__ == '__main__':
    main()
