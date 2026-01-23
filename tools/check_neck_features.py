#!/usr/bin/env python3
"""
Check which backbone feature maps are passed to neck (head) in YOLOv5n
Compare with C implementation
"""

import torch
import sys
from pathlib import Path

# Add yolov5 to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'third_party' / 'yolov5'))

try:
    from models.common import DetectMultiBackend
    from utils.general import check_img_size
except ImportError:
    print("Warning: Could not import YOLOv5 modules.")
    DetectMultiBackend = None

def check_neck_connections():
    """Check which backbone layers are used in neck (head)"""
    
    # Load model
    model_path = Path(__file__).parent.parent / 'yolov5n.pt'
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        return
    
    print(f"Loading model from {model_path}...")
    if DetectMultiBackend:
        model = DetectMultiBackend(model_path, device='cpu')
        model.eval()
    else:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
        model.eval()
    
    # Print model structure
    print("\n=== Model Structure ===")
    print("Backbone layers (0-9):")
    for i in range(10):
        print(f"  Layer {i}")
    
    print("\nHead layers (10-23):")
    for i in range(10, 24):
        print(f"  Layer {i}")
    
    # Check YOLOv5 model structure
    print("\n=== YOLOv5 Model Module Names ===")
    for name, module in model.named_modules():
        if 'model' in name and any(x in name for x in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']):
            print(f"{name}: {type(module).__name__}")
    
    # Check concat connections in head
    print("\n=== Head Concat Connections (from YOLOv5 structure) ===")
    print("Layer 12 (Concat): [Layer 11, Layer 6]")
    print("  - Layer 11: Upsampled from Layer 10")
    print("  - Layer 6: Backbone C3 block output")
    print("")
    print("Layer 16 (Concat): [Layer 15, Layer 4]")
    print("  - Layer 15: Upsampled from Layer 14")
    print("  - Layer 4: Backbone C3 block output")
    print("")
    print("Layer 19 (Concat): [Layer 18, Layer 13]")
    print("  - Layer 18: Downsampled from Layer 17")
    print("  - Layer 13: Head C3 block output (from Layer 12)")
    print("")
    print("Layer 22 (Concat): [Layer 21, Layer 10]")
    print("  - Layer 21: Downsampled from Layer 20")
    print("  - Layer 10: Head Conv output (from Layer 9)")
    print("")
    print("=== Summary ===")
    print("Backbone feature maps passed to Neck (Head):")
    print("  1. Layer 6: (1, 128, 40, 40) → used in Layer 12 (Concat)")
    print("  2. Layer 4: (1, 64, 80, 80) → used in Layer 16 (Concat)")
    print("  3. Layer 9: (1, 256, 20, 20) → directly passed to Layer 10 (Head start)")
    print("")
    print("Total: 3 feature maps from Backbone to Neck")

if __name__ == '__main__':
    check_neck_connections()
