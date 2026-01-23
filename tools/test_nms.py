#!/usr/bin/env python3
"""
Test NMS logic to understand why Python returns 4 detections but C returns 9
"""

import numpy as np
import torch
from torchvision.ops import nms as torchvision_nms
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'third_party' / 'yolov5'))
from utils.general import xywh2xyxy

# C results (Class 0 detections)
c_detections = [
    (0.8231, 0.3816, 0.5816, 0.1322, 0.4333),
    (0.8158, 0.2411, 0.5933, 0.1527, 0.6846),
    (0.7088, 0.4070, 0.5794, 0.1253, 0.8222),
    (0.6757, 0.8212, 0.5762, 0.1354, 0.4783),
    (0.6716, 0.2615, 0.5851, 0.1493, 1.3849),
    (0.6708, 0.8212, 0.5761, 0.1284, 1.1969),
]

# Python results (Class 0 detections)
python_detections = [
    (0.8231, 0.3883, 0.5882, 0.1205, 0.4212),
    (0.8158, 0.2448, 0.5991, 0.1434, 0.4600),
    (0.6757, 0.8174, 0.5774, 0.1244, 0.4563),
]

print("C detections (Class 0):")
for i, (conf, x, y, w, h) in enumerate(c_detections):
    xyxy = xywh2xyxy(np.array([[x, y, w, h]]))[0]
    print(f"  {i}: conf={conf:.4f}, xywh=({x:.4f}, {y:.4f}, {w:.4f}, {h:.4f}), xyxy={xyxy}")

print("\nPython detections (Class 0):")
for i, (conf, x, y, w, h) in enumerate(python_detections):
    xyxy = xywh2xyxy(np.array([[x, y, w, h]]))[0]
    print(f"  {i}: conf={conf:.4f}, xywh=({x:.4f}, {y:.4f}, {w:.4f}, {h:.4f}), xyxy={xyxy}")

# Convert C detections to xyxy format
c_boxes = []
c_scores = []
for conf, x, y, w, h in c_detections:
    xyxy = xywh2xyxy(np.array([[x, y, w, h]]))[0]
    c_boxes.append(xyxy)
    c_scores.append(conf)

c_boxes = torch.tensor(c_boxes, dtype=torch.float32)
c_scores = torch.tensor(c_scores, dtype=torch.float32)

print("\nApplying torchvision.ops.nms to C detections...")
keep_indices = torchvision_nms(c_boxes, c_scores, 0.45)
print(f"Kept indices: {keep_indices.tolist()}")
print(f"Number kept: {len(keep_indices)}")

print("\nKept detections:")
for idx in keep_indices:
    conf, x, y, w, h = c_detections[idx]
    print(f"  {idx}: conf={conf:.4f}, xywh=({x:.4f}, {y:.4f}, {w:.4f}, {h:.4f})")

# Calculate IoU between pairs
print("\nIoU between C detections:")
from torchvision.ops import box_iou
for i in range(len(c_detections)):
    for j in range(i + 1, len(c_detections)):
        iou = box_iou(c_boxes[i:i+1], c_boxes[j:j+1])[0, 0].item()
        if iou > 0.45:
            print(f"  Detection {i} and {j}: IoU = {iou:.4f} > 0.45 (should suppress {j})")
