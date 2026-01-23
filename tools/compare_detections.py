#!/usr/bin/env python3
"""
Compare detection results between Python and C implementations
Compares detections.txt files in same format
"""

import argparse
from pathlib import Path
import sys


def parse_detections_file(filepath):
    """Parse detections.txt file and return list of detections"""
    detections = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
        # Skip header lines
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith('Image:') or line.startswith('Total') or line.startswith('Format:'):
                i += 1
                continue
            
            # Parse detection line (normalized coordinates)
            parts = line.split()
            if len(parts) == 6:
                cls_id = int(parts[0])
                conf = float(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                
                # Skip pixel coordinates line (next line)
                if i + 1 < len(lines):
                    i += 1
                
                detections.append({
                    'cls_id': cls_id,
                    'conf': conf,
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h
                })
            i += 1
    
    return detections


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes (x, y, w, h format)"""
    # Convert to xyxy format
    x1_1 = box1['x'] - box1['w'] / 2
    y1_1 = box1['y'] - box1['h'] / 2
    x2_1 = box1['x'] + box1['w'] / 2
    y2_1 = box1['y'] + box1['h'] / 2
    
    x1_2 = box2['x'] - box2['w'] / 2
    y1_2 = box2['y'] - box2['h'] / 2
    x2_2 = box2['x'] + box2['w'] / 2
    y2_2 = box2['y'] + box2['h'] / 2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = box1['w'] * box1['h']
    area2 = box2['w'] * box2['h']
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def match_detections(golden_dets, test_dets, iou_threshold=0.5, conf_threshold=0.01):
    """Match detections between golden and test using IoU and class ID"""
    matched = []
    test_matched = set()
    
    for g_det in golden_dets:
        best_match = None
        best_iou = 0.0
        
        for j, t_det in enumerate(test_dets):
            if j in test_matched:
                continue
            
            # Must be same class
            if g_det['cls_id'] != t_det['cls_id']:
                continue
            
            # Calculate IoU
            iou = calculate_iou(g_det, t_det)
            
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_match = j
        
        if best_match is not None:
            matched.append((g_det, test_dets[best_match], best_iou))
            test_matched.add(best_match)
        else:
            matched.append((g_det, None, 0.0))
    
    # Find unmatched test detections
    unmatched_test = [test_dets[i] for i in range(len(test_dets)) if i not in test_matched]
    
    return matched, unmatched_test


def compare_detections(golden_path, test_path, iou_threshold=0.5, conf_tolerance=0.01):
    """Compare two detection result files"""
    golden_dets = parse_detections_file(golden_path)
    test_dets = parse_detections_file(test_path)
    
    print(f"Golden detections: {len(golden_dets)}")
    print(f"Test detections: {len(test_dets)}")
    print()
    
    # Match detections
    matched, unmatched_test = match_detections(golden_dets, test_dets, iou_threshold, conf_tolerance)
    
    # Count matches
    num_matched = sum(1 for _, t, _ in matched if t is not None)
    num_unmatched_golden = sum(1 for _, t, _ in matched if t is None)
    num_unmatched_test = len(unmatched_test)
    
    print(f"Matched: {num_matched}/{len(golden_dets)}")
    print(f"Unmatched in golden: {num_unmatched_golden}")
    print(f"Unmatched in test: {num_unmatched_test}")
    print()
    
    # Compare matched detections
    if num_matched > 0:
        print("Comparing matched detections...")
        conf_diffs = []
        ious = []
        
        for g_det, t_det, iou in matched:
            if t_det is not None:
                conf_diff = abs(g_det['conf'] - t_det['conf'])
                conf_diffs.append(conf_diff)
                ious.append(iou)
        
        if conf_diffs:
            avg_conf_diff = sum(conf_diffs) / len(conf_diffs)
            max_conf_diff = max(conf_diffs)
            avg_iou = sum(ious) / len(ious)
            min_iou = min(ious)
            
            print(f"  Average confidence difference: {avg_conf_diff:.6f}")
            print(f"  Max confidence difference: {max_conf_diff:.6f}")
            print(f"  Average IoU: {avg_iou:.4f}")
            print(f"  Min IoU: {min_iou:.4f}")
            print()
            
            # Check if within tolerance
            if max_conf_diff < conf_tolerance and min_iou >= iou_threshold:
                print("✓ All matched detections are within tolerance")
                return True
            else:
                print("✗ Some matched detections are outside tolerance")
                return False
    
    # Check if counts match
    if num_unmatched_golden == 0 and num_unmatched_test == 0:
        print("✓ Detection counts match perfectly")
        return True
    else:
        print("✗ Detection counts do not match")
        if num_unmatched_golden > 0:
            print(f"  {num_unmatched_golden} detections in golden not found in test")
        if num_unmatched_test > 0:
            print(f"  {num_unmatched_test} detections in test not found in golden")
        return False


def main():
    parser = argparse.ArgumentParser(description='Compare detection results between Python and C')
    parser.add_argument('golden', type=str, 
                        help='Golden detection file (e.g., testdata_n/python/bus_detections.txt)')
    parser.add_argument('test', type=str,
                        help='Test detection file (e.g., data/yolov5n/outputs/bus_detections.txt)')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                        help='IoU threshold for matching detections (default: 0.5)')
    parser.add_argument('--conf-tolerance', type=float, default=0.01,
                        help='Confidence difference tolerance (default: 0.01)')
    
    args = parser.parse_args()
    
    golden_path = Path(args.golden)
    test_path = Path(args.test)
    
    if not golden_path.exists():
        print(f"Error: Golden file does not exist: {golden_path}")
        sys.exit(1)
    
    if not test_path.exists():
        print(f"Error: Test file does not exist: {test_path}")
        sys.exit(1)
    
    print(f"Comparing detection results:")
    print(f"  Golden: {golden_path}")
    print(f"  Test: {test_path}")
    print(f"  IoU threshold: {args.iou_threshold}")
    print(f"  Confidence tolerance: {args.conf_tolerance}")
    print()
    
    success = compare_detections(golden_path, test_path, args.iou_threshold, args.conf_tolerance)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
