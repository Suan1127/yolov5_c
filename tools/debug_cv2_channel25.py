#!/usr/bin/env python3
"""
cv2 채널 25의 가중치와 출력을 확인하는 스크립트
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add yolov5 to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'third_party' / 'yolov5'))

def load_tensor_bin(filepath):
    """Load tensor from binary file (dims + data)"""
    with open(filepath, 'rb') as f:
        dims = np.frombuffer(f.read(4 * 4), dtype=np.int32)
        n, c, h, w = dims
        data = np.frombuffer(f.read(n * c * h * w * 4), dtype=np.float32)
        return data.reshape(n, c, h, w)

def main():
    model_path = "weights/yolov5s.pt"
    layer1_path = "testdata/python/layer_001.bin"
    
    print("=== cv2 채널 25 디버깅 ===\n")
    
    # Load Layer 1 output (C3 input)
    print(f"1. Loading Layer 1 output from {layer1_path}...")
    layer1_output = load_tensor_bin(layer1_path)
    print(f"   Shape: {layer1_output.shape}\n")
    
    # Load PyTorch model
    print("2. Loading PyTorch model...")
    try:
        from models.common import DetectMultiBackend
        model = DetectMultiBackend(model_path, device='cpu', dnn=False, data=None, fp16=False)
        model.eval()
    except:
        model = torch.load(model_path, map_location='cpu')
        if hasattr(model, 'eval'):
            model.eval()
    
    # Find C3 module (Layer 2)
    if hasattr(model, 'model'):
        model_obj = model.model
    else:
        model_obj = model
    
    c3_module = None
    for name, module in model_obj.named_modules():
        if '2' in name and hasattr(module, 'cv1'):
            c3_module = module
            break
    
    if c3_module is None:
        print("Error: Could not find C3 module")
        sys.exit(1)
    
    print(f"   Found C3 module: {type(c3_module)}\n")
    
    # Convert to torch tensor
    input_torch = torch.from_numpy(layer1_output.copy())
    
    # Get cv2 weights and bias
    # cv2 is a Conv object (Conv + BN)
    print("3. cv2 구조 확인:")
    print(f"   cv2 type: {type(c3_module.cv2)}")
    print(f"   cv2 attributes: {dir(c3_module.cv2)}")
    
    # Check if cv2 has conv and bn attributes
    if hasattr(c3_module.cv2, 'conv'):
        cv2_conv = c3_module.cv2.conv
        cv2_bn = c3_module.cv2.bn if hasattr(c3_module.cv2, 'bn') else None
    else:
        # Fused model - cv2 might be a nn.Conv2d directly
        cv2_conv = c3_module.cv2
        cv2_bn = None
    
    print("4. cv2 가중치 정보:")
    print(f"   cv2.conv.weight shape: {cv2_conv.weight.shape}")
    print(f"   cv2.conv.bias shape: {cv2_conv.bias.shape if cv2_conv.bias is not None else None}")
    print(f"   cv2.conv.bias is not None: {cv2_conv.bias is not None}")
    
    # Check channel 25
    oc = 25
    print(f"\n5. 채널 {oc} 정보:")
    if cv2_conv.bias is not None:
        print(f"   bias[{oc}] = {cv2_conv.bias[oc].item():.6f}")
    else:
        print(f"   bias is None")
    
    print(f"   가중치 샘플 (first 10 input channels):")
    for ic in range(min(10, cv2_conv.weight.shape[1])):
        w_val = cv2_conv.weight[oc, ic].item()
        print(f"     weight[{oc},{ic}] = {w_val:.6f}")
    
    print(f"   가중치 샘플 (last 10 input channels):")
    start_ic = max(0, cv2_conv.weight.shape[1] - 10)
    for ic in range(start_ic, cv2_conv.weight.shape[1]):
        w_val = cv2_conv.weight[oc, ic].item()
        print(f"     weight[{oc},{ic}] = {w_val:.6f}")
    
    # Run cv2 forward
    print("\n6. cv2 forward pass:")
    with torch.no_grad():
        cv2_out = c3_module.cv2(input_torch)
        print(f"   Output shape: {cv2_out.shape}")
        print(f"   Output range: [{cv2_out.min():.6f}, {cv2_out.max():.6f}]")
        
        # Check channel 25 at position (0, 25, 67, 130)
        h, w = 67, 130
        pytorch_val = cv2_out[0, oc, h, w].item()
        print(f"   PyTorch output[0,{oc},{h},{w}] = {pytorch_val:.6f}")
        
        # Manual calculation
        manual_sum = cv2_conv.bias[oc].item() if cv2_conv.bias is not None else 0.0
        print(f"\n7. 수동 계산 (위치 [{h},{w}]):")
        print(f"   bias = {manual_sum:.6f}")
        for ic in range(min(10, cv2_conv.weight.shape[1])):
            in_val = input_torch[0, ic, h, w].item()
            w_val = cv2_conv.weight[oc, ic].item()
            product = in_val * w_val
            manual_sum += product
            print(f"   ic{ic}: in={in_val:.6f} * w={w_val:.6f} = {product:.6f}, sum={manual_sum:.6f}")
        
        # Full calculation
        full_sum = cv2_conv.bias[oc].item() if cv2_conv.bias is not None else 0.0
        for ic in range(cv2_conv.weight.shape[1]):
            in_val = input_torch[0, ic, h, w].item()
            w_val = cv2_conv.weight[oc, ic].item()
            full_sum += in_val * w_val
        
        print(f"\n   전체 계산 결과: {full_sum:.6f}")
        print(f"   PyTorch 결과: {pytorch_val:.6f}")
        print(f"   차이: {abs(full_sum - pytorch_val):.6e}")
        
        # Check if BN is applied
        if cv2_bn is not None:
            print(f"\n8. BN 정보:")
            print(f"   cv2.bn.weight shape: {cv2_bn.weight.shape}")
            print(f"   cv2.bn.bias shape: {cv2_bn.bias.shape}")
            print(f"   cv2.bn.running_mean[{oc}] = {cv2_bn.running_mean[oc].item():.6f}")
            print(f"   cv2.bn.running_var[{oc}] = {cv2_bn.running_var[oc].item():.6f}")
            print(f"   cv2.bn.weight[{oc}] = {cv2_bn.weight[oc].item():.6f}")
            print(f"   cv2.bn.bias[{oc}] = {cv2_bn.bias[oc].item():.6f}")
        else:
            print(f"\n8. BN 정보: cv2 is fused (no separate BN layer)")
        
        # Check if cv2 has activation (it shouldn't)
        print(f"\n9. cv2 구조 확인:")
        if hasattr(c3_module.cv2, 'act'):
            print(f"   cv2.act: {c3_module.cv2.act}")
        print(f"   cv2 modules: {list(c3_module.cv2.named_children())}")

if __name__ == '__main__':
    main()
