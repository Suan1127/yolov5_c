#!/usr/bin/env python3
"""
Layer 2 (C3 블록) 디버깅 스크립트
C3 블록의 각 단계별 출력을 확인하여 어디서 문제가 발생하는지 찾습니다.
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

def save_tensor_bin(tensor, filepath):
    """Save tensor to binary file (dims + data)"""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    n, c, h, w = tensor.shape
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        f.write(np.array([n, c, h, w], dtype=np.int32).tobytes())
        f.write(tensor.astype(np.float32).tobytes())

def main():
    model_path = "weights/yolov5s.pt"
    layer1_path = "testdata/python/layer_001.bin"
    
    print("=== Layer 2 (C3) 디버깅 ===\n")
    
    # Load Layer 1 output (C3 input)
    print(f"1. Loading Layer 1 output from {layer1_path}...")
    layer1_output = load_tensor_bin(layer1_path)
    print(f"   Shape: {layer1_output.shape}")
    print(f"   Range: [{layer1_output.min():.6f}, {layer1_output.max():.6f}]")
    print(f"   Sample [0,0,0,0]: {layer1_output[0,0,0,0]:.6f}\n")
    
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
    
    # Step-by-step C3 forward
    print("3. C3 블록 단계별 실행:\n")
    
    with torch.no_grad():
        # Step 1: cv1 (main path)
        print("   Step 1: cv1 (Conv + BN + SiLU)...")
        cv1_out = c3_module.cv1(input_torch)
        print(f"      Output shape: {cv1_out.shape}")
        print(f"      Output range: [{cv1_out.min():.6f}, {cv1_out.max():.6f}]")
        print(f"      Sample [0,0,0,0]: {cv1_out[0,0,0,0]:.6f}")
        save_tensor_bin(cv1_out, 'debug/pytorch/c3_cv1_output.bin')
        
        # Step 2: Bottleneck
        print("\n   Step 2: Bottleneck...")
        bottleneck = c3_module.m[0]
        bottleneck_out = bottleneck(cv1_out)
        print(f"      Output shape: {bottleneck_out.shape}")
        print(f"      Output range: [{bottleneck_out.min():.6f}, {bottleneck_out.max():.6f}]")
        print(f"      Sample [0,0,0,0]: {bottleneck_out[0,0,0,0]:.6f}")
        save_tensor_bin(bottleneck_out, 'debug/pytorch/c3_bottleneck_output.bin')
        
        # Step 3: cv2 (skip path)
        print("\n   Step 3: cv2 (skip path, Conv + BN)...")
        cv2_out = c3_module.cv2(input_torch)
        print(f"      Output shape: {cv2_out.shape}")
        print(f"      Output range: [{cv2_out.min():.6f}, {cv2_out.max():.6f}]")
        print(f"      Sample [0,0,0,0]: {cv2_out[0,0,0,0]:.6f}")
        save_tensor_bin(cv2_out, 'debug/pytorch/c3_cv2_output.bin')
        
        # Step 4: Concat
        print("\n   Step 4: Concat [bottleneck_out, cv2_out]...")
        concat_out = torch.cat([bottleneck_out, cv2_out], dim=1)
        print(f"      Output shape: {concat_out.shape}")
        print(f"      Output range: [{concat_out.min():.6f}, {concat_out.max():.6f}]")
        print(f"      Sample [0,0,0,0]: {concat_out[0,0,0,0]:.6f}")
        save_tensor_bin(concat_out, 'debug/pytorch/c3_concat_output.bin')
        
        # Step 5: cv3 (final)
        print("\n   Step 5: cv3 (Conv + BN + SiLU)...")
        cv3_out = c3_module.cv3(concat_out)
        print(f"      Output shape: {cv3_out.shape}")
        print(f"      Output range: [{cv3_out.min():.6f}, {cv3_out.max():.6f}]")
        print(f"      Sample [0,0,0,0]: {cv3_out[0,0,0,0]:.6f}")
        save_tensor_bin(cv3_out, 'debug/pytorch/c3_final_output.bin')
        
        # Compare with full C3 output
        print("\n   Step 6: Full C3 forward (검증용)...")
        full_c3_out = c3_module(input_torch)
        diff = torch.abs(full_c3_out - cv3_out)
        print(f"      Step-by-step vs Full diff: max={diff.max():.6e}, mean={diff.mean():.6e}")
    
    print("\n=== 완료 ===")
    print("PyTorch 중간 출력이 debug/pytorch/에 저장되었습니다:")
    print("  - c3_cv1_output.bin")
    print("  - c3_bottleneck_output.bin")
    print("  - c3_cv2_output.bin")
    print("  - c3_concat_output.bin")
    print("  - c3_final_output.bin")
    print("\n다음 단계: C 구현과 비교하여 어느 단계에서 문제가 발생하는지 확인하세요.")

if __name__ == '__main__':
    main()
