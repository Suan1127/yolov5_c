#!/usr/bin/env python3
"""
Layer 9 (SPPF 블록) 디버깅 스크립트
SPPF 블록의 각 단계별 출력을 확인하여 어디서 문제가 발생하는지 찾습니다.
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
    layer8_path = "testdata/python/layer_008.bin"
    
    print("=== Layer 9 (SPPF) 디버깅 ===\n")
    
    # Load Layer 8 output (SPPF input)
    print(f"1. Loading Layer 8 output from {layer8_path}...")
    layer8_output = load_tensor_bin(layer8_path)
    print(f"   Shape: {layer8_output.shape}")
    print(f"   Range: [{layer8_output.min():.6f}, {layer8_output.max():.6f}]")
    print(f"   Sample [0,0,0,0]: {layer8_output[0,0,0,0]:.6f}\n")
    
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
    
    # Find SPPF module (Layer 9)
    if hasattr(model, 'model'):
        model_obj = model.model
    else:
        model_obj = model
    
    sppf_module = None
    
    # Try to directly access model[9]
    if hasattr(model_obj, '__getitem__'):
        try:
            sppf_module = model_obj[9]
            print(f"   Found SPPF module via direct access: model_obj[9]")
        except (IndexError, KeyError, TypeError):
            pass
    
    # If direct access failed, search by module type
    if sppf_module is None:
        layer_idx = 0
        for name, module in model_obj.named_modules():
            # Check if this is an SPPF module
            if 'SPPF' in str(type(module)) or (hasattr(module, 'cv1') and hasattr(module, 'cv2') and hasattr(module, 'm')):
                # Try to match by layer index from name or position
                if '.' in name:
                    parts = name.split('.')
                    for part in parts:
                        if part.isdigit():
                            idx = int(part)
                            if idx == 9:
                                sppf_module = module
                                print(f"   Found SPPF module via named_modules: {name}")
                                break
                    if sppf_module:
                        break
                # Fallback: count sequential modules
                elif layer_idx == 9:
                    sppf_module = module
                    print(f"   Found SPPF module via counting: layer_idx={layer_idx}")
                    break
                layer_idx += 1
    
    if sppf_module is None:
        print("Error: Could not find SPPF module")
        # Try to find it by iterating through model
        print("Available modules:")
        for i, (name, module) in enumerate(model_obj.named_modules()):
            if 'SPPF' in str(type(module)) or (hasattr(module, 'cv1') and hasattr(module, 'cv2') and hasattr(module, 'm')):
                print(f"  {i}: {name} - {type(module)}")
        sys.exit(1)
    
    print(f"   Found SPPF module: {type(sppf_module)}\n")
    
    # Convert to torch tensor
    input_torch = torch.from_numpy(layer8_output.copy())
    
    # Run SPPF step by step
    print("3. Running SPPF step by step...\n")
    
    with torch.no_grad():
        # Step 1: cv1
        print("Step 1: cv1 (Conv + BN + SiLU)")
        cv1_out = sppf_module.cv1(input_torch)
        print(f"   Output shape: {cv1_out.shape}")
        print(f"   Output range: [{cv1_out.min():.6f}, {cv1_out.max():.6f}]")
        save_tensor_bin(cv1_out, "debug/pytorch/sppf_cv1_output.bin")
        print(f"   Saved to: debug/pytorch/sppf_cv1_output.bin\n")
        
        # Step 2: MaxPool operations
        print("Step 2: MaxPool operations")
        x = cv1_out
        y1 = sppf_module.m(x)
        print(f"   y1 = m(x): shape={y1.shape}, range=[{y1.min():.6f}, {y1.max():.6f}]")
        save_tensor_bin(y1, "debug/pytorch/sppf_y1_output.bin")
        print(f"   Saved to: debug/pytorch/sppf_y1_output.bin")
        
        y2 = sppf_module.m(y1)
        print(f"   y2 = m(y1): shape={y2.shape}, range=[{y2.min():.6f}, {y2.max():.6f}]")
        save_tensor_bin(y2, "debug/pytorch/sppf_y2_output.bin")
        print(f"   Saved to: debug/pytorch/sppf_y2_output.bin")
        
        y4 = sppf_module.m(y2)
        print(f"   y4 = m(y2): shape={y4.shape}, range=[{y4.min():.6f}, {y4.max():.6f}]")
        save_tensor_bin(y4, "debug/pytorch/sppf_y4_output.bin")
        print(f"   Saved to: debug/pytorch/sppf_y4_output.bin\n")
        
        # Step 3: Concat
        print("Step 3: Concat [x, y1, y2, y4]")
        concat_out = torch.cat((x, y1, y2, y4), 1)
        print(f"   Output shape: {concat_out.shape}")
        print(f"   Output range: [{concat_out.min():.6f}, {concat_out.max():.6f}]")
        save_tensor_bin(concat_out, "debug/pytorch/sppf_concat_output.bin")
        print(f"   Saved to: debug/pytorch/sppf_concat_output.bin\n")
        
        # Step 4: cv2
        print("Step 4: cv2 (Conv + BN + SiLU)")
        cv2_out = sppf_module.cv2(concat_out)
        print(f"   Output shape: {cv2_out.shape}")
        print(f"   Output range: [{cv2_out.min():.6f}, {cv2_out.max():.6f}]")
        save_tensor_bin(cv2_out, "debug/pytorch/sppf_cv2_output.bin")
        print(f"   Saved to: debug/pytorch/sppf_cv2_output.bin\n")
        
        # Final output
        print("Final SPPF output:")
        final_out = sppf_module(input_torch)
        print(f"   Shape: {final_out.shape}")
        print(f"   Range: [{final_out.min():.6f}, {final_out.max():.6f}]")
        print(f"   Sample [0,0,0,0]: {final_out[0,0,0,0]:.6f}")
        
        # Verify final output matches cv2_out
        diff = (final_out - cv2_out).abs().max()
        print(f"   Difference from cv2_out: {diff:.6e}")
        if diff > 1e-5:
            print("   WARNING: Final output does not match cv2_out!")
        else:
            print("   OK: Final output matches cv2_out")
    
    print("\n=== 디버깅 완료 ===")
    print("PyTorch 중간 출력이 debug/pytorch/ 디렉토리에 저장되었습니다.")

if __name__ == '__main__':
    main()
