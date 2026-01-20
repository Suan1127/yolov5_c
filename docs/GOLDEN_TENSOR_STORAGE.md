# Golden 텐서 저장 방식

이 문서는 `testdata/python/` 디렉토리에 저장되는 PyTorch golden reference 텐서의 저장 방식을 설명합니다.

## 개요

`tools/dump_golden.py` 스크립트를 사용하여 PyTorch YOLOv5s 모델의 중간 레이어 출력을 golden reference로 저장합니다.

## 저장 방식

### 1. 파일 형식

각 텐서는 바이너리 파일로 저장되며, C 구현과 호환되는 형식을 사용합니다:

```python
def save_tensor_bin(tensor, filepath):
    """Save tensor to binary file (C format)"""
    with open(filepath, 'wb') as f:
        shape = np.array(tensor.shape, dtype=np.int32)  # [N, C, H, W]
        shape.tofile(f)                                 # 4개 int32 (16 bytes)
        tensor.astype(np.float32).tofile(f)             # float32 데이터
```

**바이너리 파일 구조:**
- **헤더**: 16 bytes (4개 int32: N, C, H, W)
- **데이터**: `N × C × H × W × 4` bytes (float32 배열)

### 2. 저장되는 레이어

기본적으로 다음 레이어들의 출력이 저장됩니다:

```python
save_layers = [0, 1, 2, 3, 4, 5, 6, 7, 9, 17, 20, 23]
```

**레이어 설명:**
- **Layer 0**: Conv(3->32, 6×6, s=2) + BN + SiLU
- **Layer 1**: Conv(32->64, 3×3, s=2) + BN + SiLU
- **Layer 2**: C3(64->64, n=1)
- **Layer 3**: Conv(64->128, 3×3, s=2) + BN + SiLU
- **Layer 4**: C3(128->128, n=2)
- **Layer 5**: Conv(128->256, 3×3, s=2) + BN + SiLU
- **Layer 6**: C3(256->256, n=3)
- **Layer 7**: Conv(256->512, 3×3, s=2) + BN + SiLU
- **Layer 9**: C3(512->512, n=1)
- **Layer 17**: Head C3 (P3 경로)
- **Layer 20**: Head C3 (P4 경로)
- **Layer 23**: Head C3 (P5 경로)

### 3. 파일 명명 규칙

```
layer_{layer_idx:03d}.bin
```

**예시:**
- `layer_000.bin`: Layer 0 출력
- `layer_001.bin`: Layer 1 출력
- `layer_002.bin`: Layer 2 출력
- `layer_023.bin`: Layer 23 출력

### 4. 저장 프로세스

#### Step 1: Forward Hook 등록

PyTorch의 forward hook을 사용하여 각 레이어의 출력을 캡처합니다:

```python
def make_hook(layer_idx):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            outputs[layer_idx] = output.detach().cpu().numpy()
    return hook

# 레이어별 hook 등록
for layer_idx in save_layers:
    module = model.model[layer_idx]
    handle = module.register_forward_hook(make_hook(layer_idx))
    handles.append(handle)
```

**Hook 등록 방법:**
1. **직접 인덱스 접근**: `model.model[layer_idx]`로 직접 접근
2. **named_modules 접근**: `named_modules()`를 사용하여 레이어 이름으로 매칭 (fallback)

#### Step 2: Forward Pass 실행

```python
with torch.no_grad():
    output = model(input_tensor)
```

Forward pass 실행 중 각 레이어의 hook이 자동으로 호출되어 출력을 캡처합니다.

#### Step 3: 텐서 저장

```python
for layer_idx in save_layers:
    if layer_idx in outputs:
        tensor = outputs[layer_idx]
        output_path = output_dir / f"layer_{layer_idx:03d}.bin"
        save_tensor_bin(tensor, output_path)
```

### 5. 저장되는 파일 목록

#### 입력 텐서
- `input.bin`: 입력 텐서 (전처리된 이미지)
- `{image_name}.bin`: 이미지 이름으로도 저장 (예: `bus.bin`)

#### 레이어 출력
- `layer_000.bin` ~ `layer_023.bin`: 각 레이어의 출력 텐서

#### 최종 출력
- `output_0.bin`: Detect head의 최종 출력 (P3, P4, P5 중 첫 번째)
- `output_1.bin`, `output_2.bin`: Detect head의 나머지 출력들 (있는 경우)

#### 메타데이터
- `golden_meta.json`: 저장된 레이어 정보 및 메타데이터

### 6. 메타데이터 파일

`golden_meta.json` 파일에는 다음 정보가 저장됩니다:

```json
{
  "save_layers": [0, 1, 2, 3, 4, 5, 6, 7, 9, 17, 20, 23],
  "input_shape": [1, 3, 640, 640],
  "saved_layers": [0, 1, 2, 3, 4, 5, 6, 7, 9, 17, 20, 23],
  "output_dir": "testdata\\python"
}
```

## 사용 방법

### 기본 사용법

```bash
python tools/dump_golden.py weights/yolov5s.pt bus --output testdata/python
```

**파라미터:**
- `weights/yolov5s.pt`: PyTorch 모델 파일 경로
- `bus`: 입력 이미지 이름 (확장자 제외) 또는 입력 텐서 파일 경로
- `--output testdata/python`: 출력 디렉토리 (기본값: `testdata/python`)

### 커스텀 레이어 저장

특정 레이어만 저장하려면:

```bash
python tools/dump_golden.py weights/yolov5s.pt bus --layers 0 1 2 3
```

### 입력 텐서 경로

입력 텐서는 다음 위치에서 자동으로 검색됩니다:

1. `data/inputs/{image_name}.bin`
2. `testdata/python/{image_name}.bin`
3. `testdata/c/{image_name}.bin`
4. `{output_dir}/{image_name}.bin`

또는 직접 경로를 지정할 수 있습니다:

```bash
python tools/dump_golden.py weights/yolov5s.pt data/inputs/bus.bin
```

## 파일 크기

각 텐서 파일의 대략적인 크기:

- **Layer 0**: `(1, 32, 320, 320)` = 약 13 MB
- **Layer 1**: `(1, 64, 160, 160)` = 약 6.5 MB
- **Layer 2**: `(1, 64, 160, 160)` = 약 6.5 MB
- **Layer 3**: `(1, 128, 80, 80)` = 약 3.3 MB
- **Layer 4**: `(1, 128, 80, 80)` = 약 3.3 MB
- **Layer 5**: `(1, 256, 40, 40)` = 약 1.6 MB
- **Layer 6**: `(1, 256, 40, 40)` = 약 1.6 MB
- **Layer 7**: `(1, 512, 20, 20)` = 약 0.8 MB
- **Layer 9**: `(1, 512, 20, 20)` = 약 0.8 MB
- **Layer 17**: `(1, 128, 80, 80)` = 약 3.3 MB
- **Layer 20**: `(1, 256, 40, 40)` = 약 1.6 MB
- **Layer 23**: `(1, 512, 20, 20)` = 약 0.8 MB

**총 크기**: 약 45-50 MB (모든 레이어 포함)

## 주의사항

1. **입력 텐서 일치**: C 구현과 비교하려면 동일한 입력 텐서를 사용해야 합니다.
2. **모델 버전**: PyTorch 모델과 C 구현이 동일한 모델 구조를 사용하는지 확인하세요.
3. **BN Fusion**: PyTorch 모델이 BN fusion을 사용하는 경우, C 구현도 동일하게 처리해야 합니다.
4. **부동소수점 정밀도**: float32로 저장되므로, 완전한 일치는 어렵습니다. 허용 오차를 적절히 설정하세요.

## C 구현과의 비교

C 구현은 `testdata/c/` 디렉토리에 동일한 형식으로 레이어 출력을 저장합니다:

```bash
# C 추론 실행
./build/yolov5_infer bus

# 비교
python tools/compare_tensors.py testdata/python testdata/c
```

## 관련 문서

- `docs/ACCURACY_VALIDATION.md`: 정확도 검증 가이드
- `docs/DEBUG_FILES.md`: 디버그 파일 가이드
- `tools/compare_tensors.py`: 텐서 비교 스크립트
