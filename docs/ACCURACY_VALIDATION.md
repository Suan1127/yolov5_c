# 정확도 검증 가이드

이 문서는 Python PyTorch 모델의 golden reference와 C 구현 결과를 비교하여 정확도를 검증하는 방법을 설명합니다.

## 개요

정확도 검증 파이프라인은 다음 단계로 구성됩니다:

1. **Python Golden 생성**: PyTorch 모델로부터 참조 텐서 덤프
2. **C 추론 실행**: C 구현으로 동일한 입력에 대해 추론 수행
3. **결과 비교**: 두 결과를 비교하여 오차 계산

## 디렉토리 구조

```
testdata/
├── python/          # Python golden reference 출력
│   ├── input.bin
│   ├── layer_000.bin
│   ├── layer_001.bin
│   ├── layer_002.bin
│   ├── layer_003.bin
│   ├── ...
│   ├── output_0.bin
│   └── output_1.bin
└── c/               # C 구현 출력
    ├── input.bin
    ├── layer_000.bin
    ├── layer_001.bin
    ├── layer_002.bin
    ├── layer_003.bin
    ├── ...
    └── output_p3.bin
```

## 단계별 실행 방법

### 1. Python Golden Reference 생성

PyTorch 모델을 사용하여 참조 텐서를 생성합니다.

```bash
python tools/dump_golden.py path/to/yolov5s.pt bus --output testdata/python
```

**파라미터 설명:**
- `path/to/yolov5s.pt`: PyTorch 모델 파일 경로
- `bus`: 입력 이미지 이름 (확장자 제외)
- `--output testdata/python`: 출력 디렉토리

**생성되는 파일:**
- `input.bin`: 전처리된 입력 텐서
- `layer_XXX.bin`: 각 레이어의 출력 텐서 (0, 1, 2, 3, 4, 5, 6, 7, 9, 17, 20, 23)
- `output_0.bin`, `output_1.bin`, `output_2.bin`: Detect head의 최종 출력 (P3, P4, P5)

**참고:**
- 입력 이미지는 `data/images/bus.jpg`에서 자동으로 찾습니다.
- 전처리된 입력 텐서는 `data/inputs/bus.bin`에서도 찾습니다.

### 2. C 추론 실행

C 구현으로 동일한 입력에 대해 추론을 수행합니다.

```bash
# 빌드 (Windows)
cd build
cmake ..
cmake --build . --config Release

# 추론 실행
build\Release\yolov5_infer.exe bus
```

**동작:**
- 입력 텐서를 `data/inputs/bus.bin`에서 로드
- 모델 가중치를 `weights/weights.bin`에서 로드
- 중간 레이어 출력을 `testdata/c/`에 저장
- 최종 detection 결과를 `data/outputs/bus_detections.txt`에 저장

**저장되는 파일:**
- `testdata/c/input.bin`: 입력 텐서 (Python과 동일한 형식)
- `testdata/c/layer_XXX.bin`: 각 레이어의 출력 텐서
- `data/outputs/bus_detections.txt`: 최종 detection 결과 (bbox, confidence, class)

### 3. 결과 비교

Python golden과 C 결과를 비교합니다.

```bash
python tools/compare_tensors.py testdata/python testdata/c --tolerance 1e-4
```

**파라미터 설명:**
- `testdata/python`: Python golden 디렉토리
- `testdata/c`: C 구현 출력 디렉토리
- `--tolerance 1e-4`: 허용 오차 (기본값: 0.0001)

**출력 예시:**
```
Comparing 12 files...
Golden directory: testdata\python
Test directory: testdata\c
Tolerance: 0.0001

Comparing input.bin...
  Shape: [  1   3 640 640]
  Max diff: 0.000000e+00
  Mean diff: 0.000000e+00
  RMSE: 0.000000e+00
  Within tolerance (0.0001): ✓

Comparing layer_000.bin...
  Shape: [  1  32 640 640]
  Max diff: 1.234567e-05
  Mean diff: 2.345678e-06
  RMSE: 3.456789e-06
  Within tolerance (0.0001): ✓

Comparing layer_001.bin...
  Shape: [  1  64 320 320]
  Max diff: 5.678901e+00
  Mean diff: 1.234567e-01
  RMSE: 2.345678e-01
  Within tolerance (0.0001): ✗
  Locations with diff > 0.0001: 819200

⚠ First mismatch detected at: layer_001.bin
  This is likely where the error originates.

...
```

## 결과 해석

### 비교 지표

- **Max diff**: 최대 절대 오차
- **Mean diff**: 평균 절대 오차
- **RMSE**: Root Mean Square Error
- **Within tolerance**: 허용 오차 내 일치 여부

### 파일 비교 순서

비교 스크립트는 다음 순서로 파일을 비교합니다:

1. `input.bin` (우선순위 0)
2. `layer_XXX.bin` (레이어 번호 순, 우선순위 1)
   - `layer_000.bin`, `layer_001.bin`, `layer_002.bin`, ...
3. `output_*.bin` (우선순위 3)

이 순서로 비교하여 **첫 번째 불일치 레이어**를 빠르게 찾을 수 있습니다.

### 문제 진단

**첫 불일치 레이어가 나타나면:**
- 해당 레이어의 구현을 확인하세요
- 가중치 로드가 올바른지 확인하세요
- 이전 레이어의 출력이 정확한지 확인하세요

**예시:**
- `layer_000.bin`이 일치하고 `layer_001.bin`이 불일치하면 → Layer 1 (Conv 32->64) 구현에 문제
- `layer_001.bin`이 일치하고 `layer_002.bin`이 불일치하면 → Layer 2 (C3) 구현에 문제

## 저장되는 레이어 목록

다음 레이어들의 출력이 저장됩니다:

- **Layer 0**: Conv(3->32, 6x6, s=2) + BN + SiLU
- **Layer 1**: Conv(32->64, 3x3, s=2) + BN + SiLU
- **Layer 2**: C3(64->64, n=1)
- **Layer 3**: Conv(64->128, 3x3, s=2) + BN + SiLU
- **Layer 4**: C3(128->128, n=2)
- **Layer 5**: Conv(128->256, 3x3, s=2) + BN + SiLU
- **Layer 6**: C3(256->256, n=3)
- **Layer 7**: Conv(256->512, 3x3, s=2) + BN + SiLU
- **Layer 9**: C3(512->512, n=1)
- **Layer 17**: Head C3 (P3 경로)
- **Layer 20**: Head C3 (P4 경로)
- **Layer 23**: Head C3 (P5 경로)

## 주의사항

1. **입력 이미지 일치**: Python과 C 모두 동일한 입력 이미지를 사용해야 합니다.
2. **전처리 일치**: `tools/preprocess.py`로 생성한 입력 텐서를 사용하세요.
3. **모델 버전 일치**: Python과 C가 동일한 모델 구조를 사용하는지 확인하세요.
4. **부동소수점 오차**: 완전한 일치는 어렵습니다. 허용 오차를 적절히 설정하세요.

## 자동화 스크립트

전체 검증 파이프라인을 자동으로 실행하려면:

```bash
python tools/validate.py bus --model path/to/yolov5s.pt
```

이 스크립트는 다음을 자동으로 수행합니다:
1. Python golden 생성
2. C 추론 실행
3. 결과 비교
4. 리포트 생성

## 문제 해결

### 파일을 찾을 수 없음

- `testdata/python/` 또는 `testdata/c/` 디렉토리가 존재하는지 확인
- Python golden이 제대로 생성되었는지 확인
- C 추론이 완료되었는지 확인

### 비교 결과가 모두 실패

- 입력 텐서가 동일한지 확인 (`input.bin` 비교)
- 모델 가중치가 올바르게 로드되었는지 확인
- 레이어 구현이 올바른지 확인

### 특정 레이어만 실패

- 해당 레이어의 구현 코드를 확인
- 가중치 매핑이 올바른지 확인 (`weights/weights_map.json`)
- 이전 레이어의 출력이 정확한지 확인
