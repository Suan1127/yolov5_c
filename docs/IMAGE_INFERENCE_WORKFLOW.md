# 이미지 입력을 사용한 YOLOv5s 추론 및 검증 워크플로우

이 문서는 이미지 파일(bus.jpg, zidane.jpg 등)을 입력으로 사용하여 YOLOv5s 모델을 실행하고 Python과 C 구현을 비교하는 전체 절차를 설명합니다.

## 목차

1. [전체 워크플로우 개요](#전체-워크플로우-개요)
2. [단계별 상세 절차](#단계별-상세-절차)
3. [디렉토리 구조](#디렉토리-구조)
4. [예제: bus 이미지로 전체 과정 수행](#예제-bus-이미지로-전체-과정-수행)

---

## 전체 워크플로우 개요

```
[이미지 파일] 
    ↓
[1. 전처리] → [텐서 파일 (.bin)]
    ↓
[2. PyTorch Golden 생성] → [testdata/python/]
    ↓
[3. C 프로그램 실행] → [testdata/c/]
    ↓
[4. 비교] → [검증 결과]
```

---

## 단계별 상세 절차

### Step 1: 이미지 전처리

**목적:** 이미지 파일을 YOLOv5s 입력 형식(NCHW 텐서)으로 변환

**스크립트:** `tools/preprocess.py`

**사용법:**

```bash
# 단일 이미지 처리
python tools/preprocess.py --image bus.jpg

# 또는 이미지 이름만 (확장자 자동 인식)
python tools/preprocess.py --image bus

# 커스텀 이미지 크기 (기본값: 640)
python tools/preprocess.py --image bus --size 640

# 디렉토리 내 모든 이미지 처리
python tools/preprocess.py
```

**입력:**
- `data/images/bus.jpg` (또는 다른 이미지 파일)

**출력:**
- `data/inputs/bus.bin`: 전처리된 텐서 파일 (NCHW 형식)
- `data/inputs/bus_meta.txt`: 메타데이터 (원본 크기, 비율 등)

**전처리 과정:**
1. 이미지 로드 (BGR 형식)
2. BGR → RGB 변환
3. Letterbox 리사이즈 (비율 유지, 640×640으로 패딩)
4. 정규화: [0, 255] → [0.0, 1.0]
5. NCHW 변환: (H, W, C) → (1, 3, H, W)
6. 바이너리 파일로 저장

---

### Step 2: PyTorch Golden 데이터 생성

**목적:** PyTorch 모델로 추론하여 모든 레이어의 출력을 저장 (golden reference)

**스크립트:** `tools/dump_golden.py`

**사용법:**

```bash
# 이미지 이름으로 입력 (자동으로 data/inputs/{image_name}.bin 찾음)
python tools/dump_golden.py weights/yolov5s.pt bus --output testdata/python

# 또는 직접 텐서 파일 경로 지정
python tools/dump_golden.py weights/yolov5s.pt data/inputs/bus.bin --output testdata/python

# 특정 레이어만 저장 (기본값: 모든 레이어 0-23)
python tools/dump_golden.py weights/yolov5s.pt bus --output testdata/python --layers 0 1 2 3 4 5 6 7 8 9
```

**입력:**
- `weights/yolov5s.pt`: PyTorch 모델 파일
- `bus`: 이미지 이름 (또는 `data/inputs/bus.bin` 직접 경로)

**출력:**
- `testdata/python/bus.bin`: 입력 텐서 (복사본)
- `testdata/python/input.bin`: 입력 텐서 (호환성)
- `testdata/python/layer_000.bin` ~ `testdata/python/layer_023.bin`: 각 레이어 출력
- `testdata/python/output_0.bin`: Detect head 출력 (선택적)
- `testdata/python/golden_meta.json`: 메타데이터

**참고:**
- `dump_golden.py`는 입력 경로가 파일이 아니면 이미지 이름으로 인식하여 `data/inputs/{image_name}.bin`을 자동으로 찾습니다.

---

### Step 3: C 프로그램 실행

**목적:** C 구현으로 추론하여 모든 레이어의 출력을 저장

**사용법:**

```bash
# 빌드 (처음 한 번만)
cd build/Release
cmake --build . --config Release

# 추론 실행 (이미지 이름만 지정)
yolov5s_infer.exe bus

# 또는 가중치 경로 직접 지정
yolov5s_infer.exe bus weights/weights.bin weights/model_meta.json
```

**입력:**
- `bus`: 이미지 이름 (프로그램이 자동으로 `data/inputs/bus.bin` 찾음)
- `weights/weights.bin`: 가중치 파일 (기본 경로)
- `weights/model_meta.json`: 모델 메타데이터 (기본 경로)

**출력 디렉토리 설정:**

C 프로그램은 자동으로 `testdata/c` 디렉토리를 찾아서 설정합니다. `src/main.c`에서:

```c
// 자동으로 testdata/c 디렉토리 찾기 및 설정
// 여러 경로 시도: testdata/c, ../testdata/c, ../../testdata/c
yolov5s_set_output_dir(model, output_dir);
```

따라서 별도 설정 없이 바로 실행 가능합니다.

**출력:**
- `testdata/c/bus.bin`: 입력 텐서 (복사본)
- `testdata/c/input.bin`: 입력 텐서 (호환성)
- `testdata/c/layer_000.bin` ~ `testdata/c/layer_023.bin`: 각 레이어 출력
- `testdata/c/output_p3.bin`, `output_p4.bin`, `output_p5.bin`: P3, P4, P5 feature map

**참고:**
- C 프로그램은 실행 위치에 따라 경로를 자동으로 조정합니다 (`build/Release/`에서 실행 시 `../../` 추가)

---

### Step 4: 결과 비교

**목적:** PyTorch와 C 구현의 출력을 비교하여 정확성 검증

**스크립트:** `tools/compare_tensors.py`

**사용법:**

```bash
# 전체 레이어 비교
python tools/compare_tensors.py testdata/python testdata/c

# 특정 레이어만 비교
python tools/compare_tensors.py testdata/python/layer_002.bin testdata/c/layer_002.bin
```

**출력:**
- 각 레이어별 비교 결과
- Max diff, Mean diff, RMSE
- 첫 번째 실패한 레이어 표시
- 자동으로 SKIP되는 파일:
  - Upsample 레이어 (11, 15): 가중치 없음
  - Output 파일들: 선택적
  - 이미지 파일 (bus.bin, zidane.bin 등): 입력 파일

---

## 디렉토리 구조

```
프로젝트 루트/
├── data/
│   ├── images/          # 원본 이미지 파일
│   │   ├── bus.jpg
│   │   └── zidane.jpg
│   └── inputs/          # 전처리된 텐서 파일
│       ├── bus.bin
│       ├── bus_meta.txt
│       ├── zidane.bin
│       └── zidane_meta.txt
│
├── testdata/
│   ├── python/          # PyTorch golden 출력
│   │   ├── bus.bin (또는 input.bin)
│   │   ├── layer_000.bin
│   │   ├── layer_001.bin
│   │   ├── ...
│   │   ├── layer_023.bin
│   │   └── output_0.bin
│   │
│   └── c/               # C 구현 출력
│       ├── bus.bin (또는 input.bin)
│       ├── layer_000.bin
│       ├── layer_001.bin
│       ├── ...
│       ├── layer_023.bin
│       ├── output_p3.bin
│       ├── output_p4.bin
│       └── output_p5.bin
│
├── debug/
│   ├── pytorch/         # PyTorch 중간 디버그 출력
│   └── c/               # C 중간 디버그 출력
│
└── weights/
    ├── yolov5s.pt       # PyTorch 모델
    ├── weights.bin      # C용 가중치
    ├── weights_map.json # 가중치 매핑
    └── model_meta.json  # 모델 메타데이터
```

---

## 예제: bus 이미지로 전체 과정 수행

### 1. 이미지 준비

```bash
# 이미지가 data/images/ 디렉토리에 있는지 확인
ls data/images/bus.jpg
```

### 2. 이미지 전처리

```bash
python tools/preprocess.py --image bus.jpg
```

**출력:**
```
Processing: bus.jpg
  ✓ Saved tensor to data/inputs/bus.bin
    Tensor shape: (1, 3, 640, 640)
  ✓ Saved metadata to data/inputs/bus_meta.txt
```

### 3. PyTorch Golden 데이터 생성

```bash
python tools/dump_golden.py weights/yolov5s.pt bus --output testdata/python
```

**출력:**
```
Loading input tensor from data/inputs/bus.bin...
Loading model from weights/yolov5s.pt...
Running forward pass...
Saving layer outputs to testdata/python...
  Layer 0: shape (1, 32, 320, 320) -> layer_000.bin
  Layer 1: shape (1, 64, 160, 160) -> layer_001.bin
  ...
  Layer 23: shape (1, 512, 20, 20) -> layer_023.bin
```

### 4. C 프로그램 실행

**먼저 출력 디렉토리 설정 확인:**

`src/main.c`에서 출력 디렉토리가 설정되어 있는지 확인:

```c
// src/main.c의 main 함수에서
yolov5s_set_output_dir(model, "testdata/c");
```

**실행:**

```bash
cd build/Release
yolov5s_infer.exe bus
```

**출력:**
```
=== YOLOv5 C Inference ===

Image name: bus
Loading input tensor...
Found input tensor: ../../data/inputs/bus.bin
Input shape: (1, 3, 640, 640)
...
  Saved layer 0 to testdata/c/layer_000.bin
  Saved layer 1 to testdata/c/layer_001.bin
  ...
  Saved layer 23 to testdata/c/layer_023.bin
```

### 5. 결과 비교

```bash
python tools/compare_tensors.py testdata/python testdata/c
```

**출력:**
```
Found 27 files in golden directory
Found 27 files in test directory
Golden directory: testdata\python
Test directory: testdata\c
Tolerance: 0.0001

Comparing bus.bin...
SKIP bus.bin: Input image file (optional)

Comparing input.bin...
  Shape: [  1   3 640 640]
  Max diff: 0.000000e+00
  Within tolerance (0.0001): OK

Comparing layer_000.bin...
  Shape: [  1  32 320 320]
  Max diff: 2.157688e-05
  Within tolerance (0.0001): OK

...

Summary:
  Compared: 24 files
  Passed: 24/24
  Skipped: 3 files (Upsample layers, output files)

[OK] All comparisons passed!
```

---

## 주의사항

### 1. 출력 디렉토리 설정

C 프로그램에서 출력을 저장하려면 `yolov5s_set_output_dir()`을 호출해야 합니다.

**현재 구현 확인:**
- `src/main.c`에서 출력 디렉토리가 설정되어 있는지 확인
- 설정되지 않았다면 추가 필요

### 2. 경로 문제

**C 프로그램 실행 위치:**
- `build/Release/`에서 실행 시 상대 경로가 달라짐
- 프로그램이 자동으로 `../../` 경로를 시도하지만, 필요시 수정

**해결 방법:**
```c
// src/main.c에서 절대 경로 또는 프로젝트 루트 기준 경로 사용
yolov5s_set_output_dir(model, "../../testdata/c");
```

### 3. 이미지 이름 vs 파일 경로

**PyTorch (`dump_golden.py`):**
- 이미지 이름(`bus`) 또는 파일 경로(`data/inputs/bus.bin`) 모두 지원
- 이미지 이름이면 자동으로 `data/inputs/{name}.bin` 찾음

**C 프로그램 (`yolov5s_infer.exe`):**
- 이미지 이름만 지원 (확장자 제외)
- 자동으로 `data/inputs/{name}.bin` 찾음

### 4. 입력 파일 형식

**텐서 파일 형식:**
- 헤더: 4개 int32 (n, c, h, w)
- 데이터: n×c×h×w 개 float32

**예시:**
```
[4 bytes: n=1]
[4 bytes: c=3]
[4 bytes: h=640]
[4 bytes: w=640]
[1×3×640×640×4 bytes: float32 데이터]
```

---

## 빠른 참조

### 전체 워크플로우 (한 번에)

```bash
# 1. 이미지 전처리
python tools/preprocess.py --image bus.jpg

# 2. PyTorch golden 생성
python tools/dump_golden.py weights/yolov5s.pt bus --output testdata/python

# 3. C 프로그램 실행
cd build/Release
yolov5s_infer.exe bus

# 4. 비교
cd ../..
python tools/compare_tensors.py testdata/python testdata/c
```

### 새로운 이미지로 테스트

```bash
# 1. 이미지 파일을 data/images/에 복사
cp new_image.jpg data/images/

# 2. 전처리
python tools/preprocess.py --image new_image.jpg

# 3. PyTorch golden 생성
python tools/dump_golden.py weights/yolov5s.pt new_image --output testdata/python

# 4. C 프로그램 실행
cd build/Release
yolov5s_infer.exe new_image

# 5. 비교
cd ../..
python tools/compare_tensors.py testdata/python testdata/c
```

---

## 문제 해결

### 문제: C 프로그램이 입력 파일을 찾지 못함

**증상:**
```
Error: Cannot find input tensor file
Tried:
  - data/inputs/bus.bin
  - ../data/inputs/bus.bin
  - ../../data/inputs/bus.bin
```

**해결:**
1. `data/inputs/bus.bin` 파일이 존재하는지 확인
2. C 프로그램을 프로젝트 루트에서 실행하거나 경로 수정
3. 절대 경로 사용

### 문제: 출력 파일이 저장되지 않음

**증상:**
- C 프로그램 실행 후 `testdata/c/`에 파일이 없음

**해결:**
1. `src/main.c`에서 `yolov5s_set_output_dir()` 호출 확인
2. 출력 디렉토리 경로가 올바른지 확인
3. 디렉토리 생성 권한 확인

### 문제: 비교 시 파일이 누락됨

**증상:**
```
MISSING layer_013.bin: Not found in test directory
```

**해결:**
1. C 프로그램이 모든 레이어를 저장하도록 확인
2. `src/models/yolov5s_infer.c`에서 모든 레이어에 `save_feature()` 호출 확인
3. 출력 디렉토리 경로 확인

---

## 참고

- **전처리 상세:** `docs/PREPROCESSING.md`
- **디버깅 방법:** `docs/DEBUGGING_PROCESS.md`
- **모델 구조:** `docs/MODULE_ARCHITECTURE.md`
