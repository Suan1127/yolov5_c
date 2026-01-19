# YOLOv5 C 포팅 모듈 아키텍처

이 문서는 YOLOv5 C 포팅 프로젝트의 각 모듈이 어떻게 구현되고 서로 상호작용하는지 설명합니다.

## 목차

1. [전체 아키텍처 개요](#전체-아키텍처-개요)
2. [Inference 관점에서의 모듈 흐름](#inference-관점에서의-모듈-흐름)
3. [Detection 관점에서의 모듈 흐름](#detection-관점에서의-모듈-흐름)
4. [모듈별 상세 설명](#모듈별-상세-설명)

---

## 전체 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Image (640×640)                    │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  Preprocessing (tools/preprocess.py)                        │
│  - Letterbox resize                                         │
│  - Normalize [0, 255] → [0.0, 1.0]                         │
│  - Convert to NCHW tensor                                   │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  Core: Tensor Management                                    │
│  - tensor_t: NCHW 레이아웃                                  │
│  - Memory allocation/deallocation                           │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  Model Build (yolov5s_build.c)                              │
│  - Load weights from weights.bin                            │
│  - Initialize all layers                                    │
│  - Setup layer connections                                  │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  Forward Pass (yolov5s_infer.c)                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Backbone (Layers 0-9)                                │   │
│  │  - Conv layers                                        │   │
│  │  - C3 blocks                                          │   │
│  │  - SPPF block                                         │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Head (Layers 10-23)                                   │   │
│  │  - FPN: Upsample + Concat                             │   │
│  │  - C3 blocks                                          │   │
│  │  - Output: P3, P4, P5 features                        │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  Detect Head (detect.c)                                      │
│  - 1×1 Conv: Features → Detection outputs                    │
│  - Decode: Grid/Anchor → BBox coordinates                   │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  NMS (nms.c)                                                 │
│  - Remove overlapping detections                            │
│  - Keep highest confidence                                  │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  Final Detections (bbox + class + confidence)                │
└─────────────────────────────────────────────────────────────┘
```

---

## Inference 관점에서의 모듈 흐름

### 1. 입력 준비 단계

**모듈**: `tools/preprocess.py`
- **역할**: 이미지를 모델 입력 형식으로 변환
- **처리 과정**:
  1. 이미지 로드 (BGR)
  2. RGB 변환
  3. Letterbox resize (640×640, 비율 유지)
  4. 정규화 [0, 255] → [0.0, 1.0]
  5. NCHW 변환: (H, W, C) → (1, 3, 640, 640)
- **출력**: `data/inputs/{image_name}.bin`, `{image_name}_meta.txt`

### 2. 모델 빌드 단계

**모듈**: `src/models/yolov5s_build.c`
- **역할**: 가중치 로드 및 레이어 초기화
- **처리 과정**:
  1. `weights_loader_create()`: weights.bin 로드
  2. 각 레이어별 초기화:
     - Backbone Conv 레이어 (0, 1, 3, 5, 7)
     - Backbone C3 블록 (2, 4, 6, 8)
     - SPPF 블록 (9)
     - Head Conv 레이어 (10, 14, 18, 21)
     - Head C3 블록 (13, 17, 20, 23)
  3. 가중치 매핑: `weights_map.json`을 통해 각 레이어의 가중치 위치 찾기
  4. 가중치 로드: 각 레이어에 가중치 복사
- **출력**: 초기화된 `yolov5s_model_t` 구조체

**관련 모듈**:
- `src/core/weights_loader.c`: 바이너리 파일에서 가중치 읽기
- `src/core/tensor.c`: 텐서 메모리 관리

### 3. Forward Pass 단계

**모듈**: `src/models/yolov5s_infer.c`
- **역할**: 입력 텐서를 모델에 통과시켜 특징맵 생성

#### 3.1 Backbone (Layers 0-9)

```
Input (1, 3, 640, 640)
  ↓
Layer 0: Conv(3→32, 6×6, s=2) + BN + SiLU
  → (1, 32, 320, 320)
  ↓
Layer 1: Conv(32→64, 3×3, s=2) + BN + SiLU
  → (1, 64, 160, 160)
  ↓
Layer 2: C3(64→64, n=1)
  → (1, 64, 160, 160)
  ↓
Layer 3: Conv(64→128, 3×3, s=2) + BN + SiLU → SAVE[3]
  → (1, 128, 80, 80)
  ↓
Layer 4: C3(128→128, n=2) → SAVE[4]
  → (1, 128, 80, 80)
  ↓
Layer 5: Conv(128→256, 3×3, s=2) + BN + SiLU → SAVE[5]
  → (1, 256, 40, 40)
  ↓
Layer 6: C3(256→256, n=3) → SAVE[6]
  → (1, 256, 40, 40)
  ↓
Layer 7: Conv(256→512, 3×3, s=2) + BN + SiLU → SAVE[7]
  → (1, 512, 20, 20)
  ↓
Layer 8: C3(512→512, n=1)
  → (1, 512, 20, 20)
  ↓
Layer 9: SPPF(512→512, k=5) → SAVE[9]
  → (1, 512, 20, 20)
```

**사용 모듈**:
- `src/ops/conv2d.c`: Convolution 연산
- `src/ops/batchnorm2d.c`: Batch normalization
- `src/ops/activation.c`: SiLU 활성화
- `src/blocks/c3.c`: C3 블록 (Bottleneck + Concat)
- `src/blocks/sppf.c`: SPPF 블록 (MaxPool 반복)

#### 3.2 Head (Layers 10-23)

**FPN Top-down 경로**:
```
Layer 9 output (1, 512, 20, 20)
  ↓
Layer 10: Conv(512→256, 1×1) + BN + SiLU
  → (1, 256, 20, 20) → SAVE for later concat
  ↓
Layer 11: Upsample(×2, nearest)
  → (1, 256, 40, 40)
  ↓
Layer 12: Concat([Layer 11, Layer 6])
  → (1, 512, 40, 40)  [256 from 11 + 256 from 6]
  ↓
Layer 13: C3(512→256, n=1, shortcut=False)
  → (1, 256, 40, 40) → SAVE for later concat
  ↓
Layer 14: Conv(256→128, 1×1) + BN + SiLU
  → (1, 128, 40, 40)
  ↓
Layer 15: Upsample(×2, nearest)
  → (1, 128, 80, 80)
  ↓
Layer 16: Concat([Layer 15, Layer 4])
  → (1, 256, 80, 80)  [128 from 15 + 128 from 4]
  ↓
Layer 17: C3(256→128, n=1, shortcut=False) → SAVE[17] (P3)
  → (1, 128, 80, 80)
```

**Bottom-up 경로**:
```
Layer 17 (P3) (1, 128, 80, 80)
  ↓
Layer 18: Conv(128→128, 3×3, s=2) + BN + SiLU
  → (1, 128, 40, 40)
  ↓
Layer 19: Concat([Layer 18, Layer 13])
  → (1, 256, 40, 40)  [128 from 18 + 256 from 13]
  ↓
Layer 20: C3(256→256, n=1, shortcut=False) → SAVE[20] (P4)
  → (1, 256, 40, 40)
  ↓
Layer 21: Conv(256→256, 3×3, s=2) + BN + SiLU
  → (1, 256, 20, 20)
  ↓
Layer 22: Concat([Layer 21, Layer 10])
  → (1, 512, 20, 20)  [256 from 21 + 256 from 10]
  ↓
Layer 23: C3(512→512, n=1, shortcut=False) → SAVE[23] (P5)
  → (1, 512, 20, 20)
```

**사용 모듈**:
- `src/ops/upsample.c`: Nearest neighbor upsampling
- `src/ops/concat.c`: Channel-wise concatenation

### 4. 메모리 관리

**Ping-pong 버퍼 전략**:
- `buf_a`, `buf_b`: 레이어 간 데이터 전달
- `saved_features[]`: FPN 연결을 위한 중간 특징맵 저장
- Workspace 버퍼: C3, SPPF 블록 내부 연산용

---

## Detection 관점에서의 모듈 흐름

### 1. Detect Head Forward

**모듈**: `src/postprocess/detect.c` - `detect_forward()`
- **입력**: P3, P4, P5 특징맵
  - P3: (1, 128, 80, 80) - 작은 객체용
  - P4: (1, 256, 40, 40) - 중간 객체용
  - P5: (1, 512, 20, 20) - 큰 객체용

- **처리**:
  1. 각 스케일별 1×1 Conv 적용
     - P3: Conv(128 → 255) → (1, 255, 80, 80)
     - P4: Conv(256 → 255) → (1, 255, 40, 40)
     - P5: Conv(512 → 255) → (1, 255, 20, 20)
  2. Reshape: (1, 255, H, W) → (1, 3, H, W, 85)
     - 255 = 3 anchors × 85
     - 85 = 4 (bbox) + 1 (obj_conf) + 80 (class_conf)

- **출력**: `detect_output_t`
  - `p3_output`: (1, 3, 80, 80, 85)
  - `p4_output`: (1, 3, 40, 40, 85)
  - `p5_output`: (1, 3, 20, 20, 85)

### 2. Decode 단계

**모듈**: `src/postprocess/detect.c` - `detect_decode()`
- **역할**: Raw detection outputs를 실제 bbox 좌표로 변환

**처리 과정**:
```
For each scale (P3, P4, P5):
  For each grid cell (y, x):
    For each anchor (0, 1, 2):
      // Get raw outputs
      tx, ty, tw, th = raw_outputs[anchor, y, x, 0:4]
      obj_conf = sigmoid(raw_outputs[anchor, y, x, 4])
      cls_conf[0:79] = sigmoid(raw_outputs[anchor, y, x, 5:84])
      
      // Decode bbox
      center_x = (x + sigmoid(tx)) * stride / input_size
      center_y = (y + sigmoid(ty)) * stride / input_size
      width = exp(tw) * anchor_w / input_size
      height = exp(th) * anchor_h / input_size
      
      // Calculate final confidence
      final_conf = obj_conf * max(cls_conf)
      
      // Filter by confidence threshold
      if final_conf > conf_threshold:
        Add to detections list
```

**Anchor 정보**:
- P3 (stride=8): anchors = [10,13, 16,30, 33,23]
- P4 (stride=16): anchors = [30,61, 62,45, 59,119]
- P5 (stride=32): anchors = [116,90, 156,198, 373,326]

**출력**: `detection_t` 배열
- 각 detection: `{x, y, w, h, conf, cls_id, cls_conf[80]}`

### 3. NMS 단계

**모듈**: `src/postprocess/nms.c` - `nms()`
- **역할**: 중복 detection 제거

**처리 과정**:
1. Confidence 기준 내림차순 정렬
2. Greedy NMS:
   ```
   For each detection (sorted by confidence):
     If already suppressed: skip
     Mark as kept
     For each remaining detection:
       If same class and IoU > threshold:
         Suppress (mark as removed)
   ```

**IoU 계산** (`calculate_iou()`):
```
box1: (x1, y1, w1, h1) → (x1_min, y1_min, x1_max, y1_max)
box2: (x2, y2, w2, h2) → (x2_min, y2_min, x2_max, y2_max)

intersection = max(0, min(x1_max, x2_max) - max(x1_min, x2_min)) ×
               max(0, min(y1_max, y2_max) - max(y1_min, y2_min))

union = area1 + area2 - intersection
IoU = intersection / union
```

**출력**: 최종 detection 리스트 (중복 제거됨)

---

## 모듈별 상세 설명

### Core 모듈

#### `src/core/tensor.h/c`
- **역할**: 다차원 배열 (텐서) 관리
- **구조**: NCHW 레이아웃 (Batch, Channels, Height, Width)
- **주요 함수**:
  - `tensor_create()`: 텐서 할당
  - `tensor_free()`: 메모리 해제
  - `tensor_copy()`: 텐서 복사
  - `tensor_dump()`/`tensor_load()`: 바이너리 파일 I/O
- **Inference에서의 역할**: 모든 레이어의 입력/출력 데이터 저장
- **Detection에서의 역할**: P3, P4, P5 특징맵 저장 및 전달

#### `src/core/memory.h/c`
- **역할**: Arena allocator (효율적인 메모리 관리)
- **특징**: 한 번에 할당, 한 번에 해제
- **사용처**: 대량의 임시 메모리 할당 시

#### `src/core/weights_loader.h/c`
- **역할**: 가중치 파일 로드 및 레이어별 가중치 접근
- **처리 과정**:
  1. `weights.bin` 바이너리 파일 로드
  2. `weights_map.json` 파싱 (간단한 JSON 파서)
  3. 레이어 이름으로 가중치 오프셋 찾기
  4. 가중치 포인터 반환
- **사용처**: 모델 빌드 시 모든 레이어 가중치 로드

### Operations 모듈

#### `src/ops/conv2d.h/c`
- **역할**: 2D Convolution 연산
- **지원 크기**: 1×1, 3×3, 6×6
- **구현 방식**: Naive loop (최적화 전)
- **Inference에서의 역할**: 모든 Conv 레이어 실행
- **Detection에서의 역할**: Detect head의 1×1 Conv

#### `src/ops/batchnorm2d.h/c`
- **역할**: Batch Normalization
- **Inference 모드**: `running_mean`, `running_var` 사용
- **공식**: `output = (input - mean) / sqrt(var + eps) * weight + bias`
- **사용처**: 모든 Conv 레이어 뒤

#### `src/ops/activation.h/c`
- **역할**: 활성화 함수
- **구현**: SiLU (x * sigmoid(x))
- **사용처**: Conv, C3, SPPF 블록 내부

#### `src/ops/pooling.h/c`
- **역할**: MaxPool2D
- **사용처**: SPPF 블록 (5×5, stride=1, padding=2)

#### `src/ops/upsample.h/c`
- **역할**: Nearest neighbor upsampling
- **사용처**: Head의 FPN 경로 (×2 upsampling)

#### `src/ops/concat.h/c`
- **역할**: Channel-wise concatenation
- **사용처**: FPN의 Concat 레이어 (다중 스케일 특징 결합)

### Blocks 모듈

#### `src/blocks/bottleneck.h/c`
- **역할**: Bottleneck 블록 (C3 내부 사용)
- **구조**:
  ```
  Input → Conv1(1×1) → BN → SiLU → Conv2(3×3) → BN → SiLU → Output
                                                              ↑
  Input ─────────────────────────────────────────────────────┘ (shortcut)
  ```
- **사용처**: C3 블록 내부의 반복 구조

#### `src/blocks/c3.h/c`
- **역할**: C3 블록 (YOLOv5의 핵심 블록)
- **구조**:
  ```
  Input
    ├─→ cv1 → BN → SiLU → Bottleneck × n → cv3 → BN → SiLU ──┐
    └─→ cv2 → BN ──────────────────────────────────────────────┤
                                                               ↓
                                                           Concat → cv3 → BN → SiLU → Output
  ```
- **Inference에서의 역할**: Backbone과 Head의 주요 특징 추출 블록

#### `src/blocks/sppf.h/c`
- **역할**: Spatial Pyramid Pooling Fast
- **구조**:
  ```
  Input → cv1 → BN → SiLU → y1
                        ↓
                    MaxPool → y2
                        ↓
                    MaxPool → y3
                        ↓
                    MaxPool → y4
                        ↓
  Concat([y1, y2, y3, y4]) → cv2 → BN → SiLU → Output
  ```
- **사용처**: Backbone의 마지막 블록 (Layer 9)

### Models 모듈

#### `src/models/yolov5s_graph.h/c`
- **역할**: 모델 그래프 정의 (레이어 연결 관계)
- **구조**: `layer_def_t` 배열로 레이어 타입, 연결, save 여부 정의
- **사용처**: Forward pass 실행 순서 결정

#### `src/models/yolov5s_build.h/c`
- **역할**: 모델 인스턴스 생성 및 가중치 로드
- **처리 과정**:
  1. 가중치 로더 생성
  2. 각 레이어 초기화 (채널 수, 커널 크기 등)
  3. 가중치 로드 (weights_map.json 기반)
  4. 모델 구조체 반환
- **출력**: `yolov5s_model_t` (모든 레이어 포함)

#### `src/models/yolov5s_infer.h/c`
- **역할**: Forward pass 실행
- **처리 과정**:
  1. Backbone 실행 (Layers 0-9)
  2. 중간 특징맵 저장 (Layers 3, 4, 5, 6, 7, 9)
  3. Head 실행 (Layers 10-23)
  4. FPN 연결 (저장된 특징맵 재사용)
  5. 최종 특징맵 저장 (Layers 17, 20, 23 → P3, P4, P5)
- **출력**: P3, P4, P5 특징맵

### Postprocess 모듈

#### `src/postprocess/detect.h/c`
- **역할**: Detection 출력 생성 및 디코딩
- **주요 함수**:
  - `detect_forward()`: 1×1 Conv로 detection 출력 생성
  - `detect_decode()`: Grid/anchor 기반 bbox 변환
- **처리 과정**:
  1. P3, P4, P5 특징맵에 1×1 Conv 적용
  2. 각 스케일별로 grid cell과 anchor 조합
  3. Raw outputs를 bbox 좌표로 변환
  4. Confidence threshold 필터링
- **출력**: `detection_t` 배열

#### `src/postprocess/nms.h/c`
- **역할**: Non-Maximum Suppression
- **처리 과정**:
  1. Confidence 기준 정렬
  2. IoU 계산
  3. 중복 제거 (같은 클래스, IoU > threshold)
- **출력**: 최종 detection 리스트

---

## 데이터 흐름 요약

### Inference 파이프라인
```
Image → Preprocess → Tensor(1,3,640,640)
  → Model Build → Forward Pass
  → P3(1,128,80,80), P4(1,256,40,40), P5(1,512,20,20)
```

### Detection 파이프라인
```
P3, P4, P5 Features
  → Detect Head (1×1 Conv)
  → Raw outputs (3 scales × anchors × 85)
  → Decode (Grid/Anchor → BBox)
  → NMS (Remove duplicates)
  → Final detections
```

---

## 모듈 간 의존성

```
main.c
  ├─→ yolov5s_build.c
  │     ├─→ weights_loader.c
  │     ├─→ conv2d.c
  │     ├─→ batchnorm2d.c
  │     ├─→ c3.c
  │     │     ├─→ bottleneck.c
  │     │     ├─→ conv2d.c
  │     │     ├─→ batchnorm2d.c
  │     │     └─→ concat.c
  │     └─→ sppf.c
  │           ├─→ conv2d.c
  │           ├─→ pooling.c
  │           └─→ concat.c
  │
  ├─→ yolov5s_infer.c
  │     ├─→ (uses all ops and blocks)
  │     └─→ tensor.c (for saved features)
  │
  ├─→ detect.c
  │     ├─→ conv2d.c (for detect head)
  │     └─→ tensor.c
  │
  └─→ nms.c
        └─→ detect.h
```

---

## 메모리 레이아웃

### 텐서 저장 순서 (NCHW)
```
[batch][channel][height][width]
data[i] = data[n * (C*H*W) + c * (H*W) + h * W + w]
```

### 가중치 저장 순서
- Conv weight: `[out_channels, in_channels, kernel_h, kernel_w]`
- BN weight: `[channels]`

---

## 성능 고려사항

### 현재 구현
- Naive loop 기반 (최적화 전)
- 메모리 효율적 (ping-pong 버퍼)
- SIMD 최적화 가능 (16-byte alignment)

### 향후 최적화 방향
- SIMD 명령어 활용
- Winograd 알고리즘 (3×3 Conv)
- 메모리 접근 패턴 최적화
