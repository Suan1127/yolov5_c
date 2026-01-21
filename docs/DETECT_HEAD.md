# Detect Head 상세 설명

YOLOv5n의 Detect head는 P3, P4, P5 특징맵을 입력으로 받아 객체 탐지를 위한 출력을 생성합니다.

## 개요

- **레이어**: Layer 24
- **입력**: 
  - P3: `(1, 64, 80, 80)` from Layer 17
  - P4: `(1, 128, 40, 40)` from Layer 20
  - P5: `(1, 256, 20, 20)` from Layer 23
- **출력**: 
  - P3: `(1, 255, 80, 80)`
  - P4: `(1, 255, 40, 40)`
  - P5: `(1, 255, 20, 20)`
- **역할**: 각 스케일의 특징맵을 탐지 출력으로 변환

---

## 구조

Detect head는 3개의 독립적인 1×1 Convolution 레이어로 구성됩니다:

```
P3 (64, 80, 80)
    ↓ Conv(64→255, 1×1)
P3 Output (255, 80, 80)

P4 (128, 40, 40)
    ↓ Conv(128→255, 1×1)
P4 Output (255, 40, 40)

P5 (256, 20, 20)
    ↓ Conv(256→255, 1×1)
P5 Output (255, 20, 20)
```

---

## 출력 형식

각 출력 텐서는 `(1, 255, H, W)` 형태이며, 255는 다음과 같이 구성됩니다:

```
255 = 3 anchors × 85 values
85 = 4 (bbox) + 1 (obj_conf) + 80 (class_conf)
```

### 메모리 레이아웃

```
Output tensor: (1, 255, H, W)
  = (1, 3*85, H, W)

각 anchor별로:
  anchor 0: [tx, ty, tw, th, obj_conf, cls_0, ..., cls_79]
  anchor 1: [tx, ty, tw, th, obj_conf, cls_0, ..., cls_79]
  anchor 2: [tx, ty, tw, th, obj_conf, cls_0, ..., cls_79]

Access pattern:
  base = anchor * 85 * H * W + y * W + x
  tx = data[base + 0 * H * W]
  ty = data[base + 1 * H * W]
  tw = data[base + 2 * H * W]
  th = data[base + 3 * H * W]
  obj_conf = data[base + 4 * H * W]
  cls_0 = data[base + 5 * H * W]
  ...
  cls_79 = data[base + 84 * H * W]
```

---

## 각 스케일별 역할

### P3 Output (80×80, stride=8)
- **입력**: `(1, 64, 80, 80)`
- **출력**: `(1, 255, 80, 80)`
- **역할**: 작은 객체 탐지
- **특징**:
  - 가장 높은 해상도
  - 작은 객체에 대한 세밀한 정보 제공
  - 총 예측 수: 80 × 80 × 3 = 19,200

### P4 Output (40×40, stride=16)
- **입력**: `(1, 128, 40, 40)`
- **출력**: `(1, 255, 40, 40)`
- **역할**: 중간 크기 객체 탐지
- **특징**:
  - 중간 해상도
  - 중간 크기 객체에 적합
  - 총 예측 수: 40 × 40 × 3 = 4,800

### P5 Output (20×20, stride=32)
- **입력**: `(1, 256, 20, 20)`
- **출력**: `(1, 255, 20, 20)`
- **역할**: 큰 객체 탐지
- **특징**:
  - 가장 낮은 해상도
  - 큰 객체에 대한 전역 정보 제공
  - 총 예측 수: 20 × 20 × 3 = 1,200

---

## Anchor 정보

각 스케일마다 3개의 앵커가 사용됩니다:

### P3 Anchors (stride=8)
```
Anchor 0: (w=10, h=13)
Anchor 1: (w=16, h=30)
Anchor 2: (w=33, h=23)
```

### P4 Anchors (stride=16)
```
Anchor 0: (w=30, h=61)
Anchor 1: (w=62, h=45)
Anchor 2: (w=59, h=119)
```

### P5 Anchors (stride=32)
```
Anchor 0: (w=116, h=90)
Anchor 1: (w=156, h=198)
Anchor 2: (w=373, h=326)
```

---

## 출력 값 의미

### Bounding Box (tx, ty, tw, th)
- **tx, ty**: 그리드 셀 내에서의 중심점 오프셋 (0-1 범위)
- **tw, th**: 앵커에 대한 너비/높이 비율
- **최종 좌표 계산**:
  ```python
  x = (tx * 2 - 0.5 + grid_x) * stride
  y = (ty * 2 - 0.5 + grid_y) * stride
  w = (tw * 2) ** 2 * anchor_w
  h = (th * 2) ** 2 * anchor_h
  ```

### Object Confidence
- 객체 존재 확률 (0-1 범위)
- Sigmoid 활성화 함수 적용

### Class Confidence
- 각 클래스에 대한 확률 (80개 클래스)
- Sigmoid 활성화 함수 적용
- 최종 클래스 확률 = object_conf × class_conf

---

## Decode 과정

Detect head의 출력은 raw 값이므로, 다음 단계를 거쳐 실제 bounding box로 변환됩니다:

1. **Grid 좌표 계산**: 각 그리드 셀의 위치 계산
2. **Bounding Box 변환**: tx, ty, tw, th를 실제 픽셀 좌표로 변환
3. **Confidence 필터링**: object_conf > threshold인 예측만 유지
4. **NMS (Non-Maximum Suppression)**: 중복 박스 제거

---

## 구현 상세

### C 구현 (`src/postprocess/detect.c`)

```c
// Detect head conv 레이어 (모델에 포함)
detect_convs[0]: Conv(64→255, 1×1)   // P3
detect_convs[1]: Conv(128→255, 1×1)  // P4
detect_convs[2]: Conv(256→255, 1×1)  // P5

// Forward pass
detect_forward(model, p3_feature, p4_feature, p5_feature, &output, &params)
  ├─→ conv2d_forward(&detect_convs[0], p3_feature, output->p3_output)
  ├─→ conv2d_forward(&detect_convs[1], p4_feature, output->p4_output)
  └─→ conv2d_forward(&detect_convs[2], p5_feature, output->p5_output)
```

### 가중치 로딩

```c
// weights_map.json에서:
"model.24.m.0.weight": P3용 conv 가중치 (64, 255, 1, 1)
"model.24.m.0.bias": P3용 conv bias (255,)
"model.24.m.1.weight": P4용 conv 가중치 (128, 255, 1, 1)
"model.24.m.1.bias": P4용 conv bias (255,)
"model.24.m.2.weight": P5용 conv 가중치 (256, 255, 1, 1)
"model.24.m.2.bias": P5용 conv bias (255,)
```

---

## 출력 파일 저장

### Python Golden
- `output_1_0.bin`: P3 detect head conv 출력 `(1, 255, 80, 80)`
- `output_1_1.bin`: P4 detect head conv 출력 `(1, 255, 40, 40)`
- `output_1_2.bin`: P5 detect head conv 출력 `(1, 255, 20, 20)`

### C Implementation
- `output_1_0.bin`: P3 detect head conv 출력 `(1, 255, 80, 80)`
- `output_1_1.bin`: P4 detect head conv 출력 `(1, 255, 40, 40)`
- `output_1_2.bin`: P5 detect head conv 출력 `(1, 255, 20, 20)`

---

## 검증 결과

비교 테스트 결과 모든 스케일의 출력이 Python golden과 일치합니다:

- **P3**: Max diff: 5.53e-05, Mean diff: 3.17e-06 ✅
- **P4**: Max diff: 3.05e-05, Mean diff: 2.65e-06 ✅
- **P5**: Max diff: 2.96e-05, Mean diff: 2.61e-06 ✅

모든 출력이 tolerance (1e-4) 내에서 일치합니다.

---

## 관련 문서

- `docs/LAYER_ARCHITECTURE.md`: 전체 레이어 아키텍처
- `docs/HEAD_LAYERS.md`: Head 레이어 상세 설명
- `docs/DETECTION_FLOW.md`: Detection 파이프라인 상세 설명
- `src/postprocess/detect.c`: Detect head 구현
- `src/postprocess/detect.h`: Detect head 헤더
