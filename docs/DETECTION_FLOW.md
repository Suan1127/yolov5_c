# Detection 파이프라인 상세 설명

이 문서는 YOLOv5 C 포팅의 Detection 파이프라인이 어떻게 동작하는지 단계별로 설명합니다.

## 전체 흐름도

```
┌─────────────────────────────────────────────┐
│  Forward Pass Output                        │
│  P3: (1, 128, 80, 80)                       │
│  P4: (1, 256, 40, 40)                       │
│  P5: (1, 512, 20, 20)                       │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  Detect Head Forward                        │  src/postprocess/detect.c
│  ┌──────────────────────────────────────┐  │
│  │ P3: Conv(128→255, 1×1)               │  │
│  │   → (1, 255, 80, 80)                 │  │
│  │   → Reshape: (1, 3, 80, 80, 85)      │  │
│  │                                       │  │
│  │ P4: Conv(256→255, 1×1)               │  │
│  │   → (1, 255, 40, 40)                 │  │
│  │   → Reshape: (1, 3, 40, 40, 85)      │  │
│  │                                       │  │
│  │ P5: Conv(512→255, 1×1)               │  │
│  │   → (1, 255, 20, 20)                 │  │
│  │   → Reshape: (1, 3, 20, 20, 85)      │  │
│  └──────────────────────────────────────┘  │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  Decode                                     │  src/postprocess/detect.c
│  ┌──────────────────────────────────────┐  │
│  │ For each scale (P3, P4, P5):         │  │
│  │   For each grid cell (y, x):         │  │
│  │     For each anchor (0, 1, 2):       │  │
│  │       Get raw outputs:               │  │
│  │         tx, ty, tw, th (bbox logits) │  │
│  │         obj_conf (object confidence) │  │
│  │         cls_conf[80] (class logits)  │  │
│  │                                       │  │
│  │       Apply sigmoid:                 │  │
│  │         obj_conf = sigmoid(obj_logit)│  │
│  │         cls_conf = sigmoid(cls_logit)│  │
│  │                                       │  │
│  │       Decode bbox:                   │  │
│  │         center_x = (x + sigmoid(tx))  │  │
│  │                    * stride / 640     │  │
│  │         center_y = (y + sigmoid(ty))  │  │
│  │                    * stride / 640     │  │
│  │         width = exp(tw) * anchor_w   │  │
│  │                    / 640             │  │
│  │         height = exp(th) * anchor_h  │  │
│  │                    / 640              │  │
│  │                                       │  │
│  │       Calculate confidence:          │  │
│  │         conf = obj_conf * max(cls)   │  │
│  │                                       │  │
│  │       Filter:                        │  │
│  │         if conf > threshold:         │  │
│  │           Add to detections          │  │
│  └──────────────────────────────────────┘  │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  NMS (Non-Maximum Suppression)              │  src/postprocess/nms.c
│  ┌──────────────────────────────────────┐  │
│  │ 1. Sort by confidence (descending)    │  │
│  │                                       │  │
│  │ 2. Greedy suppression:                │  │
│  │    For each detection:                │  │
│  │      Mark as kept                     │  │
│  │      For each remaining:              │  │
│  │        If same class:                 │  │
│  │          Calculate IoU               │  │
│  │          If IoU > threshold:          │  │
│  │            Suppress                   │  │
│  └──────────────────────────────────────┘  │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  Final Detections                           │
│  - BBox coordinates (normalized 0-1)        │
│  - Class ID (0-79)                          │
│  - Confidence score                         │
└─────────────────────────────────────────────┘
```

## 단계별 상세 설명

### Step 1: Detect Head Forward

**파일**: `src/postprocess/detect.c` - `detect_forward()`

**입력**: P3, P4, P5 특징맵
- P3: (1, 128, 80, 80) - stride=8, 작은 객체용
- P4: (1, 256, 40, 40) - stride=16, 중간 객체용
- P5: (1, 512, 20, 20) - stride=32, 큰 객체용

**처리**:

```c
// P3 처리
detect_forward(p3_feature, p4_feature, p5_feature, &output, &params)
  │
  ├─→ P3: Conv(128 → 255, 1×1)
  │     │
  │     └─→ conv2d_forward(conv_p3, p3_feature, output->p3_output)
  │           → (1, 255, 80, 80)
  │           │
  │           └─→ 255 = 3 anchors × 85
  │                85 = 4 (bbox) + 1 (obj) + 80 (classes)
  │
  ├─→ P4: Conv(256 → 255, 1×1)
  │     └─→ (1, 255, 40, 40)
  │
  └─→ P5: Conv(512 → 255, 1×1)
        └─→ (1, 255, 20, 20)
```

**출력 형식**:
- 각 스케일: (1, 255, H, W)
- 개념적으로: (1, 3, H, W, 85)
  - 3: 앵커 개수
  - H, W: 그리드 크기
  - 85: [tx, ty, tw, th, obj_conf, cls_0, ..., cls_79]

**메모리 레이아웃**:
```
output->p3_output->data[anchor * 85 * 80 * 80 + y * 80 + x]
  = anchor 0의 (y, x) 위치의 첫 번째 값 (tx)

output->p3_output->data[anchor * 85 * 80 * 80 + 1 * 80 * 80 + y * 80 + x]
  = anchor 0의 (y, x) 위치의 두 번째 값 (ty)
```

### Step 2: Decode

**파일**: `src/postprocess/detect.c` - `detect_decode()`

**목적**: Raw detection outputs를 실제 bbox 좌표로 변환

#### 2.1 Anchor 정보

```c
// P3 (stride=8, grid=80×80)
anchors[0] = [10, 13, 16, 30, 33, 23]
  → anchor 0: (w=10, h=13)
  → anchor 1: (w=16, h=30)
  → anchor 2: (w=33, h=23)

// P4 (stride=16, grid=40×40)
anchors[1] = [30, 61, 62, 45, 59, 119]
  → anchor 0: (w=30, h=61)
  → anchor 1: (w=62, h=45)
  → anchor 2: (w=59, h=119)

// P5 (stride=32, grid=20×20)
anchors[2] = [116, 90, 156, 198, 373, 326]
  → anchor 0: (w=116, h=90)
  → anchor 1: (w=156, h=198)
  → anchor 2: (w=373, h=326)
```

#### 2.2 Decode 과정

**P3 스케일 예시** (grid=80×80, stride=8):

```c
For y = 0 to 79:
  For x = 0 to 79:
    For anchor = 0 to 2:
      // 1. Get raw outputs
      base_idx = anchor * 85 * 80 * 80 + y * 80 + x;
      
      tx = feature->data[base_idx + 0 * 80 * 80];
      ty = feature->data[base_idx + 1 * 80 * 80];
      tw = feature->data[base_idx + 2 * 80 * 80];
      th = feature->data[base_idx + 3 * 80 * 80];
      obj_logit = feature->data[base_idx + 4 * 80 * 80];
      
      // 2. Apply sigmoid
      obj_conf = 1.0 / (1.0 + exp(-obj_logit));
      
      // 3. Get class confidences
      max_cls_conf = 0.0;
      max_cls_id = 0;
      for c = 0 to 79:
        cls_logit = feature->data[base_idx + (5 + c) * 80 * 80];
        cls_conf[c] = 1.0 / (1.0 + exp(-cls_logit));
        if cls_conf[c] > max_cls_conf:
          max_cls_conf = cls_conf[c];
          max_cls_id = c;
      
      // 4. Calculate final confidence
      conf = obj_conf * max_cls_conf;
      
      // 5. Filter by threshold
      if conf < conf_threshold (0.25):
        continue;  // Skip this detection
      
      // 6. Decode bbox coordinates
      anchor_w = anchors[0][anchor * 2];      // e.g., 10, 16, 33
      anchor_h = anchors[0][anchor * 2 + 1];  // e.g., 13, 30, 23
      
      // YOLOv5 decode formula:
      center_x = (x + sigmoid(tx)) * stride / input_size;
      center_y = (y + sigmoid(ty)) * stride / input_size;
      width = exp(tw) * anchor_w / input_size;
      height = exp(th) * anchor_h / input_size;
      
      // 7. Create detection
      detection = {
        .x = center_x,      // Normalized [0, 1]
        .y = center_y,      // Normalized [0, 1]
        .w = width,         // Normalized [0, 1]
        .h = height,        // Normalized [0, 1]
        .conf = conf,
        .cls_id = max_cls_id,
        .cls_conf = cls_conf[80]
      };
      
      Add to detections list;
```

**Decode 공식 설명**:

1. **Center coordinates**:
   ```
   center_x = (grid_x + sigmoid(tx)) * stride / input_size
   ```
   - `grid_x`: 그리드 셀의 x 좌표 (0~79 for P3)
   - `sigmoid(tx)`: 그리드 셀 내부 오프셋 [0, 1]
   - `stride`: 8 (P3), 16 (P4), 32 (P5)
   - 결과: 전체 이미지에서의 정규화된 x 좌표 [0, 1]

2. **Size**:
   ```
   width = exp(tw) * anchor_w / input_size
   ```
   - `exp(tw)`: 앵커에 대한 스케일 팩터
   - `anchor_w`: 해당 앵커의 기본 너비
   - 결과: 정규화된 너비 [0, 1]

**예시 계산** (P3, grid (10, 20), anchor 0):
```
Input: tx=0.5, ty=-0.3, tw=0.2, th=0.1
Anchor: w=10, h=13
Stride: 8
Input size: 640

center_x = (10 + sigmoid(0.5)) * 8 / 640
         = (10 + 0.622) * 8 / 640
         = 0.1326

center_y = (20 + sigmoid(-0.3)) * 8 / 640
         = (20 + 0.426) * 8 / 640
         = 0.2553

width = exp(0.2) * 10 / 640
      = 1.221 * 10 / 640
      = 0.0191

height = exp(0.1) * 13 / 640
       = 1.105 * 13 / 640
       = 0.0225
```

### Step 3: NMS (Non-Maximum Suppression)

**파일**: `src/postprocess/nms.c` - `nms()`

**목적**: 중복 detection 제거

#### 3.1 IoU 계산

**파일**: `src/postprocess/nms.c` - `calculate_iou()`

```c
calculate_iou(box1, box2)
  │
  ├─→ Convert center+size to corners:
  │     box1: (x1, y1, w1, h1)
  │       → x1_min = x1 - w1/2
  │       → y1_min = y1 - h1/2
  │       → x1_max = x1 + w1/2
  │       → y1_max = y1 + h1/2
  │
  ├─→ Calculate intersection:
  │     x_i_min = max(x1_min, x2_min)
  │     y_i_min = max(y1_min, y2_min)
  │     x_i_max = min(x1_max, x2_max)
  │     y_i_max = min(y1_max, y2_max)
  │
  │     if x_i_max < x_i_min or y_i_max < y_i_min:
  │       return 0.0  // No intersection
  │
  │     intersection = (x_i_max - x_i_min) * (y_i_max - y_i_min)
  │
  └─→ Calculate IoU:
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        IoU = intersection / union
```

**예시**:
```
Box1: center=(0.5, 0.5), size=(0.2, 0.2)
  → corners: (0.4, 0.4) to (0.6, 0.6)

Box2: center=(0.55, 0.55), size=(0.2, 0.2)
  → corners: (0.45, 0.45) to (0.65, 0.65)

Intersection: (0.6-0.45) × (0.6-0.45) = 0.0225
Union: 0.04 + 0.04 - 0.0225 = 0.0575
IoU: 0.0225 / 0.0575 = 0.391
```

#### 3.2 NMS 알고리즘

```c
nms(detections, num_detections, output, count, iou_threshold, max_detections)
  │
  ├─→ 1. Sort by confidence (descending)
  │     // Bubble sort (간단한 구현)
  │     // 실제로는 더 효율적인 정렬 사용 가능
  │
  ├─→ 2. Initialize keep array
  │     keep[i] = 0 (not processed)
  │
  └─→ 3. Greedy NMS:
        for i = 0 to num_detections:
          if keep[i] == -1:  // Already suppressed
            continue
          
          keep[i] = 1  // Mark as kept
          keep_count++
          
          // Suppress overlapping detections
          for j = i+1 to num_detections:
            if keep[j] == -1:  // Already suppressed
              continue
            
            // Only suppress if same class
            if detections[i].cls_id != detections[j].cls_id:
              continue
            
            // Calculate IoU
            iou = calculate_iou(&detections[i], &detections[j])
            
            if iou > iou_threshold (0.45):
              keep[j] = -1  // Suppress
```

**NMS 예시**:
```
Input detections (sorted by confidence):
  0: person, conf=0.95, bbox=(0.5, 0.5, 0.2, 0.2)
  1: person, conf=0.90, bbox=(0.52, 0.52, 0.2, 0.2)  // Overlaps with 0
  2: car, conf=0.85, bbox=(0.3, 0.3, 0.15, 0.15)
  3: person, conf=0.80, bbox=(0.51, 0.51, 0.19, 0.19)  // Overlaps with 0

Processing:
  i=0: Keep detection 0
        IoU(0,1) = 0.75 > 0.45 → Suppress 1
        IoU(0,3) = 0.82 > 0.45 → Suppress 3
  i=1: Already suppressed, skip
  i=2: Keep detection 2 (different class)
  i=3: Already suppressed, skip

Output: [detection 0, detection 2]
```

---

## Detection 파이프라인 데이터 흐름

### 입력 → 출력 변환

```
P3 Feature (1, 128, 80, 80)
  ↓ [1×1 Conv]
Detection Output (1, 255, 80, 80)
  ↓ [Reshape conceptually]
(1, 3, 80, 80, 85)
  ↓ [Decode]
Detections: [
  {x, y, w, h, conf, cls_id} at grid (0,0), anchor 0,
  {x, y, w, h, conf, cls_id} at grid (0,0), anchor 1,
  ...
  {x, y, w, h, conf, cls_id} at grid (79,79), anchor 2
]
  ↓ [Filter by confidence]
Detections: [only conf > 0.25]
  ↓ [NMS]
Final Detections: [duplicates removed]
```

### 스케일별 처리량

**P3 (80×80, stride=8)**:
- Grid cells: 80 × 80 = 6,400
- Anchors: 3
- Total predictions: 6,400 × 3 = 19,200
- After confidence filter: ~1,000-5,000 (예상)

**P4 (40×40, stride=16)**:
- Grid cells: 40 × 40 = 1,600
- Anchors: 3
- Total predictions: 1,600 × 3 = 4,800
- After confidence filter: ~500-2,000 (예상)

**P5 (20×20, stride=32)**:
- Grid cells: 20 × 20 = 400
- Anchors: 3
- Total predictions: 400 × 3 = 1,200
- After confidence filter: ~100-500 (예상)

**총 예상**: ~1,600-7,500 detections → NMS 후 ~100-1,000

---

## 핵심 구현 세부사항

### Detect Head의 1×1 Conv

**현재 구현**: Placeholder (가중치 로드 필요)

**필요한 가중치**:
- `model.24.m.0.weight`: P3용 (128 → 255)
- `model.24.m.1.weight`: P4용 (256 → 255)
- `model.24.m.2.weight`: P5용 (512 → 255)

**구현 예시**:
```c
// P3 detect head
conv2d_params_t detect_params = {
  .out_channels = 255,
  .kernel_size = 1,
  .stride = 1,
  .padding = 0
};
conv2d_init(&detect_conv_p3, 128, &detect_params);
// Load weights: model.24.m.0.weight
```

### Decode의 텐서 인덱싱

**메모리 레이아웃**:
```
Output tensor: (1, 255, H, W)
  = (1, 3*85, H, W)

Access pattern:
  base = anchor * 85 * H * W + y * W + x
  tx = data[base + 0 * H * W]
  ty = data[base + 1 * H * W]
  tw = data[base + 2 * H * W]
  th = data[base + 3 * H * W]
  obj = data[base + 4 * H * W]
  cls_0 = data[base + 5 * H * W]
  ...
  cls_79 = data[base + 84 * H * W]
```

### NMS 최적화 가능 영역

**현재 구현**: O(n²) 복잡도 (간단한 버블 정렬 + 이중 루프)

**최적화 방향**:
1. 효율적인 정렬 알고리즘 (Quick sort 등)
2. 공간 인덱싱 (Grid-based NMS)
3. IoU 계산 최적화 (SIMD)

---

## Detection 결과 형식

### `detection_t` 구조체

```c
typedef struct {
    float x;           // Center x (normalized 0-1)
    float y;           // Center y (normalized 0-1)
    float w;           // Width (normalized 0-1)
    float h;           // Height (normalized 0-1)
    float conf;        // Object confidence × class confidence
    float cls_conf[80]; // Class confidences (COCO 80 classes)
    int32_t cls_id;    // Class ID (0-79)
} detection_t;
```

### 출력 파일 형식

`data/output_detections.txt`:
```
Total detections: 25
Format: class_id confidence x y w h (normalized 0-1)
Format: class_id confidence x_pixel y_pixel w_pixel h_pixel

0 0.8523 0.1234 0.5678 0.2345 0.3456
0 0.8523 79.0 363.4 150.1 221.2
2 0.7432 0.4567 0.2345 0.1234 0.2345
2 0.7432 292.3 150.1 79.0 150.1
...
```

---

## COCO 클래스 매핑

**Class ID 0-79** (COCO dataset):
- 0: person
- 1: bicycle
- 2: car
- ...
- 79: toothbrush

**출력 해석**:
```c
detection.cls_id = 2;  // "car"
detection.conf = 0.85;  // 85% confidence
detection.x = 0.5;      // Center at 50% of image width
detection.y = 0.3;      // Center at 30% of image height
detection.w = 0.2;      // Width is 20% of image
detection.h = 0.15;     // Height is 15% of image
```

---

## 성능 고려사항

### Decode 단계
- **연산량**: 각 스케일별 grid × anchor × (sigmoid + exp)
- **병목**: Sigmoid 계산 (exp 함수 호출)
- **최적화**: LUT (Look-Up Table) 사용 가능

### NMS 단계
- **연산량**: O(n²) IoU 계산
- **병목**: IoU 계산 (특히 많은 detection일 때)
- **최적화**: 공간 인덱싱, IoU 근사

---

## 검증 포인트

1. **Decode 정확도**:
   - Grid offset 계산 정확성
   - Anchor 적용 정확성
   - Sigmoid/exp 수치 정확도

2. **NMS 정확도**:
   - IoU 계산 정확성
   - Suppression 로직 정확성
   - 최종 detection 수 적절성

3. **End-to-end 검증**:
   - Python YOLOv5와 bbox 비교
   - IoU 기반 매칭
   - 허용 오차: 위치 ±5%, 크기 ±10%
