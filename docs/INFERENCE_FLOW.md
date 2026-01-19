# Inference 파이프라인 상세 설명

이 문서는 YOLOv5 C 포팅의 Inference 파이프라인이 어떻게 동작하는지 단계별로 설명합니다.

## 전체 흐름도

```
┌─────────────┐
│ Input Image │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│  Preprocessing    │  tools/preprocess.py
│  - Letterbox      │
│  - Normalize      │
│  - NCHW           │
└──────┬────────────┘
       │
       ▼
┌──────────────────┐
│  Tensor Load     │  src/core/tensor.c
│  (1,3,640,640)   │
└──────┬────────────┘
       │
       ▼
┌──────────────────┐
│  Model Build     │  src/models/yolov5s_build.c
│  - Load weights   │
│  - Init layers    │
└──────┬────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│  Forward Pass                               │  src/models/yolov5s_infer.c
│  ┌──────────────────────────────────────┐  │
│  │ BACKBONE (Layers 0-9)                │  │
│  │                                       │  │
│  │ 0: Conv(3→32, 6×6, s=2)              │  │
│  │    → (1,32,320,320)                  │  │
│  │                                       │  │
│  │ 1: Conv(32→64, 3×3, s=2)             │  │
│  │    → (1,64,160,160)                  │  │
│  │                                       │  │
│  │ 2: C3(64→64, n=1)                    │  │
│  │    → (1,64,160,160)                  │  │
│  │                                       │  │
│  │ 3: Conv(64→128, 3×3, s=2) → SAVE[3] │  │
│  │    → (1,128,80,80)                   │  │
│  │                                       │  │
│  │ 4: C3(128→128, n=2) → SAVE[4]       │  │
│  │    → (1,128,80,80)                   │  │
│  │                                       │  │
│  │ 5: Conv(128→256, 3×3, s=2) → SAVE[5]│  │
│  │    → (1,256,40,40)                   │  │
│  │                                       │  │
│  │ 6: C3(256→256, n=3) → SAVE[6]       │  │
│  │    → (1,256,40,40)                   │  │
│  │                                       │  │
│  │ 7: Conv(256→512, 3×3, s=2) → SAVE[7]│  │
│  │    → (1,512,20,20)                   │  │
│  │                                       │  │
│  │ 8: C3(512→512, n=1)                  │  │
│  │    → (1,512,20,20)                   │  │
│  │                                       │  │
│  │ 9: SPPF(512→512, k=5) → SAVE[9]    │  │
│  │    → (1,512,20,20)                   │  │
│  └──────────────────────────────────────┘  │
│                                              │
│  ┌──────────────────────────────────────┐  │
│  │ HEAD (Layers 10-23)                  │  │
│  │                                       │  │
│  │ FPN Top-down:                        │  │
│  │  10: Conv(512→256, 1×1)             │  │
│  │      → (1,256,20,20)                │  │
│  │  11: Upsample(×2)                   │  │
│  │      → (1,256,40,40)                │  │
│  │  12: Concat([11, 6])                │  │
│  │      → (1,512,40,40)                │  │
│  │  13: C3(512→256, n=1)               │  │
│  │      → (1,256,40,40)                │  │
│  │  14: Conv(256→128, 1×1)             │  │
│  │      → (1,128,40,40)                │  │
│  │  15: Upsample(×2)                   │  │
│  │      → (1,128,80,80)                │  │
│  │  16: Concat([15, 4])                │  │
│  │      → (1,256,80,80)                │  │
│  │  17: C3(256→128, n=1) → SAVE[17]   │  │
│  │      → (1,128,80,80) = P3          │  │
│  │                                       │  │
│  │ Bottom-up:                          │  │
│  │  18: Conv(128→128, 3×3, s=2)       │  │
│  │      → (1,128,40,40)                │  │
│  │  19: Concat([18, 13])               │  │
│  │      → (1,256,40,40)                │  │
│  │  20: C3(256→256, n=1) → SAVE[20]   │  │
│  │      → (1,256,40,40) = P4          │  │
│  │  21: Conv(256→256, 3×3, s=2)       │  │
│  │      → (1,256,20,20)                │  │
│  │  22: Concat([21, 10])               │  │
│  │      → (1,512,20,20)                │  │
│  │  23: C3(512→512, n=1) → SAVE[23]   │  │
│  │      → (1,512,20,20) = P5          │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
       │
       ▼
┌──────────────────┐
│  Output Features │
│  P3, P4, P5      │
└──────────────────┘
```

## 단계별 상세 설명

### Step 1: 입력 준비

**파일**: `tools/preprocess.py`

```python
# 입력: data/images/bus.jpg
# 처리:
1. cv2.imread() → (H, W, 3) BGR
2. cv2.cvtColor(BGR2RGB) → (H, W, 3) RGB
3. letterbox() → (640, 640, 3) RGB (비율 유지, padding 추가)
4. / 255.0 → (640, 640, 3) [0.0, 1.0]
5. transpose(2,0,1) → (3, 640, 640)
6. expand_dims(0) → (1, 3, 640, 640)

# 출력: data/inputs/bus.bin
# 메타: data/inputs/bus_meta.txt
```

**핵심 함수**:
- `letterbox()`: 비율 유지하면서 640×640으로 리사이즈
- `save_tensor()`: NCHW 텐서를 바이너리로 저장

### Step 2: 모델 빌드

**파일**: `src/models/yolov5s_build.c`

**처리 과정**:

```c
yolov5s_build(weights_path, model_meta_path)
  │
  ├─→ weights_loader_create(weights_path)
  │     ├─→ Load weights.bin (전체 가중치)
  │     └─→ Parse weights_map.json (레이어별 오프셋)
  │
  ├─→ Initialize Backbone Conv layers (0, 1, 3, 5, 7)
  │     ├─→ conv2d_init() - 메모리 할당
  │     ├─→ batchnorm2d_init() - 메모리 할당
  │     └─→ load_conv_bn_layer() - 가중치 로드
  │
  ├─→ Initialize Backbone C3 blocks (2, 4, 6, 8)
  │     ├─→ c3_init() - 내부 레이어 초기화
  │     └─→ c3_load_weights() - 가중치 로드
  │
  ├─→ Initialize SPPF (9)
  │     ├─→ sppf_init()
  │     └─→ sppf_load_weights()
  │
  └─→ Initialize Head layers (10, 13, 14, 17, 18, 20, 21, 23)
        └─→ (동일한 과정)
```

**가중치 로드 예시** (Layer 0):
```c
// weights_map.json에서 찾기
"model.0.conv.weight": {offset: 256, shape: [32, 3, 6, 6]}
"model.0.bn.weight": {offset: 128, shape: [32]}
"model.0.bn.bias": {offset: 0, shape: [32]}

// weights.bin에서 해당 오프셋의 데이터 읽기
float* weight = weights_loader->data + (offset / sizeof(float));
conv2d_load_weights(&layer, weight, NULL);
```

### Step 3: Forward Pass - Backbone

**파일**: `src/models/yolov5s_infer.c`

#### Layer 0: Conv(3→32, 6×6, s=2, p=2)

```c
// 입력: input (1, 3, 640, 640)
// 출력: buf_a (1, 32, 320, 320)

conv2d_forward(&model->backbone_convs[0].conv, input, buf_a)
  │
  ├─→ Calculate output size: (640+4-6)/2+1 = 320
  │
  └─→ Convolution loop:
        for each output channel (0..31):
          for each input channel (0..2):
            for each kernel position (0..5, 0..5):
              for each output position (0..319, 0..319):
                output += input[y*2+ky-2, x*2+kx-2] * weight[oc,ic,ky,kx]

batchnorm2d_forward(&model->backbone_convs[0].bn, buf_a, buf_a)
  │
  └─→ For each channel:
        output = (input - running_mean) / sqrt(running_var + eps) * weight + bias

activation_silu(buf_a)
  │
  └─→ For each element:
        output = input * sigmoid(input)
```

#### Layer 2: C3(64→64, n=1)

```c
// 입력: buf_a (1, 64, 160, 160)
// 출력: buf_b (1, 64, 160, 160)

c3_forward(&model->backbone_c3s[0].block, buf_a, buf_b, NULL, NULL)
  │
  ├─→ Main path:
  │     cv1: Conv(64→32, 1×1) → BN → SiLU
  │     │     → workspace1 (1, 32, 160, 160)
  │     │
  │     Bottleneck × 1:
  │       Conv1(32→32, 1×1) → BN → SiLU
  │       Conv2(32→32, 3×3) → BN → SiLU
  │       Add shortcut
  │     │     → workspace1 (1, 32, 160, 160)
  │     │
  │     cv3: Conv(64→64, 1×1) → BN → SiLU
  │           → workspace1 (1, 64, 160, 160)
  │
  ├─→ Skip path:
  │     cv2: Conv(64→32, 1×1) → BN
  │           → workspace2 (1, 32, 160, 160)
  │
  └─→ Concat + cv3:
        Concat([workspace1, workspace2]) → (1, 64, 160, 160)
        cv3: Conv(64→64, 1×1) → BN → SiLU
              → buf_b (1, 64, 160, 160)
```

#### Layer 9: SPPF(512→512, k=5)

```c
// 입력: buf_a (1, 512, 20, 20)
// 출력: buf_b (1, 512, 20, 20)

sppf_forward(&model->sppf, buf_a, buf_b, NULL, NULL, NULL)
  │
  ├─→ cv1: Conv(512→256, 1×1) → BN → SiLU
  │     → workspace1 (1, 256, 20, 20) = y1
  │
  ├─→ MaxPool chain:
  │     y2 = MaxPool(y1, 5×5, s=1, p=2)
  │     y3 = MaxPool(y2, 5×5, s=1, p=2)
  │     y4 = MaxPool(y3, 5×5, s=1, p=2)
  │
  └─→ Concat + cv2:
        Concat([y1, y2, y3, y4]) → (1, 1024, 20, 20)
        cv2: Conv(1024→512, 1×1) → BN → SiLU
              → buf_b (1, 512, 20, 20)
```

### Step 4: Forward Pass - Head

#### FPN Top-down 경로

**Layer 10-17**:
```c
// Layer 10: Conv(512→256, 1×1)
conv2d_forward(&head_convs[0].conv, buf_a, buf_b)
  → (1, 256, 20, 20)
// Save for later concat (Layer 22)

// Layer 11: Upsample(×2)
upsample_forward(&upsample_params, buf_b, buf_a)
  → (1, 256, 40, 40)
  // Nearest neighbor: 각 픽셀을 2×2로 복제

// Layer 12: Concat([11, 6])
// Get saved feature from Layer 6
tensor_t* layer6 = yolov5s_get_saved_feature(model, 6);  // (1, 256, 40, 40)
concat_forward([buf_a, layer6], 2, buf_b)
  → (1, 512, 40, 40)  // 256 + 256

// Layer 13: C3(512→256, n=1, shortcut=False)
c3_forward(&head_c3s[0].block, buf_b, buf_a, NULL, NULL)
  → (1, 256, 40, 40)
// Save for later concat (Layer 19)

// ... (Layer 14-16) ...

// Layer 17: C3(256→128, n=1) → P3
c3_forward(&head_c3s[1].block, buf_b, buf_a, NULL, NULL)
  → (1, 128, 80, 80)
save_feature(model, 17, buf_a)  // P3 저장
```

#### Bottom-up 경로

**Layer 18-23**:
```c
// Layer 18: Conv(128→128, 3×3, s=2)
conv2d_forward(&head_convs[2].conv, buf_a, buf_b)
  → (1, 128, 40, 40)

// Layer 19: Concat([18, 13])
tensor_t* layer13 = yolov5s_get_saved_feature(model, 13);  // (1, 256, 40, 40)
concat_forward([buf_b, layer13], 2, buf_a)
  → (1, 256, 40, 40)  // 128 + 256 (실제로는 256으로 맞춤)

// Layer 20: C3(256→256, n=1) → P4
c3_forward(&head_c3s[2].block, buf_a, buf_b, NULL, NULL)
  → (1, 256, 40, 40)
save_feature(model, 20, buf_b)  // P4 저장

// ... (Layer 21-22) ...

// Layer 23: C3(512→512, n=1) → P5
c3_forward(&head_c3s[3].block, buf_b, buf_a, NULL, NULL)
  → (1, 512, 20, 20)
save_feature(model, 23, buf_a)  // P5 저장
```

### Step 5: Save 리스트 관리

**저장되는 특징맵**:
- Layer 3: (1, 128, 80, 80) - FPN 연결용 (사용 안 함)
- Layer 4: (1, 128, 80, 80) - Layer 16 Concat용
- Layer 5: (1, 256, 40, 40) - 사용 안 함
- Layer 6: (1, 256, 40, 40) - Layer 12 Concat용
- Layer 7: (1, 512, 20, 20) - 사용 안 함
- Layer 9: (1, 512, 20, 20) - Layer 10 입력
- Layer 17: (1, 128, 80, 80) - **P3** (Detect 입력)
- Layer 20: (1, 256, 40, 40) - **P4** (Detect 입력)
- Layer 23: (1, 512, 20, 20) - **P5** (Detect 입력)

**메모리 관리**:
```c
// Save feature 함수
save_feature(model, layer_idx, feature)
  │
  ├─→ Find save index: layer_idx → save_map[] → save_idx
  │
  ├─→ Free old feature if exists
  │
  └─→ Allocate and copy:
        model->saved_features[save_idx] = tensor_create(...)
        tensor_copy(model->saved_features[save_idx], feature)
```

---

## 핵심 구현 세부사항

### Ping-pong 버퍼 전략

```c
tensor_t* buf_a = tensor_create(...);
tensor_t* buf_b = tensor_create(...);

// Layer 0: input → buf_a
conv2d_forward(..., input, buf_a);

// Layer 1: buf_a → buf_b
conv2d_forward(..., buf_a, buf_b);

// Swap
tensor_t* temp = buf_a;
buf_a = buf_b;
buf_b = temp;

// Layer 2: buf_a → buf_b
c3_forward(..., buf_a, buf_b);
```

**장점**:
- 메모리 효율적 (2개 버퍼만 사용)
- 복사 최소화

### C3 블록 내부 처리

```c
// C3 forward 과정
c3_forward(c3_block, input, output, workspace1, workspace2)
  │
  ├─→ Main path:
  │     cv1(input) → workspace1
  │     │
  │     For each bottleneck:
  │       bottleneck(workspace1) → workspace1
  │     │
  │     cv3(workspace1) → workspace1
  │
  ├─→ Skip path:
  │     cv2(input) → workspace2
  │
  └─→ Concat + cv3:
        Concat([workspace1, workspace2]) → workspace2
        cv3(workspace2) → output
```

### Upsample 구현

```c
// Nearest neighbor ×2
for each input pixel (y, x):
  output[2*y, 2*x] = input[y, x]
  output[2*y, 2*x+1] = input[y, x]
  output[2*y+1, 2*x] = input[y, x]
  output[2*y+1, 2*x+1] = input[y, x]
```

### Concat 구현

```c
// Channel-wise concatenation
output_channels = 0;
for each input tensor:
  for each channel in input:
    Copy input[channel] → output[output_channels]
    output_channels++
```

---

## 메모리 사용량 예상

**640×640 입력 기준 (FP32)**:
- 최대 단일 텐서: (1, 512, 80, 80) = 13.1 MB
- Ping-pong 버퍼: 2 × 13.1 MB = 26.2 MB
- Saved features (9개): 약 50 MB
- Workspace (C3, SPPF): 약 20 MB
- **총 예상**: 약 100-150 MB

---

## 성능 병목 지점

1. **Conv2D**: 가장 많은 연산량 (특히 3×3 Conv)
2. **C3 블록**: 여러 Conv + Concat
3. **SPPF**: MaxPool 반복
4. **메모리 복사**: Concat, Save operations

**최적화 가능 영역**:
- SIMD를 이용한 Conv 연산 가속
- 메모리 접근 패턴 최적화
- Winograd 알고리즘 (3×3 Conv)
