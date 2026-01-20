# 레이어 구조 비교 문서

이 문서는 YOLOv5s의 레이어 구조와 Python/C 구현에서의 저장 방식을 비교 분석합니다.

## YOLOv5s 모델 구조

YOLOv5s 모델 구조는 `third_party/yolov5/models/yolov5s.yaml`에 정의되어 있습니다.

### 전체 레이어 구조 (25개 레이어)

#### Backbone (0-9)

| Layer | Type | Description | Input | Output Channels | Save |
|-------|------|-------------|-------|----------------|------|
| 0 | Conv | Conv(3→32, 6×6, s=2, p=2) | input | 32 | ✅ |
| 1 | Conv | Conv(32→64, 3×3, s=2) | 0 | 64 | ✅ |
| 2 | C3 | C3(64→64, n=1) | 1 | 64 | ✅ |
| 3 | Conv | Conv(64→128, 3×3, s=2) | 2 | 128 | ✅ |
| 4 | C3 | C3(128→128, n=2) | 3 | 128 | ✅ |
| 5 | Conv | Conv(128→256, 3×3, s=2) | 4 | 256 | ✅ |
| 6 | C3 | C3(256→256, n=3) | 5 | 256 | ✅ |
| 7 | Conv | Conv(256→512, 3×3, s=2) | 6 | 512 | ✅ |
| 8 | C3 | C3(512→512, n=1) | 7 | 512 | ❌ |
| 9 | SPPF | SPPF(512→512, k=5) | 8 | 512 | ✅ |

#### Head (10-24)

| Layer | Type | Description | Input | Output Channels | Save |
|-------|------|-------------|-------|----------------|------|
| 10 | Conv | Conv(512→256, 1×1) | 9 | 256 | ❌ |
| 11 | Upsample | Upsample(×2) | 10 | 256 | ❌ |
| 12 | Concat | Concat([11, 6]) | 11, 6 | 512 | ❌ |
| 13 | C3 | C3(512→256, n=1, shortcut=False) | 12 | 256 | ❌ |
| 14 | Conv | Conv(256→128, 1×1) | 13 | 128 | ❌ |
| 15 | Upsample | Upsample(×2) | 14 | 128 | ❌ |
| 16 | Concat | Concat([15, 4]) | 15, 4 | 256 | ❌ |
| 17 | C3 | C3(256→128, n=1, shortcut=False) | 16 | 128 | ✅ (P3) |
| 18 | Conv | Conv(128→128, 3×3, s=2) | 17 | 128 | ❌ |
| 19 | Concat | Concat([18, 13]) | 18, 13 | 256 | ❌ |
| 20 | C3 | C3(256→256, n=1, shortcut=False) | 19 | 256 | ✅ (P4) |
| 21 | Conv | Conv(256→256, 3×3, s=2) | 20 | 256 | ❌ |
| 22 | Concat | Concat([21, 10]) | 21, 10 | 512 | ❌ |
| 23 | C3 | C3(512→512, n=1, shortcut=False) | 22 | 512 | ✅ (P5) |
| 24 | Detect | Detect([17, 20, 23]) | 17, 20, 23 | - | ❌ |

**참고**: YAML 파일의 주석에 표시된 레이어 번호는 실제 모델 인덱스와 일치합니다.

## 저장되는 레이어

### 저장 레이어 목록

Python과 C 모두 동일한 레이어를 저장합니다:

```python
save_layers = [0, 1, 2, 3, 4, 5, 6, 7, 9, 17, 20, 23]
```

**총 12개 레이어**가 저장됩니다.

### 저장되는 레이어의 의미

1. **Backbone 주요 레이어** (0-9):
   - `0, 1`: 초기 Conv 레이어들
   - `2, 4, 6`: C3 블록들 (feature extraction)
   - `3, 5, 7`: Downsampling Conv 레이어들
   - `9`: SPPF (Spatial Pyramid Pooling Fast) - backbone의 마지막

2. **Head의 Detect 입력 레이어** (17, 20, 23):
   - `17`: P3 경로 (small objects, 80×80)
   - `20`: P4 경로 (medium objects, 40×40)
   - `23`: P5 경로 (large objects, 20×20)

### 저장되지 않는 레이어

다음 레이어들은 저장되지 않습니다:

- **Layer 8**: C3 블록 (backbone 내부, 중간 단계)
- **Layer 10-16**: Head의 중간 처리 레이어들 (Upsample, Concat, C3 등)
- **Layer 18-22**: Head의 중간 처리 레이어들
- **Layer 24**: Detect 레이어 (최종 출력은 별도로 저장)

## Python 구현 저장 방식

### 파일: `tools/dump_golden.py`

#### 저장 프로세스

1. **Forward Hook 등록**:
   ```python
   for layer_idx in save_layers:
       module = model.model[layer_idx]  # 직접 인덱스 접근
       handle = module.register_forward_hook(make_hook(layer_idx))
   ```

2. **Forward Pass 실행**:
   ```python
   output = model(input_tensor)  # Hook이 자동으로 출력 캡처
   ```

3. **파일 저장**:
   ```python
   for layer_idx in save_layers:
       output_path = output_dir / f"layer_{layer_idx:03d}.bin"
       save_tensor_bin(tensor, output_path)
   ```

#### 저장 위치

- **디렉토리**: `testdata/python/`
- **파일명 형식**: `layer_{layer_idx:03d}.bin`
- **예시**: `layer_000.bin`, `layer_001.bin`, `layer_017.bin`

#### 레이어 접근 방법

Python은 PyTorch 모델의 `model.model` Sequential 컨테이너를 사용하여 직접 인덱스로 접근합니다:

```python
model.model[0]  # Layer 0
model.model[1]  # Layer 1
model.model[17] # Layer 17
```

이는 YOLOv5s.yaml의 레이어 순서와 정확히 일치합니다.

## C 구현 저장 방식

### 파일: `src/models/yolov5s_infer.c`

#### 레이어 구조 정의

C 구현은 `src/models/yolov5s_graph.c`에 레이어 구조를 명시적으로 정의합니다:

```c
const layer_def_t yolov5s_graph[YOLOV5S_NUM_LAYERS] = {
    // Backbone
    {0, LAYER_CONV, {1, {0, 0, 0, 0}}, 0},      // 0: Conv
    {1, LAYER_CONV, {1, {0, 0, 0, 0}}, 0},      // 1: Conv
    {2, LAYER_C3, {1, {1, 0, 0, 0}}, 0},        // 2: C3
    {3, LAYER_CONV, {1, {2, 0, 0, 0}}, 1},      // 3: Conv → save
    // ... (총 25개 레이어)
};
```

이 구조는 YOLOv5s.yaml과 정확히 일치합니다.

#### 저장 프로세스

1. **Forward Pass 중 저장**:
   ```c
   // 각 레이어 실행 후
   save_feature(model, layer_idx, feature_tensor);
   ```

2. **save_feature 함수**:
   ```c
   static void save_feature(yolov5s_model_t* model, int32_t layer_idx, tensor_t* feature) {
       int32_t save_map[] = {0, 1, 2, 3, 4, 5, 6, 7, 9, 17, 20, 23};
       // save_map에 포함된 레이어만 저장
       if (save_idx >= 0) {
           snprintf(filepath, sizeof(filepath), "%s/layer_%03d.bin", g_output_dir, layer_idx);
           tensor_dump(feature, filepath);
       }
   }
   ```

#### 저장 위치

- **디렉토리**: `testdata/c/` (레이어 출력), `debug/c/` (디버그 출력)
- **파일명 형식**: `layer_{layer_idx:03d}.bin`
- **예시**: `layer_000.bin`, `layer_001.bin`, `layer_017.bin`

#### 레이어 실행 순서

C 구현은 `yolov5s_forward` 함수에서 레이어를 순차적으로 실행합니다:

```c
// Layer 0
conv2d_forward(...);
save_feature(model, 0, buf_a);

// Layer 1
conv2d_forward(...);
save_feature(model, 1, buf_b);

// Layer 2
c3_forward(...);
save_feature(model, 2, buf_b);

// ... (계속)
```

## 레이어 구조 일치 확인

### ✅ Python과 C의 일치성

1. **레이어 인덱스**: 완전히 일치
   - Python: `model.model[layer_idx]`
   - C: `yolov5s_graph[layer_idx]`

2. **저장 레이어 목록**: 완전히 일치
   - Python: `[0, 1, 2, 3, 4, 5, 6, 7, 9, 17, 20, 23]`
   - C: `[0, 1, 2, 3, 4, 5, 6, 7, 9, 17, 20, 23]`

3. **파일명 형식**: 완전히 일치
   - Python: `layer_{layer_idx:03d}.bin`
   - C: `layer_{layer_idx:03d}.bin`

4. **YOLOv5s.yaml 구조**: 완전히 일치
   - 모든 레이어 인덱스와 타입이 일치

### 레이어 매핑 확인

| YAML | Python | C | Description |
|------|--------|---|-------------|
| 0 | ✅ | ✅ | Conv(3→32, 6×6, s=2) |
| 1 | ✅ | ✅ | Conv(32→64, 3×3, s=2) |
| 2 | ✅ | ✅ | C3(64→64, n=1) |
| 3 | ✅ | ✅ | Conv(64→128, 3×3, s=2) |
| 4 | ✅ | ✅ | C3(128→128, n=2) |
| 5 | ✅ | ✅ | Conv(128→256, 3×3, s=2) |
| 6 | ✅ | ✅ | C3(256→256, n=3) |
| 7 | ✅ | ✅ | Conv(256→512, 3×3, s=2) |
| 8 | ❌ | ❌ | C3(512→512, n=1) - 저장 안 함 |
| 9 | ✅ | ✅ | SPPF(512→512, k=5) |
| 10-16 | ❌ | ❌ | Head 중간 레이어 - 저장 안 함 |
| 17 | ✅ | ✅ | C3(256→128) - P3 |
| 18-19 | ❌ | ❌ | Head 중간 레이어 - 저장 안 함 |
| 20 | ✅ | ✅ | C3(256→256) - P4 |
| 21-22 | ❌ | ❌ | Head 중간 레이어 - 저장 안 함 |
| 23 | ✅ | ✅ | C3(512→512) - P5 |
| 24 | ❌ | ❌ | Detect - 별도 처리 |

## 파일 저장 비교

### Python (`testdata/python/`)

```
testdata/python/
├── input.bin              # 입력 텐서
├── layer_000.bin          # Layer 0 출력
├── layer_001.bin          # Layer 1 출력
├── layer_002.bin          # Layer 2 출력
├── layer_003.bin          # Layer 3 출력
├── layer_004.bin          # Layer 4 출력
├── layer_005.bin          # Layer 5 출력
├── layer_006.bin          # Layer 6 출력
├── layer_007.bin          # Layer 7 출력
├── layer_009.bin          # Layer 9 출력
├── layer_017.bin          # Layer 17 출력 (P3)
├── layer_020.bin          # Layer 20 출력 (P4)
├── layer_023.bin          # Layer 23 출력 (P5)
├── output_0.bin           # Detect head 출력
└── golden_meta.json       # 메타데이터
```

### C (`testdata/c/`)

```
testdata/c/
├── input.bin              # 입력 텐서
├── layer_000.bin          # Layer 0 출력
├── layer_001.bin          # Layer 1 출력
├── layer_002.bin          # Layer 2 출력
├── layer_003.bin          # Layer 3 출력
├── layer_004.bin          # Layer 4 출력
├── layer_005.bin          # Layer 5 출력
├── layer_006.bin          # Layer 6 출력
├── layer_007.bin          # Layer 7 출력
├── layer_009.bin          # Layer 9 출력
├── layer_017.bin          # Layer 17 출력 (P3)
├── layer_020.bin          # Layer 20 출력 (P4)
├── layer_023.bin          # Layer 23 출력 (P5)
├── output_p3.bin          # P3 feature map
├── output_p4.bin          # P4 feature map
└── output_p5.bin          # P5 feature map
```

**참고**: C 구현은 Detect head의 출력을 `output_p3.bin`, `output_p4.bin`, `output_p5.bin`으로 저장합니다.

## 결론

### ✅ 구조 일치 확인

1. **YOLOv5s.yaml 구조**: Python과 C 모두 동일한 구조를 따릅니다.
2. **레이어 인덱스**: 완전히 일치합니다 (0-24).
3. **저장 레이어**: 동일한 12개 레이어를 저장합니다.
4. **파일명 형식**: 동일한 형식을 사용합니다.

### 비교 방법

Python과 C의 출력을 비교하려면:

```bash
# 레이어별 비교
python tools/compare_tensors.py testdata/python/layer_000.bin testdata/c/layer_000.bin

# 전체 디렉토리 비교
python tools/compare_tensors.py testdata/python testdata/c
```

## 관련 문서

- `docs/GOLDEN_TENSOR_STORAGE.md`: Golden 텐서 저장 방식
- `docs/DEBUG_FILES.md`: 디버그 파일 가이드
- `src/models/yolov5s_graph.c`: C 레이어 구조 정의
- `third_party/yolov5/models/yolov5s.yaml`: YOLOv5s 모델 정의
