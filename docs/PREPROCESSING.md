# 이미지 전처리 과정 (Image Preprocessing)

이 문서는 YOLOv5s C 구현을 위한 이미지 전처리 과정을 설명합니다.

## 개요

YOLOv5s 모델은 고정된 입력 크기(기본값: 640×640)를 요구합니다. 다양한 크기의 입력 이미지를 모델에 맞게 전처리하는 과정을 설명합니다.

## 전처리 파이프라인

전처리는 다음 단계로 구성됩니다:

```
원본 이미지 → BGR→RGB 변환 → Letterbox Resize → 정규화 → NCHW 변환 → 저장
```

## 단계별 상세 설명

### 1. 이미지 로드

**입력**: 이미지 파일 경로 (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`)

```python
img = cv2.imread(image_path)  # BGR 형식으로 로드
original_h, original_w = img.shape[:2]  # 원본 크기 저장
```

- OpenCV는 이미지를 **BGR** 형식으로 로드합니다.
- 원본 이미지 크기(`original_h`, `original_w`)는 나중에 바운딩 박스 좌표를 원본 이미지로 변환할 때 사용됩니다.

### 2. BGR → RGB 변환

```python
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

- YOLOv5는 **RGB** 형식을 사용하므로 BGR을 RGB로 변환합니다.
- 채널 순서: `[B, G, R]` → `[R, G, B]`

### 3. Letterbox Resize

Letterbox resize는 **종횡비를 유지**하면서 이미지를 목표 크기로 조정하고, 부족한 부분을 패딩으로 채웁니다.

#### 3.1 스케일 비율 계산

```python
r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
```

- `new_shape`: 목표 크기 (예: 640×640)
- `shape`: 원본 이미지 크기 (예: 1080×810)
- **최소 비율**을 사용하여 이미지가 잘리지 않도록 합니다.

**예시**:
- 원본: 1080×810
- 목표: 640×640
- `r = min(640/1080, 640/810) = min(0.593, 0.790) = 0.593`

#### 3.2 리사이즈된 크기 계산

```python
new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
```

**예시**:
- `new_unpad = (int(round(810 * 0.593)), int(round(1080 * 0.593)))`
- `new_unpad = (480, 640)`

#### 3.3 패딩 계산

```python
dw = new_shape[1] - new_unpad[0]  # width padding
dh = new_shape[0] - new_unpad[1]  # height padding
dw /= 2  # 양쪽으로 균등 분배
dh /= 2
```

**예시**:
- `dw = 640 - 480 = 160` → `dw = 80` (좌우 각각 80픽셀)
- `dh = 640 - 640 = 0` → `dh = 0` (상하 패딩 없음)

#### 3.4 이미지 리사이즈 및 패딩

```python
img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
img = cv2.copyMakeBorder(img, top, bottom, left, right, 
                         cv2.BORDER_CONSTANT, value=(114, 114, 114))
```

- **리사이즈**: `cv2.INTER_LINEAR` 보간법 사용
- **패딩**: 회색(`114, 114, 114`)으로 채움 (ImageNet 평균값)

**결과**: 640×640 크기의 이미지 (종횡비 유지, 패딩 추가)

### 4. 정규화 (Normalization)

```python
img = img.astype(np.float32) / 255.0
```

- 픽셀 값을 `[0, 255]` → `[0.0, 1.0]` 범위로 정규화
- 데이터 타입: `uint8` → `float32`

### 5. NCHW 형식 변환

```python
# (H, W, C) → (C, H, W)
img = np.transpose(img, (2, 0, 1))
# 배치 차원 추가: (C, H, W) → (1, C, H, W)
img = np.expand_dims(img, axis=0)
```

- **NCHW**: Batch, Channels, Height, Width
- 최종 형태: `(1, 3, 640, 640)`

### 6. 저장

#### 6.1 바이너리 텐서 파일 (`.bin`)

```python
# 형식: [n, c, h, w] (int32_t) + data (float32)
dims = np.array(tensor.shape, dtype=np.int32)
dims.tofile(f)  # 차원 정보 저장
tensor.astype(np.float32).tofile(f)  # 데이터 저장
```

**파일 구조**:
```
[4 bytes: n (int32)]
[4 bytes: c (int32)]
[4 bytes: h (int32)]
[4 bytes: w (int32)]
[데이터: n*c*h*w * 4 bytes (float32)]
```

#### 6.2 메타데이터 텍스트 파일 (`_meta.txt`)

메타데이터는 다음 정보를 포함합니다:

- **Tensor shape**: `(1, 3, 640, 640)`
- **Image size**: 목표 크기 (640)
- **Scale ratio**: 리사이즈 비율 `(r, r)`
- **Padding**: 패딩 값 `(dw, dh)`
- **Format**: NCHW
- **Data type**: float32
- **Normalized**: [0.0, 1.0]

## 사용 방법

### 단일 이미지 처리

```bash
python tools/preprocess.py --image bus.jpg
```

### 디렉토리 내 모든 이미지 처리

```bash
python tools/preprocess.py
```

### 커스텀 입력/출력 디렉토리

```bash
python tools/preprocess.py --input-dir data/images --output-dir data/inputs
```

### 커스텀 이미지 크기

```bash
python tools/preprocess.py --size 320
```

## 출력 파일

전처리 후 다음 파일들이 생성됩니다:

```
data/inputs/
├── bus.bin          # 바이너리 텐서 파일
└── bus_meta.txt     # 메타데이터 텍스트 파일
```

## 메타데이터 예시

`bus_meta.txt` 파일 내용:

```
Image: bus.jpg
Tensor shape: [1, 3, 640, 640]
Image size: 640
Scale ratio: [0.592593, 0.592593]
Padding: (dw=80.00, dh=0.00)

Format: NCHW (Batch, Channels, Height, Width)
Data type: float32
Normalized: [0.0, 1.0]
```

## 바운딩 박스 좌표 변환

전처리 과정에서 저장된 메타데이터를 사용하여 검출된 바운딩 박스 좌표를 원본 이미지 크기로 변환할 수 있습니다:

```python
# 검출 좌표 (640×640 기준)
x, y, w, h = detection_coords

# 원본 이미지로 변환
scale = metadata['ratio'][0]  # 또는 ratio[1]
pad_w, pad_h = metadata['padding']

# 패딩 제거 및 스케일 복원
x_orig = (x - pad_w) / scale
y_orig = (y - pad_h) / scale
w_orig = w / scale
h_orig = h / scale
```

## 주의사항

1. **종횡비 유지**: Letterbox resize는 이미지의 종횡비를 유지하므로 왜곡이 발생하지 않습니다.
2. **패딩 색상**: 패딩은 회색(`114, 114, 114`)으로 채워지며, 이는 ImageNet 데이터셋의 평균 픽셀 값입니다.
3. **정규화 범위**: 픽셀 값은 `[0.0, 1.0]` 범위로 정규화됩니다.
4. **메모리 레이아웃**: 텐서는 **NCHW** 형식으로 저장되며, C 코드에서 직접 로드할 수 있습니다.

## 참고

- YOLOv5 원본 구현: [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- OpenCV 문서: [cv2.resize](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231aec9a3dfb0)
