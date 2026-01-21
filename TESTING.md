# 테스트 가이드

## YOLOv5n 모델 테스트

이 문서는 YOLOv5n 모델의 테스트 및 검증 방법을 설명합니다.

## 1. 사전 준비

### 1.1 모델 가중치 준비

YOLOv5n 가중치는 이미 `weights/yolov5n/` 디렉토리에 준비되어 있습니다:
- `weights_yolov5n.bin`: 바이너리 가중치 파일
- `weights_map_yolov5n.json`: 가중치 맵 파일
- `model_meta_yolov5n.json`: 모델 메타데이터

### 1.2 입력 데이터 준비

```bash
# 이미지 전처리 (YOLOv5n용)
python tools/preprocess.py --image bus.jpg --output data/yolov5n/inputs/
```

또는 기존 입력 파일 사용:
- `data/yolov5n/inputs/bus.bin`

### 1.3 빌드

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release  # Windows
# 또는
make  # Linux/macOS
cd ..
```

## 2. 중간 텐서 비교 테스트

### 2.1 Python Golden 데이터 생성

```bash
# 모든 레이어 (0-23) 저장
python tools/dump_golden.py yolov5n.pt bus --output testdata_n/python

# 또는 특정 레이어만 저장
python tools/dump_golden.py yolov5n.pt bus --output testdata_n/python
```

**출력 위치**: `testdata_n/python/`
- `input.bin`: 입력 텐서
- `layer_000.bin` ~ `layer_023.bin`: 각 레이어 출력
- `output_0.bin`: Detect head 출력
- `golden_meta.json`: 메타데이터

### 2.2 C 구현 실행

```bash
# Windows
.\build\Release\yolov5_infer.exe bus

# Linux/macOS
./build/yolov5_infer bus
```

**출력 위치**: `testdata_n/c/`
- `input.bin`: 입력 텐서
- `layer_000.bin` ~ `layer_023.bin`: 각 레이어 출력
- `output_p3.bin`, `output_p4.bin`, `output_p5.bin`: Detect head 입력 feature maps

### 2.3 텐서 비교

```bash
# 전체 디렉토리 비교
python tools/compare_tensors.py testdata_n/python testdata_n/c

# 특정 레이어만 비교
python tools/compare_tensors.py testdata_n/python/layer_000.bin testdata_n/c/layer_000.bin
```

**비교 결과 예시**:
```
Comparing layer_000.bin...
  Shape: [  1  16 320 320]
  Max diff: 1.811981e-05
  Mean diff: 1.023055e-06
  RMSE: 1.591757e-06
  Within tolerance (0.0001): OK
```

**기대 결과**: 모든 레이어가 tolerance (0.0001) 내에서 일치해야 합니다.

## 3. 전체 테스트 워크플로우

```bash
# 1. Python golden 생성
python tools/dump_golden.py yolov5n.pt bus --output testdata_n/python


# 2. C 구현 실행
.\build\Release\yolov5_infer.exe bus

# 3. 비교
python tools/compare_tensors.py testdata_n/python testdata_n/c
```

## 4. 레이어별 채널 수 (YOLOv5n)

### Backbone
- Layer 0: Conv(3→16)
- Layer 1: Conv(16→32)
- Layer 2: C3(32→32)
- Layer 3: Conv(32→64)
- Layer 4: C3(64→64)
- Layer 5: Conv(64→128)
- Layer 6: C3(128→128)
- Layer 7: Conv(128→256)
- Layer 8: C3(256→256)
- Layer 9: SPPF(256→256)

### Head
- Layer 10: Conv(256→128)
- Layer 11: Upsample(128)
- Layer 12: Concat([11, 6]) = 128 + 128 = 256
- Layer 13: C3(256→128)
- Layer 14: Conv(128→64)
- Layer 15: Upsample(64)
- Layer 16: Concat([15, 4]) = 64 + 64 = 128
- Layer 17: C3(128→64) - **P3**
- Layer 18: Conv(64→64)
- Layer 19: Concat([18, 14]) = 64 + 64 = 128
- Layer 20: C3(128→128) - **P4**
- Layer 21: Conv(128→128)
- Layer 22: Concat([21, 10]) = 128 + 128 = 256
- Layer 23: C3(256→256) - **P5**

## 5. 테스트 체크리스트

### 기본 기능 테스트
- [x] 모델 빌드 성공
- [x] 가중치 로드 성공 (weights/yolov5n/)
- [x] Forward pass 실행 성공
- [x] 출력 텐서 shape 확인
- [x] Saved features 확인

### 정확도 테스트
- [x] 골든 참조와 텐서 비교 (24/24 레이어 통과)
- [x] 레이어별 출력 검증
- [ ] 최종 bbox 결과 비교 (Detect head 완성 후)

### 성능 테스트
- [ ] 인퍼런스 시간 측정
- [ ] 메모리 사용량 측정

## 6. 검증된 레이어

다음 레이어들이 Python과 C 구현 간 일치함을 확인했습니다:

✅ **Backbone (0-9)**: 모든 레이어 통과
- Layer 0-7: Conv 및 C3 블록
- Layer 8: C3 블록
- Layer 9: SPPF 블록

✅ **Head (10-23)**: 모든 레이어 통과
- Layer 10-14: Conv 및 Upsample
- Layer 15: Upsample (SKIP, 가중치 없음)
- Layer 16-23: Concat 및 C3 블록

**최종 결과**: 24/24 레이어 비교 통과 ✅

## 7. 문제 해결

### 모델 빌드 실패
- `weights/yolov5n/weights_yolov5n.bin` 파일 확인
- `weights/yolov5n/weights_map_yolov5n.json` 파일 확인
- `weights/yolov5n/model_meta_yolov5n.json` 파일 확인

### Forward pass 실패
- 입력 텐서 shape 확인 (1, 3, 640, 640)
- 메모리 부족 확인
- 채널 수 불일치 확인 (YOLOv5n은 YOLOv5s의 절반)

### 텐서 비교 실패
- 입력 텐서가 동일한지 확인 (`input.bin` 비교)
- 레이어별 채널 수 확인 (YOLOv5n: 16, 32, 64, 128, 256)
- Tolerance 설정 확인 (기본값: 0.0001)

### 출력이 모두 0
- 가중치 로드 확인
- 레이어 실행 순서 확인
- Fused Batch Normalization 처리 확인

## 8. 참고 사항

- **모델 타입**: YOLOv5n (width_multiple=0.25, depth_multiple=0.33)
- **입력 크기**: 640×640
- **출력 채널**: YOLOv5s의 절반 (16, 32, 64, 128, 256)
- **테스트 데이터**: `testdata_n/` 디렉토리 사용
- **결과 저장**: `data/yolov5n/outputs/` 디렉토리
