# YOLOv5 C 포팅 프로젝트

YOLOv5s 모델을 Python/PyTorch에서 순수 C로 포팅하여 임베디드/엣지 디바이스에서 실행 가능하도록 구현하는 프로젝트입니다.

## 주요 특징

- ✅ **완전한 YOLOv5s 구현**: Backbone, Head, Detect 헤드 포함
- ✅ **동적 입력 크기 지원**: 640×640 외 다양한 입력 크기 처리
- ✅ **Cross-platform**: Windows/MSVC 및 Linux/GCC 지원
- ✅ **메모리 효율적**: Arena allocator 및 ping-pong 버퍼 사용
- ✅ **End-to-end 파이프라인**: 이미지 입력부터 검출 결과 출력까지

## 프로젝트 구조

```
yolov5_c/
├── third_party/          # YOLOv5 원본 (git submodule)
├── tools/                # Python 기반 export/검증 도구
│   ├── preprocess.py     # 이미지 전처리
│   ├── export_yolov5s.py # 가중치 export
│   ├── dump_golden.py    # 골든 텐서 덤프
│   └── compare_tensors.py # 텐서 비교
├── src/
│   ├── core/            # 텐서/메모리 관리
│   ├── ops/             # primitive 연산 (Conv, BN, SiLU, Pool, Upsample, Concat)
│   ├── blocks/          # C3, Bottleneck, SPPF 블록
│   ├── models/          # YOLOv5s 모델 빌드/인퍼런스
│   └── postprocess/     # Detect decode + NMS
├── tests/               # 단위 테스트
├── data/                # 입력/골든/출력 텐서
│   ├── images/          # 원본 이미지
│   ├── inputs/          # 전처리된 텐서
│   └── outputs/         # 검출 결과
├── weights/             # 모델 가중치
└── docs/                # 문서
```

## 빌드 방법

### Linux/macOS

```bash
mkdir build
cd build
cmake ..
make -j4
```

### Windows (Visual Studio)

```cmd
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release
```

실행 파일은 `build/Release/yolov5_infer.exe`에 생성됩니다.

## 사용 방법

### 1. 이미지 전처리

`data/images/` 폴더에 이미지를 넣고 전처리를 수행합니다:

```bash
# 단일 이미지 처리
python tools/preprocess.py --image bus.jpg

# 디렉토리 내 모든 이미지 처리
python tools/preprocess.py

# 커스텀 이미지 크기
python tools/preprocess.py --size 320
```

전처리된 텐서는 `data/inputs/`에 저장됩니다:
- `{image_name}.bin`: 바이너리 텐서 파일
- `{image_name}_meta.txt`: 메타데이터 파일

### 2. 모델 가중치 Export

PyTorch 모델 파일(`.pt`)을 C에서 사용할 수 있는 형식으로 변환:

```bash
python tools/export_yolov5s.py path/to/yolov5s.pt --output weights/
```

출력 파일:
- `weights/weights.bin`: 바이너리 가중치 파일
- `weights/weights_map.json`: 가중치 매핑 정보
- `weights/model_meta.json`: 모델 메타데이터

### 3. 인퍼런스 실행

```bash
# Linux/macOS
./build/yolov5_infer bus

# Windows
build\Release\yolov5_infer.exe bus
```

인수는 이미지 이름만 지정하면 됩니다 (확장자 제외). 프로그램이 자동으로 다음 경로를 검색합니다:
- `data/inputs/{image_name}.bin`
- `weights/weights.bin`
- `weights/model_meta.json`

검출 결과는 `data/outputs/{image_name}_detections.txt`에 저장됩니다.

### 4. 통합 테스트 실행

```bash
cd build
ctest --output-on-failure
```

또는 개별 테스트 실행:

```bash
./test_conv1x1
./test_integration
```

## 출력 형식

검출 결과는 `data/outputs/{image_name}_detections.txt`에 저장됩니다:

```
Total detections: N

Detection 1:
  Class ID: 0
  Confidence: 0.8523
  BBox: (0.1234, 0.5678, 0.2345, 0.3456)  # normalized [0-1]
  Pixel coords: x=79.0, y=363.4, w=150.1, h=221.2  # pixel coordinates

...

# 파일 끝에 요약 정보
class_id confidence x y w h (normalized)
class_id confidence x_pixel y_pixel w_pixel h_pixel
0 0.8523 0.1234 0.5678 0.2345 0.3456
0 0.8523 79.0 363.4 150.1 221.2
...
```

## 개발 단계

- [x] Stage A: Conv(1×1) 구현 및 검증
- [x] Stage B: Conv(3×3) + padding/stride 검증
- [x] Stage C: SiLU 활성화 함수
- [x] Stage D: Neck (Upsample/Concat)
- [x] Stage E: Detect 헤드 + decode + NMS
- [x] 모델 빌드 및 인퍼런스 파이프라인
- [x] 동적 입력 크기 지원
- [x] 통합 테스트 및 End-to-end 검증

## 요구사항

### 빌드 요구사항
- CMake 3.10 이상
- C 컴파일러 (GCC, Clang, 또는 MSVC)
- Python 3.6 이상 (도구 사용 시)

### Python 패키지 (도구 사용 시)
```bash
pip install torch torchvision opencv-python numpy
```

### 모델 파일
- YOLOv5s 모델 파일 (`.pt`): [ultralytics/yolov5](https://github.com/ultralytics/yolov5)에서 다운로드

## 참고 문서

- `PROJECT_STATUS.md`: 현재 프로젝트 상태 및 완료된 작업
- `TESTING.md`: 테스트 가이드 및 검증 방법
- `docs/MODULE_ARCHITECTURE.md`: 모듈 아키텍처 상세 설명
- `docs/INFERENCE_FLOW.md`: 인퍼런스 파이프라인 설명
- `docs/DETECTION_FLOW.md`: 검출 파이프라인 설명
- `docs/PREPROCESSING.md`: 이미지 전처리 과정 설명

## 라이선스

이 프로젝트는 YOLOv5의 C 포팅 구현입니다. YOLOv5 원본 라이선스를 따릅니다.