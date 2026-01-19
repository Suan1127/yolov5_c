# YOLOv5 C 포팅 프로젝트

YOLOv5s 모델을 Python/PyTorch에서 순수 C로 포팅하여 임베디드/엣지 디바이스에서 실행 가능하도록 구현하는 프로젝트입니다.

## 프로젝트 구조

```
yolov5_c/
├── third_party/          # YOLOv5 원본 (git submodule)
├── tools/                # Python 기반 export/검증 도구
├── src/
│   ├── core/            # 텐서/메모리 관리
│   ├── ops/             # primitive 연산
│   ├── blocks/          # C3, Bottleneck, SPPF 블록
│   ├── models/          # YOLOv5s 모델 빌드/인퍼런스
│   └── postprocess/     # Detect decode + NMS
├── tests/               # 단위 테스트
├── data/                # 입력/골든/출력 텐서
└── weights/             # .pt 및 변환된 .bin 파일
```

## 빌드 방법

```bash
mkdir build
cd build
cmake ..
make
```

Windows (Visual Studio):
```cmd
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019"
cmake --build . --config Release
```

## 사용 방법

### 1. 이미지 전처리
```bash
python tools/preprocess.py input.jpg --output data/input_tensor.bin
```

### 2. 모델 가중치 export
```bash
python tools/export_yolov5s.py yolov5s.pt --output weights/
```

### 3. 인퍼런스 실행
```bash
./yolov5_infer data/input_tensor.bin weights/weights.bin weights/model_meta.json
```

### 4. 통합 테스트 실행
```bash
cd build
./test_integration
```

## 개발 단계

- [x] Stage A: Conv(1×1) 구현 및 검증
- [x] Stage B: Conv(3×3) + padding/stride 검증
- [x] Stage C: SiLU 활성화 함수
- [x] Stage D: Neck (Upsample/Concat)
- [x] Stage E: Detect 헤드 + decode + NMS
- [x] 모델 빌드 및 인퍼런스 파이프라인
- [x] 통합 테스트

## 출력 형식

인퍼런스 결과는 `data/output_detections.txt`에 저장됩니다:
```
Total detections: N
Format: class_id confidence x y w h (normalized 0-1)
Format: class_id confidence x_pixel y_pixel w_pixel h_pixel

0 0.8523 0.1234 0.5678 0.2345 0.3456
0 0.8523 79.0 363.4 150.1 221.2
...
```

## 참고 문서

- `PROJECT_BRIEF.md`: 프로젝트 전체 개요 및 요구사항
- `YOLOV5S_ARCHITECTURE.md`: 모델 아키텍처 상세 설계
- `PROJECT_STATUS.md`: 현재 프로젝트 상태