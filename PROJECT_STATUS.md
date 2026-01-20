# YOLOv5 C 포팅 프로젝트 상태

## 완료된 작업 ✅

### 1. 프로젝트 구조 생성 ✅
- 디렉토리 구조 생성 완료
- CMakeLists.txt 설정 완료 (Windows/MSVC 지원 포함)
- .gitignore 파일 생성

### 2. C 코어 구현 ✅
- **텐서 구조체** (`src/core/tensor.h/c`)
  - NCHW 레이아웃 지원
  - 바이너리 파일 저장/로드 기능
- **메모리 관리** (`src/core/memory.h/c`)
  - Arena allocator 구현
  - 16-byte 정렬 지원 (SIMD 최적화)
  - Cross-platform 호환성 (Windows/MSVC 지원)
- **가중치 로더** (`src/core/weights_loader.h/c`)
  - 바이너리 가중치 파일 로드
  - JSON 메타데이터 파싱

### 3. Primitive 연산 구현 ✅
- **Conv2D** (`src/ops/conv2d.h/c`)
  - 1×1, 3×3, 6×6 convolution
  - Padding, stride, dilation 지원
  - 동적 출력 크기 계산
- **BatchNorm2D** (`src/ops/batchnorm2d.h/c`)
  - 학습된 파라미터 적용
- **Activation** (`src/ops/activation.h/c`)
  - SiLU 활성화 함수 (x * sigmoid(x))
- **MaxPool2D** (`src/ops/pooling.h/c`)
  - SPPF 블록용 5×5 MaxPool
- **Upsample** (`src/ops/upsample.h/c`)
  - Nearest-neighbor ×2 업샘플링
- **Concat** (`src/ops/concat.h/c`)
  - 채널 차원 기준 텐서 결합

### 4. 블록 구현 ✅
- **Bottleneck** (`src/blocks/bottleneck.h/c`)
  - C3 블록 내부 구성 요소
- **C3** (`src/blocks/c3.h/c`)
  - Cross-stage partial bottleneck
  - Shortcut 연결 지원
- **SPPF** (`src/blocks/sppf.h/c`)
  - Spatial Pyramid Pooling Fast
  - 5×5 MaxPool 반복 적용

### 5. 모델 구조 ✅
- **모델 그래프 정의** (`src/models/yolov5s_graph.h/c`)
  - 25개 레이어 정의 (Backbone + Head)
  - Feature map 저장 지점 지정
- **모델 빌드 함수** (`src/models/yolov5s_build.h/c`)
  - 모든 레이어 초기화 완료
  - 가중치 로드 완료 (Conv, BN, C3, SPPF)
  - 모델 파라미터 자동 추출 (depth_multiple, width_multiple, anchors, num_classes)
- **인퍼런스 파이프라인** (`src/models/yolov5s_infer.h/c`)
  - Forward pass 구현 완료
  - 동적 입력 크기 지원 (640×640 외 다양한 크기)
  - Ping-pong 버퍼 메모리 관리
  - Feature map 저장 및 FPN 연결
  - 모든 레이어 디버깅 출력 추가
  - 크기 불일치 처리 (Layer 19 Concat 리사이즈)

### 6. Post-processing ✅
- **Detect 헤드** (`src/postprocess/detect.h/c`)
  - 1×1 Conv 레이어 적용
  - Decode 구현 (grid/anchor 기반 bbox 변환)
  - 동적 feature map 크기 지원
- **NMS** (`src/postprocess/nms.h/c`)
  - Non-Maximum Suppression 구현
  - IoU 계산 및 중복 제거

### 7. Python 도구 ✅
- `preprocess.py`: 이미지 전처리
  - Letterbox resize (종횡비 유지)
  - 정규화 및 NCHW 변환
  - 바이너리 텐서 및 메타데이터 저장
  - 디렉토리 일괄 처리 지원
- `export_yolov5s.py`: 모델 가중치 export
  - third_party/yolov5 사용
  - 바이너리 가중치 파일 생성
  - JSON 메타데이터 생성
- `dump_golden.py`: 골든 참조 텐서 덤프
- `compare_tensors.py`: C 출력 vs Python 골든 비교

### 8. 통합 테스트 ✅
- End-to-end 테스트 완료
  - 이미지 입력 → 전처리 → 인퍼런스 → 검출 결과 출력
  - 바운딩 박스 좌표 변환 (정규화 → 픽셀 좌표)
  - 결과 파일 저장 (`data/outputs/{image_name}_detections.txt`)

### 9. 문서화 ✅
- `docs/MODULE_ARCHITECTURE.md`: 모듈 아키텍처 설명
- `docs/INFERENCE_FLOW.md`: 인퍼런스 파이프라인 설명
- `docs/DETECTION_FLOW.md`: 검출 파이프라인 설명
- `docs/PREPROCESSING.md`: 이미지 전처리 과정 설명

## 진행 중 / TODO

### 1. 정확도 검증
- [ ] Python 골든 참조와 C 출력 비교
- [ ] 레이어별 텐서 검증
- [ ] 최종 검출 결과 정확도 검증

### 2. 최적화
- [ ] 메모리 사용량 최적화
- [ ] 성능 최적화 (SIMD 등)
- [ ] 빌드 최적화 (Release 모드)

### 3. 추가 기능
- [ ] 다양한 입력 크기 테스트
- [ ] 배치 처리 지원
- [ ] GPU/가속기 지원 (선택사항)

## 파일 구조

```
yolov5_c/
├── CMakeLists.txt          # 빌드 설정 ✅
├── README.md               # 프로젝트 개요 ✅
├── PROJECT_STATUS.md       # 이 파일 ✅
├── TESTING.md              # 테스트 가이드 ✅
├── src/
│   ├── core/              # 텐서/메모리/가중치 로더 ✅
│   │   ├── tensor.h/c     # 텐서 구조체 및 유틸리티
│   │   ├── memory.h/c     # Arena allocator
│   │   ├── weights_loader.h/c  # 가중치 로더
│   │   └── common.h       # 공통 매크로 (SNPRINTF 등)
│   ├── ops/               # Primitive 연산 ✅
│   │   ├── conv2d.h/c     # 2D Convolution
│   │   ├── batchnorm2d.h/c # Batch Normalization
│   │   ├── activation.h/c  # SiLU 활성화
│   │   ├── pooling.h/c    # MaxPool2D
│   │   ├── upsample.h/c   # Upsample
│   │   └── concat.h/c     # Concat
│   ├── blocks/            # C3, Bottleneck, SPPF ✅
│   │   ├── bottleneck.h/c
│   │   ├── c3.h/c
│   │   └── sppf.h/c
│   ├── models/            # 모델 빌드/인퍼런스 ✅
│   │   ├── yolov5s_graph.h/c      # 모델 그래프 정의
│   │   ├── yolov5s_build.h/c      # 모델 빌드
│   │   ├── yolov5s_infer.h/c      # 인퍼런스 파이프라인
│   │   └── yolov5s_infer_utils.h  # 유틸리티 매크로
│   ├── postprocess/       # Detect decode + NMS ✅
│   │   ├── detect.h/c     # Detect 헤드 및 decode
│   │   └── nms.h/c        # Non-Maximum Suppression
│   └── main.c             # 메인 실행 파일 ✅
├── tools/                 # Python 도구 ✅
│   ├── preprocess.py      # 이미지 전처리
│   ├── export_yolov5s.py  # 가중치 export
│   ├── dump_golden.py     # 골든 텐서 덤프
│   └── compare_tensors.py # 텐서 비교
├── tests/                 # 단위 테스트
│   ├── CMakeLists.txt
│   ├── test_conv1x1.c
│   └── test_integration.c
├── docs/                  # 문서 ✅
│   ├── MODULE_ARCHITECTURE.md
│   ├── INFERENCE_FLOW.md
│   ├── DETECTION_FLOW.md
│   └── PREPROCESSING.md
├── data/                  # 데이터
│   ├── images/           # 원본 이미지
│   ├── inputs/           # 전처리된 텐서 (.bin, _meta.txt)
│   └── outputs/          # 검출 결과 (.txt)
└── weights/               # 모델 가중치 ✅
    ├── weights.bin       # 바이너리 가중치
    ├── weights_map.json  # 가중치 매핑
    └── model_meta.json   # 모델 메타데이터
```

## 주요 기능

### ✅ 완전히 구현된 기능
1. **동적 입력 크기 지원**: 640×640 외 다양한 입력 크기 처리
2. **전체 모델 파이프라인**: Backbone → Head → Detect → NMS
3. **메모리 효율적 관리**: Ping-pong 버퍼 및 Arena allocator
4. **Cross-platform 지원**: Windows/MSVC 및 Linux/GCC
5. **상세한 디버깅 출력**: 각 레이어별 진행 상황 표시

### 🔧 최근 개선사항
- Layer 19 Concat 크기 불일치 처리 (자동 리사이즈)
- 동적 텐서 크기 계산 (CONV_OUT_D1 매크로)
- 모든 레이어 디버깅 출력 추가
- 전처리 문서화 완료

## 다음 단계

1. **정확도 검증**: Python 골든 참조와 비교
2. **성능 최적화**: SIMD 활용 및 메모리 최적화
3. **추가 테스트**: 다양한 입력 크기 및 이미지 테스트

## 참고 문서

- `README.md`: 프로젝트 개요 및 사용 방법
- `TESTING.md`: 테스트 가이드
- `docs/MODULE_ARCHITECTURE.md`: 모듈 아키텍처 설명
- `docs/INFERENCE_FLOW.md`: 인퍼런스 파이프라인 설명
- `docs/DETECTION_FLOW.md`: 검출 파이프라인 설명
- `docs/PREPROCESSING.md`: 이미지 전처리 과정 설명
