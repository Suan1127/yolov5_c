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
  - 디버깅용 텐서 덤프 기능
- **메모리 관리** (`src/core/memory.h/c`)
  - Arena allocator 구현
  - 16-byte 정렬 지원 (SIMD 최적화)
  - Cross-platform 호환성 (Windows/MSVC 지원)
- **가중치 로더** (`src/core/weights_loader.h/c`)
  - 바이너리 가중치 파일 로드
  - JSON 메타데이터 파싱
  - Fused Batch Normalization 감지

### 3. Primitive 연산 구현 ✅
- **Conv2D** (`src/ops/conv2d.h/c`)
  - 1×1, 3×3, 6×6 convolution
  - Padding, stride, dilation 지원
  - 동적 출력 크기 계산
  - Fused BN 지원 (bias에 BN 파라미터 포함)
- **BatchNorm2D** (`src/ops/batchnorm2d.h/c`)
  - 학습된 파라미터 적용
  - Fused 모드에서는 identity로 설정
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
  - Conv → BN → SiLU → Conv → BN → SiLU
- **C3** (`src/blocks/c3.h/c`)
  - Cross-stage partial bottleneck
  - Shortcut 연결 지원
  - cv1, cv2, cv3 경로 모두 구현
  - **Fused BN 지원**: `cv1_is_fused`, `cv2_is_fused`, `cv3_is_fused` 플래그
  - **cv2에 SiLU activation 추가** (PyTorch Conv 클래스는 기본적으로 SiLU 포함)
  - 디버깅 출력 기능 (`c3_set_debug_dir`)
- **SPPF** (`src/blocks/sppf.h/c`)
  - Spatial Pyramid Pooling Fast
  - **로직 수정 완료**: `y1 = m(x)`, `y2 = m(y1)`, `y4 = m(y2)` (y3 제거)
  - **Concat 순서 수정**: `[x, y1, y2, y4]` (PyTorch와 일치)
  - **Fused BN 지원**: `cv1_is_fused`, `cv2_is_fused` 플래그
  - 디버깅 출력 기능 (`sppf_set_debug_dir`)

### 5. 모델 구조 ✅
- **모델 그래프 정의** (`src/models/yolov5s_graph.h/c`)
  - 25개 레이어 정의 (Backbone + Head)
  - Feature map 저장 지점 지정 (P3, P4, P5)
- **모델 빌드 함수** (`src/models/yolov5s_build.h/c`)
  - 모든 레이어 초기화 완료
  - 가중치 로드 완료 (Conv, BN, C3, SPPF)
  - **Fused BN 감지 및 처리**: `model.X.conv.bias` 존재 여부 확인
  - 모델 파라미터 자동 추출 (depth_multiple, width_multiple, anchors, num_classes)
- **인퍼런스 파이프라인** (`src/models/yolov5s_infer.h/c`)
  - Forward pass 구현 완료
  - 동적 입력 크기 지원 (640×640 외 다양한 크기)
  - Ping-pong 버퍼 메모리 관리
  - Feature map 저장 및 FPN 연결
  - **모든 레이어 출력 저장**: Layer 0-23 모두 `save_feature()` 호출
  - 중간 텐서 저장 기능 (`yolov5s_set_output_dir`)
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
- **`preprocess.py`**: 이미지 전처리
  - Letterbox resize (종횡비 유지)
  - 정규화 및 NCHW 변환
  - 바이너리 텐서 및 메타데이터 저장
  - 디렉토리 일괄 처리 지원
- **`export_yolov5s.py`**: 모델 가중치 export
  - third_party/yolov5 사용
  - 바이너리 가중치 파일 생성
  - JSON 메타데이터 생성
- **`dump_golden.py`**: 골든 참조 텐서 덤프
  - **모든 레이어 출력 저장** (Layer 0-23 기본값)
  - 이미지 이름으로 입력 자동 찾기
  - PyTorch 모델로 forward pass 실행
- **`compare_tensors.py`**: C 출력 vs Python 골든 비교
  - 레이어별 비교 (Max diff, Mean diff, RMSE)
  - Upsample 레이어 자동 SKIP
  - 이미지 파일 자동 SKIP
  - 요약 통계 제공
- **디버깅 도구**:
  - `debug_layer2.py`: C3 블록 중간 출력 생성
  - `debug_layer9.py`: SPPF 블록 중간 출력 생성
  - `compare_c3_steps.py`: C3 블록 단계별 비교
  - `compare_sppf_steps.py`: SPPF 블록 단계별 비교

### 8. 통합 테스트 ✅
- End-to-end 테스트 완료
  - 이미지 입력 → 전처리 → 인퍼런스 → 검출 결과 출력
  - 바운딩 박스 좌표 변환 (정규화 → 픽셀 좌표)
  - 결과 파일 저장 (`data/outputs/{image_name}_detections.txt`)

### 9. 정확도 검증 ✅
- **레이어별 검증 완료**: Layer 0-23 모두 PyTorch와 비교 검증
- **Layer 2 (C3 블록) 수정 완료**:
  - 문제: cv2 경로에 SiLU activation 누락
  - 해결: `activation_silu(skip_output)` 추가
  - 검증: `compare_c3_steps.py`로 모든 단계 일치 확인
- **Layer 9 (SPPF 블록) 수정 완료**:
  - 문제 1: MaxPool 로직 오류 (y3 불필요, concat 순서 잘못)
  - 해결: `y1 = m(x)`, `y2 = m(y1)`, `y4 = m(y2)`, `concat([x, y1, y2, y4])`
  - 문제 2: Fused BN 처리 누락
  - 해결: `cv1_is_fused`, `cv2_is_fused` 플래그 추가
  - 검증: `compare_sppf_steps.py`로 모든 단계 일치 확인
- **Fused Batch Normalization 처리 완료**:
  - C3 블록: cv1, cv2, cv3 모두 Fused BN 지원
  - SPPF 블록: cv1, cv2 모두 Fused BN 지원
  - Conv 레이어: Fused BN 감지 및 처리
- **모든 레이어 출력 저장 시스템 구축**:
  - `dump_golden.py`: 모든 레이어 저장 (기본값)
  - `yolov5s_infer.c`: 모든 레이어에서 `save_feature()` 호출
  - `testdata/python/`, `testdata/c/`에 레이어별 출력 저장

### 10. 문서화 ✅
- **핵심 문서**:
  - `README.md`: 프로젝트 개요 및 전체 가이드 (AI agent용 포괄적 문서)
  - `docs/MODULE_ARCHITECTURE.md`: 모듈 아키텍처 설명
  - `docs/INFERENCE_FLOW.md`: 인퍼런스 파이프라인 설명
  - `docs/DETECTION_FLOW.md`: 검출 파이프라인 설명
  - `docs/PREPROCESSING.md`: 이미지 전처리 과정 설명
- **검증 및 디버깅 문서**:
  - `docs/DEBUGGING_PROCESS.md`: 디버깅 방법론 및 해결 사례 (C3, SPPF)
  - `docs/IMAGE_INFERENCE_WORKFLOW.md`: 이미지 입력 전체 워크플로우
  - `docs/ACCURACY_VALIDATION.md`: 정확도 검증 방법
  - `docs/GOLDEN_TENSOR_STORAGE.md`: Golden 텐서 저장 형식
  - `docs/LAYER_STRUCTURE_COMPARISON.md`: PyTorch vs C 레이어 구조 비교
- **프로젝트 관리 문서**:
  - `PROJECT_STATUS.md`: 이 파일 (프로젝트 상태)
  - `TESTING.md`: 테스트 가이드

## 프로젝트 요구사항 대비 현재 상태

### ✅ 완료된 요구사항
- **YOLO-v5 inference 구현**: YOLOv5s 모델을 C로 완전히 포팅 완료
- **정확도 검증**: PyTorch와 레이어별 비교 검증 완료 (Layer 0-23 모두 일치)

### ❌ 미완료 요구사항
- **MicroBlaze RISC-V / FPGA 타겟팅**: 현재 일반 PC 환경 (Windows/Linux)만 지원
- **Vitis 환경 지원**: Vitis 빌드 시스템 및 플랫폼 설정 없음
- **Digilent Arty A7-35 타겟팅**: FPGA 보드 특화 설정 없음
- **성능 병목 분석**: 프로파일링 도구 및 분석 없음
- **Custom NPU/Accelerator 설계**: 하드웨어 가속기 설계 없음

## 진행 중 / TODO

### Phase 1: FPGA 플랫폼 포팅 (필수)

#### 1.1 Vitis 환경 설정
- [ ] Vitis 프로젝트 생성 및 설정
- [ ] MicroBlaze RISC-V 프로세서 설정 (별도 프로젝트에서 제공)
- [ ] Digilent Arty A7-35 보드 지원 파일 추가
- [ ] 플랫폼 정의 파일 (platform.xsa) 통합
- [ ] Vitis 빌드 시스템 (CMakeLists.txt 또는 Makefile) 구성

#### 1.2 코드 포팅 및 최적화
- [ ] **메모리 제약 대응**:
  - [ ] FPGA 메모리 제약에 맞는 메모리 사용량 최적화
  - [ ] Arena allocator 크기 조정 (제한된 BRAM/DDR)
  - [ ] 텐서 크기 제한 및 동적 할당 최적화
- [ ] **RISC-V 호환성**:
  - [ ] RISC-V 컴파일러 (riscv64-unknown-elf-gcc)로 빌드 테스트
  - [ ] 엔디안 처리 확인 (Little-endian)
  - [ ] 플랫폼별 매크로 정의 (__riscv__, __MICROBLAZE__ 등)
- [ ] **라이브러리 의존성**:
  - [ ] 표준 라이브러리 최소화 (FPGA 환경 제약)
  - [ ] JSON 파서 (jsmn) RISC-V 호환성 확인
  - [ ] 수학 함수 (math.h) 사용 최소화 또는 자체 구현

#### 1.3 하드웨어 인터페이스
- [ ] **입력/출력 인터페이스**:
  - [ ] 이미지 입력 인터페이스 (UART, SPI, I2C, 또는 커스텀)
  - [ ] 결과 출력 인터페이스
  - [ ] 가중치 로드 방법 (Flash, SD 카드, 또는 하드코딩)
- [ ] **디버깅 지원**:
  - [ ] UART를 통한 디버그 출력
  - [ ] Vitis 디버거 연동
  - [ ] 중간 텐서 덤프 기능 (제한된 메모리 고려)

#### 1.4 빌드 및 배포
- [ ] Vitis에서 빌드 및 ELF 생성
- [ ] FPGA 비트스트림 생성 및 통합
- [ ] 부팅 및 실행 테스트
- [ ] 실제 하드웨어에서 End-to-end 테스트

### Phase 2: 성능 분석 및 최적화 (Optional)

#### 2.1 성능 프로파일링
- [ ] **병목 구간 분석**:
  - [ ] 각 레이어별 실행 시간 측정
  - [ ] 메모리 접근 패턴 분석
  - [ ] 연산 집약도 분석 (Conv, MatMul, Activation 등)
  - [ ] 프로파일링 도구 통합 (Vitis Analyzer, 또는 커스텀 타이머)
- [ ] **리소스 사용량 분석**:
  - [ ] 메모리 사용량 (BRAM, DDR) 측정
  - [ ] CPU 사용률 분석
  - [ ] 캐시 미스 분석 (있는 경우)

#### 2.2 성능 리포트 작성
- [ ] 레이어별 실행 시간 리포트
- [ ] 전체 추론 시간 측정
- [ ] 메모리 사용량 리포트
- [ ] 병목 구간 식별 및 우선순위 결정

### Phase 3: Custom NPU/Accelerator 설계 (Optional)

#### 3.1 가속기 타겟 선정
- [ ] 병목 분석 결과 기반으로 가속할 연산 결정
  - [ ] Conv2D 연산 (가장 시간 소모적일 가능성 높음)
  - [ ] MatMul 연산
  - [ ] Activation 함수 (SiLU)
  - [ ] 기타 연산

#### 3.2 하드웨어 설계
- [ ] **Custom IP 설계**:
  - [ ] AXI4 인터페이스 설계
  - [ ] 연산 유닛 설계 (예: Convolution Engine)
  - [ ] 메모리 인터페이스 설계
  - [ ] 제어 로직 설계
- [ ] **통합**:
  - [ ] MicroBlaze와 Custom IP 연결
  - [ ] 메모리 맵 정의
  - [ ] 인터럽트 또는 폴링 방식 선택

#### 3.3 소프트웨어 인터페이스
- [ ] **드라이버 개발**:
  - [ ] Custom IP 제어 함수 작성
  - [ ] 메모리 매핑 및 DMA 설정
  - [ ] 동기화 메커니즘 (인터럽트/폴링)
- [ ] **코드 수정**:
  - [ ] 가속 가능한 연산을 Custom IP로 오프로드
  - [ ] 폴백 메커니즘 (가속기 실패 시 CPU 실행)
  - [ ] 성능 모니터링 코드 추가

#### 3.4 검증 및 비교
- [ ] **기능 검증**:
  - [ ] 가속기 출력과 CPU 출력 비교
  - [ ] 정확도 검증 (PyTorch와 비교)
- [ ] **성능 비교**:
  - [ ] 가속 전·후 실행 시간 측정
  - [ ] 처리량 (FPS) 비교
  - [ ] 리소스 사용량 비교 (LUT, BRAM, DSP 등)
- [ ] **최종 리포트**:
  - [ ] 성능 개선율 계산
  - [ ] 리소스 오버헤드 분석
  - [ ] 비용-효과 분석

### Phase 4: 추가 최적화 (선택사항)

#### 4.1 소프트웨어 최적화
- [ ] SIMD 명령어 활용 (RISC-V Vector Extension, 있는 경우)
- [ ] 루프 언롤링 및 최적화
- [ ] 메모리 접근 패턴 최적화
- [ ] 캐시 친화적 데이터 배치

#### 4.2 하드웨어 최적화
- [ ] 파이프라인 최적화
- [ ] 병렬 처리 확대
- [ ] 메모리 대역폭 최적화
- [ ] 전력 소비 최적화

### Phase 5: 테스트 및 검증

#### 5.1 기능 테스트
- [ ] 다양한 입력 크기 테스트 (320, 480, 640, 960 등)
- [ ] 다양한 테스트 이미지로 검증
- [ ] 엣지 케이스 테스트 (빈 이미지, 극단적 크기 등)

#### 5.2 성능 테스트
- [ ] 벤치마크 스위트 실행
- [ ] 장기 안정성 테스트
- [ ] 온도 및 전압 변동 테스트

#### 5.3 문서화
- [ ] FPGA 포팅 가이드 작성
- [ ] Vitis 빌드 가이드 작성
- [ ] 하드웨어 설정 가이드 작성
- [ ] 성능 분석 리포트 작성
- [ ] Custom IP 사용 가이드 작성 (있는 경우)

## 파일 구조

```
yolov5_c/
├── CMakeLists.txt          # 빌드 설정 ✅
├── README.md               # 프로젝트 개요 (AI agent용 포괄적 문서) ✅
├── PROJECT_STATUS.md       # 이 파일 ✅
├── TESTING.md              # 테스트 가이드 ✅
├── src/
│   ├── core/              # 텐서/메모리/가중치 로더 ✅
│   │   ├── tensor.h/c     # 텐서 구조체 및 유틸리티
│   │   ├── memory.h/c      # Arena allocator
│   │   ├── weights_loader.h/c  # 가중치 로더
│   │   └── common.h       # 공통 매크로 (SNPRINTF 등)
│   ├── ops/               # Primitive 연산 ✅
│   │   ├── conv2d.h/c     # 2D Convolution (Fused BN 지원)
│   │   ├── batchnorm2d.h/c # Batch Normalization
│   │   ├── activation.h/c  # SiLU 활성화
│   │   ├── pooling.h/c    # MaxPool2D
│   │   ├── upsample.h/c   # Upsample
│   │   └── concat.h/c      # Concat
│   ├── blocks/            # C3, Bottleneck, SPPF ✅
│   │   ├── bottleneck.h/c
│   │   ├── c3.h/c         # Fused BN 지원, 디버깅 출력
│   │   └── sppf.h/c       # 로직 수정 완료, Fused BN 지원, 디버깅 출력
│   ├── models/            # 모델 빌드/인퍼런스 ✅
│   │   ├── yolov5s_graph.h/c      # 모델 그래프 정의
│   │   ├── yolov5s_build.h/c      # 모델 빌드 (Fused BN 감지)
│   │   ├── yolov5s_infer.h/c      # 인퍼런스 파이프라인 (모든 레이어 출력 저장)
│   │   └── yolov5s_infer_utils.h  # 유틸리티 매크로
│   ├── postprocess/       # Detect decode + NMS ✅
│   │   ├── detect.h/c     # Detect 헤드 및 decode
│   │   └── nms.h/c        # Non-Maximum Suppression
│   └── main.c             # 메인 실행 파일 ✅
├── tools/                 # Python 도구 ✅
│   ├── preprocess.py      # 이미지 전처리
│   ├── export_yolov5s.py  # 가중치 export
│   ├── dump_golden.py     # 골든 텐서 덤프 (모든 레이어)
│   ├── compare_tensors.py # 텐서 비교
│   ├── debug_layer2.py    # C3 블록 디버깅
│   ├── debug_layer9.py    # SPPF 블록 디버깅
│   ├── compare_c3_steps.py # C3 단계별 비교
│   └── compare_sppf_steps.py # SPPF 단계별 비교
├── tests/                 # 단위 테스트
│   ├── CMakeLists.txt
│   ├── test_conv1x1.c
│   └── test_integration.c
├── docs/                  # 문서 ✅
│   ├── MODULE_ARCHITECTURE.md
│   ├── INFERENCE_FLOW.md
│   ├── DETECTION_FLOW.md
│   ├── PREPROCESSING.md
│   ├── DEBUGGING_PROCESS.md
│   ├── IMAGE_INFERENCE_WORKFLOW.md
│   ├── ACCURACY_VALIDATION.md
│   ├── GOLDEN_TENSOR_STORAGE.md
│   └── LAYER_STRUCTURE_COMPARISON.md
├── data/                  # 데이터
│   ├── images/           # 원본 이미지
│   ├── inputs/           # 전처리된 텐서 (.bin, _meta.txt)
│   └── outputs/          # 검출 결과 (.txt)
├── testdata/             # 검증 데이터 ✅
│   ├── python/           # PyTorch golden 출력
│   │   ├── input.bin
│   │   ├── layer_000.bin ~ layer_023.bin
│   │   └── output_0.bin
│   └── c/                # C 구현 출력
│       ├── input.bin
│       ├── layer_000.bin ~ layer_023.bin
│       ├── output_p3.bin
│       ├── output_p4.bin
│       └── output_p5.bin
├── debug/                # 디버깅 중간 출력 ✅
│   ├── pytorch/          # PyTorch 중간 텐서
│   └── c/                # C 중간 텐서
└── weights/              # 모델 가중치 ✅
    ├── yolov5s.pt        # PyTorch 모델 (원본)
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
6. **정확도 검증 완료**: PyTorch와 레이어별 비교 검증 (Layer 0-23)
7. **Fused Batch Normalization 처리**: 모든 Conv+BN 조합에서 지원
8. **모든 레이어 출력 저장**: 디버깅 및 검증을 위한 완전한 텐서 저장

### 🔧 최근 개선사항 (완료)
- **Layer 2 (C3 블록) 수정**: cv2 경로에 SiLU activation 추가
- **Layer 9 (SPPF 블록) 수정**: MaxPool 로직 수정, Fused BN 처리 추가
- **Fused BN 처리**: C3, SPPF, Conv 레이어 모두 지원
- **모든 레이어 출력 저장**: `dump_golden.py` 및 `yolov5s_infer.c` 수정
- **디버깅 도구 개발**: 단계별 비교 스크립트 추가
- **문서화 완료**: 디버깅 방법론, 이미지 워크플로우, 정확도 검증 문서 추가
- **README 개선**: AI agent가 프로젝트를 완전히 이해할 수 있도록 포괄적 문서 작성

## 검증된 수정 사항

### Layer 2 (C3 블록)
- **문제**: cv2 경로에 SiLU activation 누락
- **원인**: PyTorch의 `Conv` 클래스는 기본적으로 SiLU activation을 포함하는데, C 구현에서는 누락됨
- **해결**: `src/blocks/c3.c`의 `c3_forward` 함수에서 `activation_silu(skip_output)` 추가
- **검증**: `compare_c3_steps.py`로 모든 단계 일치 확인

### Layer 9 (SPPF 블록)
- **문제 1**: MaxPool 로직 오류
  - PyTorch: `y1 = m(x)`, `y2 = m(y1)`, `y4 = m(y2)` (3번만 MaxPool 호출)
  - C (이전): `y1 = x`, `y2 = m(y1)`, `y3 = m(y2)`, `y4 = m(y3)` (4번 호출, y3 불필요)
- **문제 2**: Concat 순서 오류
  - PyTorch: `concat([x, y1, y2, y4])`
  - C (이전): `concat([y1, y2, y3, y4])`
- **문제 3**: Fused BN 처리 누락
  - cv1, cv2 모두 Fused BN을 확인하지 않고 항상 BN을 실행
- **해결**:
  1. SPPF 로직 수정: `y1 = m(x)`, `y2 = m(y1)`, `y4 = m(y2)`, `concat([x, y1, y2, y4])`
  2. Fused BN 처리 추가: `cv1_is_fused`, `cv2_is_fused` 플래그 추가
- **검증**: `compare_sppf_steps.py`로 모든 단계 일치 확인

## 다음 단계

1. **성능 최적화**: SIMD 활용 및 메모리 최적화
2. **추가 테스트**: 다양한 입력 크기 및 이미지 테스트
3. **문서 개선**: 필요시 추가 문서 작성

## 참고 문서

### 핵심 문서
- `README.md`: 프로젝트 개요 및 전체 가이드 (AI agent용 포괄적 문서)
- `TESTING.md`: 테스트 가이드
- `docs/MODULE_ARCHITECTURE.md`: 모듈 아키텍처 설명
- `docs/INFERENCE_FLOW.md`: 인퍼런스 파이프라인 설명
- `docs/DETECTION_FLOW.md`: 검출 파이프라인 설명
- `docs/PREPROCESSING.md`: 이미지 전처리 과정 설명

### 검증 및 디버깅 문서
- `docs/DEBUGGING_PROCESS.md`: 디버깅 방법론 및 해결 사례 (C3, SPPF)
- `docs/IMAGE_INFERENCE_WORKFLOW.md`: 이미지 입력 전체 워크플로우
- `docs/ACCURACY_VALIDATION.md`: 정확도 검증 방법
- `docs/GOLDEN_TENSOR_STORAGE.md`: Golden 텐서 저장 형식
- `docs/LAYER_STRUCTURE_COMPARISON.md`: PyTorch vs C 레이어 구조 비교

---

**마지막 업데이트**: 2024년

**현재 상태**: ✅ **정확도 검증 완료** - PyTorch와 레이어별 비교 검증 완료 (Layer 0-23 모두 일치)
