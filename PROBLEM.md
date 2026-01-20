# YOLOv5s C 포팅 디버깅 문제 정리

## 개요
YOLOv5s 모델을 Python/PyTorch에서 순수 C로 포팅하는 과정에서 발생한 문제들과 진단 결과를 정리한 문서입니다.

---

## 1. 초기 문제: Layer 0 (Conv0) 출력 불일치

### 증상
- `layer_000.bin` (Conv0 출력)이 Python golden과 큰 차이를 보임
- `compare_tensors.py`에서 `Max diff: ~100+` 수준의 오차 발생

### 원인
- **BN (Batch Normalization) Fusion 문제**
- PyTorch YOLOv5는 추론 시 Conv와 BN 레이어를 자동으로 융합(fuse)하여 단일 Conv 레이어로 변환
- 융합된 모델에서는 Conv 레이어에 bias가 포함되어 있음 (원래는 BN이 bias 역할)
- C 구현에서는 Conv와 BN을 별도로 처리하고 있었음

### 해결
1. `tools/export_yolov5s.py` 수정:
   - `attempt_load` 호출 시 `fuse=True` 옵션 추가
   - 융합된 가중치를 `weights.bin`에 저장

2. `src/models/yolov5s_build.c` 수정:
   - `load_conv_bn_layer` 함수에서 `model.X.conv.bias` (융합된 bias) 확인
   - 융합된 bias가 있으면 Conv에 직접 로드하고, BN은 identity로 설정 (gamma=1, beta=0, mean=0, var=1)
   - `is_fused` 플래그 추가하여 추론 시 BN skip 가능하도록 함

3. `src/models/yolov5s_infer.c` 수정:
   - `is_fused` 플래그가 true이면 `batchnorm2d_forward` 호출 스킵

### 결과
- Layer 0 출력이 Python golden과 일치하게 됨
- 이후 Layer 1, Layer 2로 디버깅 진행

---

## 2. Weight Loading 문제: weights_map.json 파싱 실패

### 증상
- C 추론 실행 시 "Failed to load weight for model.X.conv.weight" 에러
- `parse_weights_map: Parsed 0 entries` 출력
- 모델 빌드 실패

### 원인
- **라인 기반 JSON 파서의 취약성**
- `src/core/weights_loader.c`의 `parse_weights_map` 함수가 라인 단위로 파싱
- JSON 포맷터/pretty-print에 따라 다음 경우 실패:
  - `"offset"`과 `:`가 다른 줄에 있을 때
  - `"shape": [` 배열이 여러 줄로 나뉠 때
  - 공백/들여쓰기 형식이 달라질 때

### 해결
- **jsmn 라이브러리 도입**
- `third_party/jsmn/` 디렉토리에 `jsmn.h`, `jsmn.c` 추가
- `parse_weights_map` 함수를 완전히 재작성:
  - JSON 파일 전체를 메모리에 로드
  - `jsmn_parse`로 토큰화
  - 토큰 배열을 순회하며 `name`, `offset`, `shape` 추출
- 포맷에 독립적으로 파싱 가능

### 결과
- 모든 120개 weight entry 정상 파싱
- 모델 빌드 성공

---

## 3. C3 Block 문제: Layer 2 출력 불일치

### 증상
- `layer_002.bin` (C3 block 출력)이 Python golden과 불일치
- `Max diff: ~80+` 수준의 오차

### 원인
- C3 block 내부의 Conv 레이어들(cv1, cv2, cv3)도 BN fusion이 적용됨
- Bottleneck 내부의 Conv 레이어들(conv1, conv2)도 BN fusion이 적용됨
- C 구현에서는 이들에 대한 BN fusion 처리가 없었음

### 해결
1. `src/blocks/c3.h` 수정:
   - `cv1_is_fused`, `cv2_is_fused`, `cv3_is_fused` 플래그 추가

2. `src/blocks/c3.c` 수정:
   - `c3_load_weights`에서 각 cv1, cv2, cv3의 `.conv.bias` 확인
   - 융합된 bias가 있으면 Conv에 로드하고 BN을 identity로 설정
   - `c3_forward`에서 `is_fused` 플래그에 따라 BN skip

3. `src/blocks/bottleneck.h` 수정:
   - `conv1_is_fused`, `conv2_is_fused` 플래그 추가

4. `src/blocks/bottleneck.c` 수정:
   - `bottleneck_load_weights`에서 각 conv1, conv2의 `.conv.bias` 확인
   - 융합된 bias가 있으면 Conv에 로드하고 BN을 identity로 설정
   - `bottleneck_forward`에서 `is_fused` 플래그에 따라 BN skip

### 결과
- C3 cv1 Conv 출력이 Python golden과 일치 (`Max diff: ~1e-05`)
- C3 cv1+BN+SiLU 출력도 일치
- 문제가 Bottleneck 0 출력으로 이동

---

## 4. 현재 문제: Bottleneck cv1 Conv 출력이 Bias만 출력됨

### 증상
- `test_bottleneck_cv1_single_value.py` 실행 시:
  - C 출력이 bias 값과 정확히 일치 (예: `1.400894`)
  - Conv 연산 기여도가 0 (`conv_sum=0.000000`)
- `compare_c3_steps.py`에서:
  - `Bottleneck 0 output`: `Max diff: 83.3` (FAIL)

### 디버깅 과정

#### 4.1 초기 관찰
- `bottleneck_forward`에서 `input` tensor 확인:
  - `input ptr=0000029F916612F0, data ptr=0000029F96393040`
  - `input shape=(1,32,160,160)`
  - `sample[0,0,0,0]=0.244718, [0,1,0,0]=1.508039, [0,31,0,0]=-0.156047`
  - **입력 데이터는 유효함**

#### 4.2 conv2d_forward 내부 확인
- `conv2d_forward` 진입 시:
  - `input tensor ptr=0000029F916612F0, data ptr=0000029F96393040` (포인터는 동일)
  - `input tensor shape: (1,32,160,160)` (shape도 동일)
  - **하지만 `input->data` 직접 접근 시 모두 0:**
    - `input direct[0,0,0,0]=0.000000`
    - `input direct[0,1,0,0]=0.000000`
    - `input direct[0,31,0,0]=0.000000`
  - `tensor_at_const`로 접근해도 0

#### 4.3 문제 분석
- **포인터는 동일하지만 데이터가 0으로 읽힘**
- 가능한 원인:
  1. **메모리 오버라이드**: `bottleneck_forward`와 `conv2d_forward` 사이에 다른 코드가 `input->data`를 덮어씀
  2. **잘못된 포인터 해석**: `tensor_t` 구조체의 `data` 포인터가 실제 데이터를 가리키지 않음
  3. **메모리 해제**: `input` tensor의 메모리가 조기에 해제됨
  4. **버퍼 재사용**: `workspace` 버퍼가 `input`과 같은 메모리를 가리키고, `tensor_zero` 등으로 초기화됨

### 현재 상태
- **진단 중**: `bottleneck_forward`에서 `input` tensor의 데이터가 유효하지만, `conv2d_forward` 내부에서 0으로 읽힘
- **추가 조사 필요**:
  - `bottleneck_forward`에서 `conv2d_forward` 호출 직전/직후 메모리 상태 확인
  - `workspace` 버퍼와 `input` 버퍼의 메모리 관계 확인
  - `tensor_zero` 호출 위치 확인

---

## 5. 디버깅 도구

### Python 스크립트
- `tools/debug_conv0_weights.py`: Conv0 가중치 비교
- `tools/debug_conv0_forward.py`: Conv0 forward pass 단계별 비교
- `tools/debug_c3_forward.py`: C3 block forward pass 단계별 비교
- `tools/debug_bottleneck_forward.py`: Bottleneck block forward pass 단계별 비교
- `tools/test_c3_cv1_single_value.py`: C3 cv1 Conv 단일 값 수동 계산
- `tools/test_bottleneck_cv1_single_value.py`: Bottleneck cv1 Conv 단일 값 수동 계산
- `tools/compare_c3_steps.py`: C3 block 중간 단계별 비교

### C 디버그 출력
- `src/blocks/bottleneck.c`: Bottleneck 내부 중간 텐서 dump
- `src/blocks/c3.c`: C3 내부 중간 텐서 dump
- `src/ops/conv2d.c`: Conv 연산 디버그 출력 (1x1 path)

---

## 6. 다음 단계

### 우선순위 1: Bottleneck cv1 Conv 문제 해결
1. `bottleneck_forward`에서 `workspace` 버퍼 할당/초기화 확인
2. `conv2d_forward` 호출 전 `input` tensor 메모리 무결성 확인
3. `tensor_zero` 호출이 `input`에 영향을 주는지 확인

### 우선순위 2: 전체 파이프라인 검증
1. 모든 레이어 출력 비교 (`layer_000.bin` ~ `layer_023.bin`)
2. Detect head 출력 비교 (`output_p3.bin`, `output_p4.bin`, `output_p5.bin`)
3. 최종 detection 결과 비교 (`bus_detections.txt`)

### 우선순위 3: 성능 최적화
1. SIMD 활용 (AVX, NEON 등)
2. 메모리 할당 최적화
3. 빌드 최적화 플래그 적용

---

## 참고
- Python golden 출력은 `testdata/python/` 디렉토리에 저장됨
- C 레이어 출력은 `testdata/c/` 디렉토리에 저장됨 (layer_*.bin, output_p*.bin)
- PyTorch 디버그 출력은 `debug/pytorch/` 디렉토리에 저장됨
- C 디버그 출력은 `debug/c/` 디렉토리에 저장됨 (c3_*.bin, bottleneck_*.bin 등)
- 비교 스크립트: `python tools/compare_tensors.py`
