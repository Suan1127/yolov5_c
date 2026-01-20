# YOLOv5s C 포팅 디버깅 과정 정리

이 문서는 YOLOv5s 모델을 Python/PyTorch에서 순수 C로 포팅하는 과정에서 발생한 문제들을 단계별로 디버깅하고 해결한 과정을 정리합니다.

## 목차

1. [전체 디버깅 워크플로우](#전체-디버깅-워크플로우)
2. [주요 문제 및 해결 과정](#주요-문제-및-해결-과정)
3. [중간 텐서 비교 방법](#중간-텐서-비교-방법)
4. [학습한 교훈](#학습한-교훈)

---

## 전체 디버깅 워크플로우

### 1. 초기 설정

```bash
# 1. PyTorch golden 데이터 생성
python tools/dump_golden.py weights/yolov5s.pt input.bin --output testdata/python

# 2. C 프로그램 실행
cd build/Release
cmake --build . --config Release
yolov5s_infer.exe

# 3. 전체 레이어 비교
python tools/compare_tensors.py testdata/python testdata/c
```

### 2. 문제 발견 → 단계별 디버깅

1. **전체 비교로 첫 번째 불일치 레이어 찾기**
   - `compare_tensors.py`로 전체 레이어 비교
   - 첫 번째 실패한 레이어 확인

2. **해당 레이어의 중간 출력 생성**
   - Python: `debug_layerX.py` 스크립트 작성
   - C: 디버그 출력 추가 (`c3_set_debug_dir`, `sppf_set_debug_dir`)

3. **단계별 비교**
   - `compare_c3_steps.py` 또는 `compare_sppf_steps.py` 실행
   - 어느 단계에서 불일치가 발생하는지 확인

4. **근본 원인 파악 및 수정**
   - PyTorch 구현 확인
   - C 구현과 비교
   - 수정 후 재검증

---

## 주요 문제 및 해결 과정

### 문제 1: Layer 2 (C3 블록) 출력 불일치

#### 증상
- `layer_002.bin`에서 첫 번째 불일치 발생
- `compare_tensors.py` 결과: Max diff가 매우 큼

#### 디버깅 과정

**1단계: C3 블록 중간 출력 생성**

**Python (golden reference):**
```python
# tools/debug_layer2.py 생성
# C3 블록의 각 단계별 출력 저장:
# - c3_cv1_output.bin (Conv + BN + SiLU)
# - c3_bottleneck_output.bin (Bottleneck 통과)
# - c3_cv2_output.bin (skip path: Conv + BN + SiLU)
# - c3_concat_output.bin (cv1 path와 cv2 path concat)
# - c3_final_output.bin (cv3 통과 후 최종 출력)
```

**C 구현:**
```c
// src/blocks/c3.c에 디버그 출력 추가
void c3_set_debug_dir(const char* dir);
// 각 단계별로 tensor_dump 호출
```

**2단계: 단계별 비교**

```bash
python tools/compare_c3_steps.py
```

**결과:**
- ✅ cv1: OK
- ✅ Bottleneck: OK
- ❌ cv2: FAIL (C 출력 범위: [-89.988815, 27.259954], PyTorch: [-0.278465, 27.259943])

**3단계: 근본 원인 파악**

**PyTorch 코드 확인:**
```python
# third_party/yolov5/models/common.py
class Conv(nn.Module):
    def __init__(self, ...):
        self.conv = nn.Conv2d(...)
        self.bn = nn.BatchNorm2d(...)
        self.act = nn.SiLU()  # 기본적으로 SiLU activation 포함
```

**C 코드 확인:**
```c
// src/blocks/c3.c
// cv2 경로에 activation_silu 호출이 없었음!
// 주석: "Note: cv2 doesn't have activation in original YOLOv5"
```

**4단계: 수정**

```c
// src/blocks/c3.c의 c3_forward 함수
// cv2 경로에 SiLU activation 추가
if (!block->cv2_is_fused) {
    if (batchnorm2d_forward(&block->cv2_bn, skip_output, skip_output) != 0) goto error;
}
activation_silu(skip_output);  // ← 추가!
```

**결과:**
- ✅ 모든 C3 단계 일치
- ✅ Layer 2 출력 일치

---

### 문제 2: Layer 9 (SPPF 블록) 출력 불일치

#### 증상
- `layer_009.bin`에서 불일치 발생
- Max diff: 5.826019e+05 (매우 큰 오차)

#### 디버깅 과정

**1단계: SPPF 블록 구조 확인**

**PyTorch 구현:**
```python
# third_party/yolov5/models/common.py
def forward(self, x):
    x = self.cv1(x)  # Conv + BN + SiLU
    y1 = self.m(x)    # MaxPool
    y2 = self.m(y1)   # MaxPool
    y4 = self.m(y2)   # MaxPool (y3 없음!)
    return self.cv2(torch.cat((x, y1, y2, y4), 1))  # [x, y1, y2, y4] concat
```

**C 구현 (잘못된 버전):**
```c
// 이전 구현
y1 = x
y2 = m(y1)
y3 = m(y2)  // ← 불필요!
y4 = m(y3)
concat([y1, y2, y3, y4])  // ← 잘못된 순서!
```

**2단계: 중간 출력 생성**

**Python:**
```python
# tools/debug_layer9.py 생성
# SPPF 블록의 각 단계별 출력 저장:
# - sppf_cv1_output.bin
# - sppf_y1_output.bin (m(x))
# - sppf_y2_output.bin (m(y1))
# - sppf_y4_output.bin (m(y2))
# - sppf_concat_output.bin
# - sppf_cv2_output.bin
```

**C:**
```c
// src/blocks/sppf.c에 디버그 출력 추가
void sppf_set_debug_dir(const char* dir);
```

**3단계: 단계별 비교**

```bash
python tools/compare_sppf_steps.py
```

**결과:**
- ❌ cv1: FAIL (C 범위: [-0.278464, 1442.601074], PyTorch: [-0.278465, 5.597557])
- ❌ 모든 단계 실패

**4단계: 근본 원인 파악**

**문제 1: SPPF 로직 오류**
- PyTorch: `y1 = m(x)`, `y2 = m(y1)`, `y4 = m(y2)`, `concat([x, y1, y2, y4])`
- C (이전): `y1 = x`, `y2 = m(y1)`, `y3 = m(y2)`, `y4 = m(y3)`, `concat([y1, y2, y3, y4])`

**문제 2: Fused BN 처리 누락**
- C3 블록과 달리 SPPF에는 fused BN 처리가 없었음
- cv1, cv2 모두 fused BN을 확인하지 않고 항상 BN을 실행

**5단계: 수정**

**수정 1: SPPF 로직 수정**
```c
// src/blocks/sppf.c
// PyTorch와 일치하도록 수정
tensor_t* x = workspace1;  // cv1 output
tensor_t* y1 = tensor_create(...);
tensor_t* y2 = tensor_create(...);
tensor_t* y4 = tensor_create(...);

y1 = m(x);   // ← 수정
y2 = m(y1);  // ← 수정
y4 = m(y2);  // ← 수정 (y3 제거)
concat([x, y1, y2, y4]);  // ← 수정
```

**수정 2: Fused BN 처리 추가**
```c
// src/blocks/sppf.h
typedef struct {
    ...
    int cv1_is_fused;  // ← 추가
    int cv2_is_fused;  // ← 추가
} sppf_block_t;

// src/blocks/sppf.c
// cv1, cv2 모두 fused BN 확인 및 처리
if (fused_bias) {
    block->cv1_is_fused = 1;
    // BN을 identity로 설정
} else {
    block->cv1_is_fused = 0;
    // BN 가중치 로드
}

// Forward pass에서
if (!block->cv1_is_fused) {
    batchnorm2d_forward(...);
}
```

**결과:**
- ✅ 모든 SPPF 단계 일치
- ✅ Layer 9 출력 일치

---

## 중간 텐서 비교 방법

### 1. 전체 레이어 비교

**스크립트:** `tools/compare_tensors.py`

**사용법:**
```bash
python tools/compare_tensors.py testdata/python testdata/c
```

**출력:**
- 각 레이어별 비교 결과
- Max diff, Mean diff, RMSE
- 첫 번째 실패한 레이어 표시
- Upsample 레이어(11, 15)는 자동으로 SKIP
- Output 파일들은 자동으로 SKIP

### 2. 특정 블록 단계별 비교

**C3 블록:**
```bash
# 1. PyTorch 중간 출력 생성
python tools/debug_layer2.py

# 2. C 프로그램 실행 (디버그 모드)
# src/models/yolov5s_infer.c에서 Layer 2 실행 전:
c3_set_debug_dir("debug/c");

# 3. 비교
python tools/compare_c3_steps.py
```

**SPPF 블록:**
```bash
# 1. PyTorch 중간 출력 생성
python tools/debug_layer9.py

# 2. C 프로그램 실행 (디버그 모드)
# src/models/yolov5s_infer.c에서 Layer 9 실행 전:
sppf_set_debug_dir("debug/c");

# 3. 비교
python tools/compare_sppf_steps.py
```

### 3. 디버그 출력 구조

```
debug/
├── pytorch/          # PyTorch golden 중간 출력
│   ├── c3_cv1_output.bin
│   ├── c3_bottleneck_output.bin
│   ├── c3_cv2_output.bin
│   ├── c3_concat_output.bin
│   ├── c3_final_output.bin
│   ├── sppf_cv1_output.bin
│   ├── sppf_y1_output.bin
│   ├── sppf_y2_output.bin
│   ├── sppf_y4_output.bin
│   ├── sppf_concat_output.bin
│   └── sppf_cv2_output.bin
└── c/                # C 구현 중간 출력
    └── (동일한 파일명)
```

---

## 학습한 교훈

### 1. PyTorch 모듈 구조 정확히 이해하기

**문제:**
- `Conv` 클래스가 기본적으로 `SiLU` activation을 포함한다는 것을 놓침
- C3 블록의 `cv2`도 `Conv` 인스턴스이므로 activation이 필요함

**해결:**
- PyTorch 소스 코드를 직접 확인
- `third_party/yolov5/models/common.py`의 `Conv` 클래스 구조 확인

### 2. Fused BN 처리 일관성

**문제:**
- C3 블록에는 fused BN 처리가 있었지만, SPPF에는 없었음
- 일부 레이어만 fused BN을 처리하여 불일치 발생

**해결:**
- 모든 Conv+BN 조합에 대해 fused BN 처리 추가
- `is_fused` 플래그로 일관된 처리

### 3. 단계별 디버깅의 중요성

**접근법:**
1. 전체 비교로 문제 레이어 찾기
2. 해당 레이어를 구성 요소로 분해
3. 각 구성 요소의 중간 출력 비교
4. 첫 번째 실패한 구성 요소에서 문제 해결

**효과:**
- 문제를 빠르게 좁혀나갈 수 있음
- 여러 문제가 있을 때도 하나씩 해결 가능

### 4. PyTorch 구현과 C 구현의 정확한 대응

**주의사항:**
- PyTorch 코드의 주석이나 변수명만 믿지 말 것
- 실제 forward 함수의 로직을 직접 확인
- 중간 변수명(y1, y2, y3, y4)이 혼동을 줄 수 있음

**예시:**
- SPPF에서 PyTorch는 `y1 = m(x)`, `y2 = m(y1)`, `y4 = m(y2)`로 3번만 MaxPool 호출
- C 구현은 `y1 = x`, `y2 = m(y1)`, `y3 = m(y2)`, `y4 = m(y3)`로 4번 호출 (잘못됨)

### 5. 모든 레이어 출력 저장의 중요성

**초기 문제:**
- 일부 레이어만 저장하여 디버깅 시 필요한 입력을 찾을 수 없음
- 예: Layer 9 디버깅 시 Layer 8 출력이 필요했지만 없었음

**해결:**
- 모든 레이어(0-23) 출력 저장
- `dump_golden.py`: 모든 레이어 저장
- `yolov5s_infer.c`: 모든 레이어에서 `save_feature` 호출

---

## 디버깅 체크리스트

새로운 불일치가 발견되면:

1. ✅ **전체 비교로 첫 번째 실패 레이어 확인**
   ```bash
   python tools/compare_tensors.py testdata/python testdata/c
   ```

2. ✅ **해당 레이어의 구조 확인**
   - Conv 블록인가? C3 블록인가? SPPF 블록인가?
   - PyTorch 구현 확인

3. ✅ **중간 출력 생성 스크립트 작성**
   - Python: `tools/debug_layerX.py`
   - C: 디버그 출력 추가

4. ✅ **단계별 비교**
   - 각 구성 요소별로 비교
   - 첫 번째 실패한 단계 확인

5. ✅ **근본 원인 파악**
   - PyTorch 구현과 C 구현 비교
   - Activation 누락? Fused BN? 로직 오류?

6. ✅ **수정 및 검증**
   - 수정 후 재빌드
   - 비교 스크립트로 검증

---

## 참고 파일

### Python 스크립트
- `tools/compare_tensors.py`: 전체 레이어 비교
- `tools/debug_layer2.py`: C3 블록 중간 출력 생성
- `tools/debug_layer9.py`: SPPF 블록 중간 출력 생성
- `tools/compare_c3_steps.py`: C3 블록 단계별 비교
- `tools/compare_sppf_steps.py`: SPPF 블록 단계별 비교
- `tools/dump_golden.py`: PyTorch golden 데이터 생성

### C 구현
- `src/blocks/c3.c`: C3 블록 구현 (디버그 출력 포함)
- `src/blocks/sppf.c`: SPPF 블록 구현 (디버그 출력 포함)
- `src/models/yolov5s_infer.c`: 전체 forward pass

### 디버그 출력
- `debug/pytorch/`: PyTorch 중간 출력
- `debug/c/`: C 구현 중간 출력
- `testdata/python/`: PyTorch 레이어별 출력
- `testdata/c/`: C 구현 레이어별 출력

---

## 결론

체계적인 단계별 디버깅을 통해:
1. ✅ Layer 2 (C3 블록) 문제 해결: cv2에 SiLU activation 추가
2. ✅ Layer 9 (SPPF 블록) 문제 해결: 로직 수정 및 fused BN 처리 추가
3. ✅ 모든 레이어 출력 저장 시스템 구축
4. ✅ 재사용 가능한 디버깅 도구 개발

이 과정을 통해 YOLOv5s C 포팅의 정확성을 높이고, 향후 유사한 문제를 빠르게 해결할 수 있는 방법론을 확립했습니다.
