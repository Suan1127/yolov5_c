# Fused(weights_fused) vs Unfused(weights_unfused) 결과가 다를 때

같은 yolov5n.pt에서 export했는데 **model_meta_fused**(fused)으로 돌렸을 때와 **unfused**로 돌렸을 때 결과가 다르게 나오는 이유와 점검 방법을 정리합니다.

---

## 1. 두 가지 export 방식 차이

| 항목 | Fused (weights_fused.bin) | Unfused (weights_unfused.bin) |
|------|-----------------------------|-----------------------------|
| **로딩** | `attempt_load(..., fuse=True)` | `attempt_load(..., fuse=False)` |
| **state_dict** | Conv+BN이 **이미 합쳐진** conv.weight, conv.bias | **원본** conv.weight + bn.bias, bn.running_mean, bn.running_var, bn.weight |
| **bin 내용** | W_fused = W×(γ/√(var+ε)), b_fused = β − μ×γ/√(var+ε) | W 그대로, BN 4개 텐서 별도 |
| **수학** | y = W_fused·x + b_fused → SiLU(y) | y = W·x → BN(y) = (y−μ)×γ/√(var+ε)+β → SiLU |

즉, **같은 모델(.pt)에서 나왔지만 저장 형식만 다릅니다.**  
수식으로 보면 **fused 한 번에 계산**과 **Conv → BN 순서대로 계산**(unfused)은 동일한 식이므로, **같은 입력이면 같은 출력**이 나와야 합니다.

---

## 2. 왜 결과가 “다르게” 보일 수 있는가

### (1) 바이너리 값 자체는 다름 (정상)

- Fused: **이미 BN이 섞인** conv.weight / conv.bias만 저장.
- Unfused: **원본** conv.weight + BN 4종 저장.
- 그래서 **파일 내용(숫자)은 당연히 다르고**, 그 차이 때문에 **중간 레이어 통계**(chk, head hex, min/max/mean)도 조금씩 다를 수 있습니다.  
→ **중간값이 조금 다르다고 해서 “틀린 것”은 아닙니다.**

### (2) 부동소수 연산 순서

- Fused: `(W_fused·x + b_fused)` 한 번에 계산.
- Unfused: `W·x` → 그다음 BN 식 적용.
- 연산 순서가 다르면 **소수점 아래에서 미세한 차이**가 나는 건 정상입니다.  
→ **최종 검출(개수, 클래스, bbox)이 같으면 “같은 결과”로 봐도 됩니다.**

### (3) 실제로 다른 경우 (점검 필요)

- **최종 검출 결과**가 다름 (개수, 클래스, 박스가 확 달라짐).
- 또는 **Layer 0 한 픽셀 silu_out**을 fused vs unfused로 비교했을 때 **크게** 다름.

이럴 때는 아래를 의심할 수 있습니다.

1. **같은 .pt에서 export했는지**
   - weights_fused.bin은 **fuse=True**로 export한 스크립트(예: export_yolov5s.py에 yolov5n.pt 넣은 경우 등)로 만들었는지.
   - weights_unfused.bin은 **export_yolov5n_vitis.py** (suffix=unfused)로 **같은 yolov5n.pt**에서 만들었는지.
   - 서로 다른 .pt 또는 다른 옵션으로 만들었으면 결과가 달라질 수 있습니다.

2. **C 쪽에서 쓰는 가중치**
   - Fused: `weights_fused.bin` + `weights_map_fused.json` (conv.bias, conv.weight만 있는 맵).
   - Unfused: `weights_unfused.bin` + `weights_map_unfused.json` (BN 분리 맵).
   - 맵/경로를 바꿨을 때 **실제로 로드되는 파일**이 맞는지 확인 (같은 이름이어도 이전 빌드 캐시 등으로 옛날 파일이 로드되진 않는지).

3. **BN 수식/eps**
   - C 쪽 BN: `(x - mean) * (gamma / sqrt(var + eps)) + beta`, `eps=1e-5`.
   - PyTorch fuse도 동일한 eps를 쓰면 이론상 동일.  
   → 차이가 크면 eps나 수식이 한쪽에서 다르게 적용됐는지 확인할 필요가 있음.

---

## 3. 점검 방법 (같은 가중치인데 결과가 다를 때)

1. **동일 .pt에서 두 형식 다시 export**
   - 같은 yolov5n.pt로  
     - fused: `fuse=True` 로 weights_fused.bin + weights_map_fused.json 생성  
     - unfused: `export_yolov5n_vitis.py --suffix unfused` 로 weights_unfused.bin + weights_map_unfused.json 생성  
   - 그 상태에서 각각 한 번씩만 추론해서 비교.

2. **Layer 0 한 픽셀 비교**
   - 같은 입력(bus.bin 등)으로  
     - fused 설정으로 한 번 돌리고 Layer 0 (b=0, oc=0, oh=0, ow=0) **silu_out** 값 기록  
     - unfused 설정으로 한 번 돌리고 동일 위치 **silu_out** 기록  
   - 두 값이 **거의 같으면** (오차 1e-4 이하): 수식/구현은 맞고, 이후 레이어나 후처리 쪽만 보면 됨.  
   - **크게 다르면**: Conv/BN 로딩 또는 BN 적용 경로에 버그 가능성.

3. **최종 검출만 비교**
   - 검출 개수, 클래스 ID, confidence, bbox를 fused vs unfused로 비교.  
   - 여기서 **같거나 거의 같으면** “같은 가중치로 수행한 결과”로 보면 됨.

---

## 4. 요약

- **같은 가중치 파일(yolov5n.pt)을 export**했지만,  
  - **fused** = BN이 이미 합쳐진 conv weight/bias만 저장 (weights_fused.bin)  
  - **unfused** = 원본 conv weight + BN 파라미터 분리 저장 (weights_unfused.bin)  
  이라서 **디스크에 있는 숫자(바이너리)는 당연히 다릅니다.**

- 수학적으로는 **동일 연산**이므로, **같은 입력이면 출력이 같아야** 하고,  
  실제로도 **최종 검출이 같으면** “결과가 같다”고 보면 됩니다.

- **중간 레이어 값**(chk, head, 통계)이 조금 다르게 나오는 것은 **저장 형식 차이 + 부동소수 순서** 때문에 흔히 있을 수 있는 일이며, 그 자체로 “틀린 것”은 아닙니다.

- **최종 검출이 다르다**면:  
  (1) 두 bin이 **정말 같은 yolov5n.pt**에서 나왔는지,  
  (2) **Layer 0 silu_out**은 fused/unfused가 거의 같은지  
  부터 확인하는 것을 권장합니다.
