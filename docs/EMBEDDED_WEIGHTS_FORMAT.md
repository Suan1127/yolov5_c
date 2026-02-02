# 임베디드 vs PC 가중치/맵 형식 차이 (모델 구현 실패 시 참고)

임베디드에서 **PC와 동일한** weight, weights_map, model_meta를 쓰면 “모델 구현이 안 된다”는 경우, 대부분 **weights_map 키와 bin 레이아웃**이 달라서입니다.

---

## 1. model_meta 차이

| 항목 | 임베디드(old) | PC (YOLOv5n) |
|------|----------------|--------------|
| **width_multiple** | **0.5** | **0.25** |
| **channels.actual** | [32, 64, 128, 256, 512] | [16, 32, 64, 128, 256] |
| **total_weights_size** | 7,526,644 | 7,469,620 |
| **num_parameters** | 291 | 120 |

→ old는 **더 큰 모델**(YOLOv5s급 채널), PC는 **YOLOv5n**(작은 채널).  
→ **PC 것을 쓸 때는 반드시 PC model_meta**를 같이 써야 하고, **채널 수가 16,32,64,128,256**으로 맞아야 합니다.

---

## 2. weights_map / bin 레이아웃 차이 (핵심)

### 2.1 임베디드(old) 형식 — Conv와 BN **분리**

- **키**: `model.0.bn.bias`, `model.0.bn.running_mean`, `model.0.bn.running_var`, `model.0.bn.weight`, `model.0.conv.weight`
- **bin 순서**:  
  `[bn.bias] [bn.running_mean] [bn.running_var] [bn.weight] [conv.weight]`  
- Conv는 **bias 없음**(또는 0). BN은 **별도 4개 텐서**로 존재.

예 (Layer 0):

- model.0.bn.bias       offset 0
- model.0.bn.running_mean offset 64
- model.0.bn.running_var  offset 128
- model.0.bn.weight     offset 192
- model.0.conv.weight   offset 256

### 2.2 PC(YOLO_c) 형식 — BN **융합** (fused)

- **키**: `model.0.conv.bias`, `model.0.conv.weight` 만 있음.  
  **`model.0.bn.*` 없음.**
- **bin 순서**:  
  `[conv.bias] [conv.weight]`  
  여기서 `conv.bias` = **BN을 Conv bias에 융합한 값**.
- BN은 파일에 **저장되지 않음**. 추론 시 “BN 융합된 Conv”만 사용.

예 (Layer 0):

- model.0.conv.bias   offset 0
- model.0.conv.weight offset 64

---

## 3. “PC 걸로 바꿨는데 모델 구현이 안 된다” 이유

임베디드 쪽 코드가 **old 형식에 맞춰** 작성되어 있을 때:

1. **model.0.bn.bias** 같은 키를 찾는데, PC weights_map에는 **없음** →  
   키 없음/오프셋 조회 실패로 빌드 또는 로드 실패.
2. **model.0.conv.weight** 오프셋을 old처럼 256으로 기대하는데, PC에서는 64 →  
   잘못된 위치에서 읽어서 모델이 깨짐.
3. **채널 수**는 model_meta의 `channels.actual` / width_multiple로 결정되는데,  
   PC model_meta를 안 쓰면 32,64,128,… 그대로 쓰다가 shape 불일치로 실패.

그래서 **파일만 PC 걸로 바꾸고, 로더/빌드 로직은 old 기준**이면 “모델 구현이 안 된다”가 나옵니다.

---

## 4. 해결 방향: 임베디드에서 PC weight/model 쓰려면

- **쓸 파일**
  - weight: **PC** `weights_fused.bin` (7,469,620 bytes)
  - weights_map: **PC** `weights_map_fused.json`
  - model_meta: **PC** `model_meta_fused.json`

- **수정할 것 (임베디드 쪽)**

1. **weights_map 해석**
   - Conv+BN 레이어에서  
     - **PC 형식**: `model.X.conv.bias`, `model.X.conv.weight` 만 있는지 먼저 확인.
   - 이 경우:
     - `model.X.conv.bias` → Conv의 bias로 로드.
     - `model.X.conv.weight` → Conv weight로 로드.
     - **`model.X.bn.*` 는 찾지 않음** (없으면 스킵).  
       → 이 레이어는 “BN 융합 Conv”로 취급 (BN 파라미터 로드 안 함).
   - (선택) old 형식도 유지하려면:  
     `model.X.bn.bias` 등이 **있을 때만** BN을 따로 로드하고, **없으면** “이 레이어는 fused”로 처리.

2. **오프셋 사용**
   - 반드시 **현재 사용 중인 weights_map.json**의 `offset` 값을 그대로 사용.  
     PC map이면 conv.weight 오프셋이 64처럼 작은 값이 정상.

3. **model_meta**
   - **PC model_meta**를 쓰면 `width_multiple: 0.25`, `channels.actual: [16,32,64,128,256]`.
   - 빌드/초기화 시 **이 채널 수**로 텐서/레이어 shape를 잡아야 함.  
     old(0.5) 기준 32,64,128,… 로 하면 shape/파라미터 수가 맞지 않아 구현 실패할 수 있음.

4. **정리**
   - “weight, weight map, model을 모두 PC에서 쓰던 걸로 바꿨는데 구현이 안 된다”  
     → **weights_map 키(BN 분리 vs 융합)와 오프셋을 PC 형식에 맞게** 로더를 수정하고,  
     **model_meta도 PC 걸로** 써서 채널 수를 16,32,64,128,256으로 맞추면 됩니다.

---

## 5. 요약 표

| 항목 | 임베디드(old) | PC (YOLOv5n) |
|------|----------------|--------------|
| width_multiple | 0.5 | 0.25 |
| 채널(actual) | 32,64,128,256,512 | 16,32,64,128,256 |
| Layer0 in map | bn.bias, bn.running_mean, bn.running_var, bn.weight, conv.weight | conv.bias, conv.weight |
| Layer0 bin | BN 4개 + conv.weight | conv.bias(융합) + conv.weight |
| 모델 구현 성공 조건 | old map + old bin + old meta | **PC map + PC bin + PC meta** + **로더가 “conv.bias/conv.weight만 있는 레이어”를 fused로 처리** |

PC와 동일한 weight, weights_map, model을 쓰려면 **로더가 PC 형식(BN 융합, 키 이름, 오프셋)**을 지원하도록 위처럼 바꾸면 됩니다.
