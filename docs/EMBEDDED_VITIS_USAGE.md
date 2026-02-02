# 임베디드에서 _vitis 가중치/맵/메타 사용하기

YOLO_c에서 생성한 **weights_vitis.bin**, **weights_map_vitis.json**, **model_meta_vitis.json**을 임베디드(Vivado/MicroBlaze)에서 쓰는 방법과, PC 코드와의 차이를 정리합니다.

---

## 1. PC vs 임베디드 코드 차이

| 항목 | PC (YOLO_c) | 임베디드 (yolov5_ver2) |
|------|-------------|-------------------------|
| **weights_loader** | `weights_loader_create(weights_path)` — 파일 경로로 **파일에서** bin + map 로드 | `weights_loader_create_from_mem(weights_data, weights_size, map_data, map_size)` — **메모리 포인터**로 bin + map 사용 |
| **weights 구조체** | `map_path` (파일 경로), `data`/`size` (파일 읽어서 할당) | `map_data`/`map_size` (맵 JSON 문자열), `data`/`size` (가중치 버퍼, 복사 없음) |
| **모델 빌드** | `yolov5n_build(weights_path, model_meta_path)` | `yolov5n_build_from_mem(weights_data, weights_size, weights_map_json, weights_map_size, model_meta_json, model_meta_size)` |
| **Conv+BN 로드** | PC는 fused 형식만 사용 → `conv.bias` 필수 | **conv.bias는 `weights_loader_get_optional`**, BN은 `bn.bias`/`running_mean`/`running_var`/`bn.weight`로 로드 → **BN 분리(_vitis) 형식 지원** |

요약: 임베디드는 **파일이 아니라 메모리**에서 가중치/맵을 받고, **conv.bias 없이 BN만 있는 형식**을 이미 지원합니다.

---

## 2. _vitis 형식과의 호환성

- **_vitis**는 **BN 분리** 형식:  
  `model.X.bn.bias`, `model.X.bn.running_mean`, `model.X.bn.running_var`, `model.X.bn.weight`, `model.X.conv.weight`  
  (conv.bias 없음)
- 임베디드 쪽은 이미:
  - **weights_loader_get_optional**로 `conv.bias` 조회 → _vitis에는 없으므로 NULL
  - **weights_loader_get**으로 `bn.bias`, `bn.running_mean`, `bn.running_var`, `bn.weight`, `conv.weight` 조회 → _vitis에 모두 있음
- 따라서 **weights_loader.c**, **yolov5n_build.c**, **c3.c**, **sppf.c**, **bottleneck.c** 모두 **수정 없이** _vitis 가중치/맵을 사용할 수 있습니다.

---

## 3. 임베디드에서 해야 할 일 (코드 수정 없음)

1. **사용할 파일**
   - **weights_vitis.bin** (7,526,644 bytes)
   - **weights_map_vitis.json** (내용 전체를 메모리/플래시에 보관)

2. **호출 방법**
   - 가중치 바이너리를 DDR(또는 사용하는 버퍼)에 로드한 뒤, 그 **포인터**와 **크기(7526644)** 를 준비.
   - weights_map_vitis.json **문자열**의 **시작 주소**와 **길이(바이트)** 를 준비.
   - 아래처럼 한 번만 호출하면 됩니다.

   ```c
   // 예: weights_ptr = DDR에 로드한 weights_vitis.bin, map_ptr = weights_map_vitis.json 문자열
   model = yolov5n_build_from_mem(
       weights_ptr, 7526644,
       map_ptr, map_size,   // weights_map_vitis.json
       NULL, 0              // model_meta는 현재 미사용 가능
   );
   ```

3. **파일을 어디서 읽는지**
   - SD 카드, 플래시, 초기화 시 복사 등 **기존에 weights.bin / weights_map.json을 로드하던 경로**에서  
     **weights_vitis.bin**과 **weights_map_vitis.json**으로 **파일만 바꿔서** 같은 버퍼에 넣어 주면 됩니다.
   - C 소스(weights_loader.c, yolov5n_build.c 등)는 **수정할 필요 없습니다**.

---

## 4. model_meta_vitis.json 사용 (선택)

- 현재 임베디드 `yolov5n_build_from_mem`은 **model_meta_json / model_meta_size**를 받지만 **사용하지 않습니다** (주석: unused parameter).
- **depth_multiple**, **width_multiple**, **channels** 등은 코드에 **하드코딩**(0.33f, 0.25f, get_actual_channels 등)되어 있어서, _vitis와 이미 일치합니다.
- **model_meta_vitis.json**을 쓰고 싶은 경우 예:
  - **total_weights_size**(7526644)로 전달한 `weights_size` 검증
  - 나중에 메타에서 depth/width/channels 읽어서 동적으로 설정
- 이때는 **model_meta JSON을 파싱하는 코드**가 추가로 필요합니다.  
  필요하면 "model_meta 파싱용 코드(예: jsmn으로 total_weights_size, depth_multiple 등만 읽는 함수)"를 요청하면 해당 코드를 작성해 드리겠습니다.

---

## 5. 요약

| 질문 | 답변 |
|------|------|
| 가중치/맵/메타 형식 맞추려고 **임베디드 C 코드를 수정해야 하나?** | **아니요.** weights_loader.c, yolov5n_build.c, c3/sppf/bottleneck 모두 그대로 두고 사용 가능합니다. |
| 해야 할 일 | **weights_vitis.bin**(7526644 bytes)과 **weights_map_vitis.json** 내용을 메모리에 넣고, `yolov5n_build_from_mem(weights_ptr, 7526644, map_ptr, map_size, NULL, 0)` 호출. 기존에 weights.bin / weights_map.json 쓰던 경로만 _vitis 파일로 바꾸면 됩니다. |
| model_meta_vitis.json은? | 현재는 사용하지 않아도 됩니다. 나중에 검증/설정용으로 쓰려면 **model_meta 파싱 코드**가 필요하며, 필요 시 그 코드를 요청하면 됩니다. |

추가로 필요한 코드(예: model_meta 파싱, weights_size 검증용 래퍼)가 있으면 "model_meta 파싱 코드 달라"처럼 구체적으로 요청해 주세요.

---

## 6. 추가 점검 사항 (체크리스트)

_vitis로 전환 후 한 번만 확인하면 좋은 항목입니다.

| 항목 | 권장 값 / 확인 내용 |
|------|----------------------|
| **가중치 버퍼 크기** | `weights_vitis.bin` = **7,526,644 bytes**. DDR(또는 로드 버퍼)가 이 크기 이상인지 확인. |
| **맵 버퍼 크기** | `weights_map_vitis.json` ≈ **43KB**. 맵을 올리는 메모리/플래시 버퍼가 이보다 크거나 같아야 함. |
| **맵 파싱 한계 (임베디드)** | `weights_loader.c`에 `WEIGHTS_MAP_MAX_ENTRIES 512`, `WEIGHTS_MAP_MAX_TOKENS 6144`. _vitis는 **291개 엔트리**라서 둘 다 여유 있음. 수정 불필요. |
| **맵 데이터 종료** | `parse_weights_map`은 `map_data` 안에서 **첫 `\0` 전까지** 또는 `map_size`까지만 파싱. 맵을 버퍼에 넣을 때: **끝에 `\0`을 넣거나**, `map_size`를 **정확한 바이트 수**로 넘기면 됨. |
| **엔디언** | `weights_vitis.bin`은 **float32 little-endian**. MicroBlaze/ARM이 little-endian이면 그대로 사용. big-endian이면 바이트 스왑이 필요할 수 있음. |
| **로더 재사용** | 가중치/맵을 **다른 파일로 바꿀 때**는 반드시 **새 버퍼**로 `weights_loader_create_from_mem`을 다시 호출해야 함. 같은 로더를 재사용하면 내부 캐시가 이전 맵을 가리킬 수 있음. |
| **jsmn** | 임베디드용 jsmn과 PC용 jsmn은 **동일**하다고 확인됨. _vitis 맵 파싱에 그대로 사용 가능. |

위 항목만 맞으면 C 소스 수정 없이 _vitis 가중치/맵을 임베디드에서 사용할 수 있습니다.

---

## 7. _vitis 파일에 맞게 수정할 위치

**weights_loader.c / yolov5n_build.c 는 건드리지 않습니다.**  
아래는 **가중치·맵을 읽어서 메모리에 넣고, `yolov5n_build_from_mem`을 호출하는 쪽**에서만 바꾸면 됩니다.

| 수정할 곳 | 기존(예시) | _vitis에 맞게 |
|-----------|------------|----------------|
| **가중치 파일 경로/이름** | `weights.bin` 또는 `weights_fused.bin` | **`weights_unfused.bin`** |
| **가중치 크기(weights_size)** | `7469620` (fused) 또는 다른 값 | **`7526644`** (반드시 이 값) |
| **맵 파일 경로/이름** | `weights_map.json` 또는 `weights_map_fused.json` | **`weights_map_unfused.json`** |
| **맵 버퍼 크기(map_size)** | 이전 맵 파일 크기 | **weights_map_unfused.json 실제 바이트 수** (약 43KB, 파일/버퍼 크기로 확인) |

### 구체적으로 찾을 위치

1. **SD/플래시에서 파일을 열거나, 바이너리를 DDR로 복사하는 코드**  
   - `fopen("weights.bin", "rb")` / `open("weights_map.json", ...)` 같은 부분  
   - → 파일 이름을 **`weights_unfused.bin`**, **`weights_map_unfused.json`** 으로 변경.

2. **`yolov5n_build_from_mem`을 호출하는 코드**  
   - 예: `yolov5n_build_from_mem(weights_ptr, ???, map_ptr, ???, NULL, 0);`  
   - → 두 번째 인자(weights 크기)를 **`7526644`** 로 설정.  
   - → 네 번째 인자(map 크기)를 **weights_map_unfused.json을 읽은 버퍼의 실제 바이트 수**로 설정.

3. **가중치/맵 크기를 상수로 둔 경우**  
   - `#define WEIGHTS_SIZE 7469620` 또는 `const size_t weights_size = ...`  
   - → **`7526644`** 로 변경.  
   - 맵 크기 상수도 **weights_map_unfused.json 크기**로 변경.

4. **부트로더/스크립트에서 이미지에 넣는 파일 이름**  
   - BSP나 플래시 이미지에 `weights.bin`, `weights_map.json`을 넣는 경우  
   - → **`weights_unfused.bin`**, **`weights_map_unfused.json`** 으로 교체.

정리: **yolov5/weights_loader, yolov5n_build 등 모델/로더 소스는 수정하지 않고**,  
**파일 이름·경로·크기(7526644, 맵 크기)만** unfused 파일에 맞게 위 위치에서 수정하면 됩니다.
