# YOLOv5 C 추론 기대 출력 (Expected Inference Output)

같은 입력(예: bus)과 가중치로 PC/임베디드 추론을 돌렸을 때, 아래와 같은 형식·순서로 출력이 나와야 한다.

---

## 1. Forward pass 및 레이어 통계

```
Running forward pass...
=== Layer stats (chk/min/max/mean) ===
  input: chk=0x41A2EE62 min=0.0000 max=1.0000 mean=0.4555
  Calculated Layer 0 output size: 320x320
  Backbone: Layers 0-9...
  Input size: 640x640
    Layer 0: Conv(3->16, 6x6, s=2, p=2)...
    Layer 0: BN skipped (fused)
  Layer 0: chk=0xF12084B3 min=-0.2785 max=26.0442 mean=1.9060
    Layer 0 completed (485.84 ms)
    Layer 1: Conv(16->32, 3x3, s=2)...
  Layer 1: chk=0x2046B811 min=-0.2785 max=79.2952 mean=2.0127
    Layer 1 completed (330.36 ms)
    Layer 2: C3(64->64, n=1)...
  Layer 2: chk=0x1576A20F min=-0.2785 max=10.0509 mean=0.7094
    Layer 2 completed (219.91 ms)
    Layer 3: Conv(32->64, 3x3, s=2)...
  Layer 3: chk=0x89C498BD min=-0.2785 max=7.9441 mean=0.3819
    Layer 3 completed (316.38 ms)
    Layer 4: C3(64->64, n=2)...
  Layer 4: chk=0xD2D44FE8 min=-0.2785 max=4.6321 mean=0.0916
    Layer 4 completed (386.22 ms)
    Layer 5: Conv(64->128, 3x3, s=2)...
  Layer 5: chk=0x18251B8F min=-0.2785 max=5.0343 mean=0.1082
    Layer 5 completed (311.06 ms)
    Layer 6: C3(128->128, n=3)...
  Layer 6: chk=0xFC55DEE4 min=-0.2785 max=7.7201 mean=0.0819
    Layer 6 completed (496.27 ms)
    Layer 7: Conv(128->256, 3x3, s=2)...
  Layer 7: chk=0x35C8C3A0 min=-0.2785 max=5.7734 mean=-0.0299
    Layer 7 completed (305.64 ms)
    Layer 8: C3(256->256, n=1)...
  Layer 8: chk=0xAF23D813 min=-0.2785 max=5.4823 mean=0.0785
    Layer 8 completed (198.00 ms)
    Layer 9: SPPF(512->512, k=5)...
  Layer 9: chk=0x209E842C min=-0.2785 max=3.0844 mean=-0.1343
    Layer 9 completed (74.69 ms)
  Backbone completed
  Neck: Layers 10-23...
    Layer 10: Conv(256->128, 1x1)...
  Layer 10: chk=0x8E93D76A min=-0.2785 max=4.2127 mean=0.1351
    Layer 10 completed (12.56 ms)
    Layer 11: Upsample(x2)...
  Layer 11: chk=0xA1F78E2D min=-0.2785 max=4.2127 mean=0.1351
    Layer 11 completed (2.30 ms)
    Layer 12: Concat([11, 6])...
  Layer 12: chk=0x9E4D6D12 min=-0.2785 max=7.7201 mean=0.1085
    Layer 12 completed (5.81 ms)
    Layer 13: C3(256->128, n=1, shortcut=False)...
  Layer 13: chk=0x1BA78E1E min=-0.2785 max=3.7610 mean=0.0236
    Layer 13 completed (237.40 ms)
    Layer 14: Conv(128->64, 1x1)...
  Layer 14: chk=0x9638D7A6 min=-0.2785 max=4.4835 mean=0.4551
    Layer 14 completed (12.41 ms)
    Layer 15: Upsample(x2)...
    Layer 15 completed (1.59 ms)
  Layer 15: chk=0x23565674 min=-0.2785 max=4.4835 mean=0.4551
    Layer 16: Concat([15, 4])...
  Layer 16: chk=0xF62AA65C min=-0.2785 max=4.6321 mean=0.2733
    Layer 16 completed (9.20 ms)
    Layer 17: C3(128->64, n=1, shortcut=False)...
    Layer 17 completed (242.36 ms)
  Layer 17: chk=0x57D2C6FF min=-0.2785 max=18.8651 mean=1.1451
    Layer 18: Conv(64->64, 3x3, s=2)...
  Layer 18: chk=0x8A2E3B93 min=-0.2785 max=4.3951 mean=0.1632
    Layer 18 completed (159.15 ms)
    Layer 19: Concat([18, 14])...
  Layer 19: chk=0x2067133A min=-0.2785 max=4.4835 mean=0.3091
    Layer 19 completed (2.56 ms)
    Layer 20: C3(128->128, n=1, shortcut=False)...
    Layer 20 completed (196.68 ms)
  Layer 20: chk=0x9874E5F1 min=-0.2785 max=22.1850 mean=0.5926
    Layer 21: Conv(128->128, 3x3, s=2)...
  Layer 21: chk=0x1F00DD79 min=-0.2785 max=5.9768 mean=0.1038
    Layer 21 completed (157.09 ms)
    Layer 22: Concat([21, 10])...
  Layer 22: chk=0xAD94B4E3 min=-0.2785 max=5.9768 mean=0.1194
    Layer 22 completed (1.90 ms)
    Layer 23: C3(256->256, n=1, shortcut=False)...
    Layer 23 completed (195.27 ms)
  Layer 23: chk=0xD2D26977 min=-0.2785 max=18.3557 mean=0.3677
  Neck completed
  Total forward pass time: 4386.11 ms
Forward pass completed successfully
```

---

## 2. P3 / P4 / P5 출력 shape 및 통계

```
P3 output: (1, 64, 80, 80)
P4 output: (1, 128, 40, 40)
P5 output: (1, 256, 20, 20)
  P3: chk=0x57D2C6FF min=-0.2785 max=18.8651 mean=1.1451
  P4: chk=0x9874E5F1 min=-0.2785 max=22.1850 mean=0.5926
  P5: chk=0xD2D26977 min=-0.2785 max=18.3557 mean=0.3677
```

---

## 3. Detect head 및 디코딩

```
Running Detect head...
Using input size: 640 for detection (from input 640x640)
Detect head completed

Decoding detections...
Found 41 detections (confidence > 0.25)

Running NMS...
After NMS: 4 detections
```

---

## 4. 검출 결과 (Detection Results)

```
=== Detection Results ===
Total detections: 4

Detection 1:
  Class ID: 0
  Confidence: 0.8231
  BBox: (0.3883, 0.5882, 0.1205, 0.4212)
  Pixel coords: x=248.5, y=376.5, w=77.1, h=269.6

Detection 2:
  Class ID: 0
  Confidence: 0.8158
  BBox: (0.2448, 0.5991, 0.1434, 0.4600)
  Pixel coords: x=156.6, y=383.4, w=91.8, h=294.4

Detection 3:
  Class ID: 0
  Confidence: 0.6757
  BBox: (0.8174, 0.5774, 0.1244, 0.4563)
  Pixel coords: x=523.1, y=369.5, w=79.6, h=292.1

Detection 4:
  Class ID: 5
  Confidence: 0.5255
  BBox: (0.5267, 0.4671, 0.7131, 0.4956)
  Pixel coords: x=337.1, y=298.9, w=456.4, h=317.2
```

---

## 검증 시 참고

- **입력·가중치 동일**: 위 출력은 `bus` 이미지 전처리 결과와 YOLOv5n 가중치 기준이다. 다른 이미지/가중치면 chk·min·max·mean·검출 개수 등이 달라질 수 있다.
- **레이어별 검증**: input, Layer 0~23, P3/P4/P5의 **chk**가 PC와 임베디드(UART 등)에서 같으면 해당 텐서는 동일하다고 보면 된다. min/max/mean은 대략적인 분포 확인용이다.
- **검출 개수**: confidence > 0.25 구간에서 41개, NMS 후 4개가 나오는 것이 이 설정·입력 기준 기대값이다.
