# 테스트 가이드

## 통합 테스트 실행

### 1. 사전 준비

```bash
# 1. 모델 가중치 export
python tools/export_yolov5s.py yolov5s.pt --output weights/

# 2. 빌드
mkdir build
cd build
cmake ..
make
```

### 2. 통합 테스트 실행

```bash
cd build
./test_integration
```

또는:

```bash
cd build
ctest
```

### 3. End-to-End 테스트

```bash
# 1. 이미지 전처리
python tools/preprocess.py --image bus.jpg

# 2. 인퍼런스 실행
cd build
yolov5_infer.exe ../data/input_tensor.bin ../weights/weights.bin ../weights/model_meta.json

# 3. 결과 확인
cat ../data/output_detections.txt
```

## 테스트 체크리스트

### 기본 기능 테스트
- [ ] 모델 빌드 성공
- [ ] 가중치 로드 성공
- [ ] Forward pass 실행 성공
- [ ] 출력 텐서 shape 확인
- [ ] Saved features 확인

### 정확도 테스트
- [ ] 골든 참조와 텐서 비교
- [ ] 레이어별 출력 검증
- [ ] 최종 bbox 결과 비교

### 성능 테스트
- [ ] 인퍼런스 시간 측정
- [ ] 메모리 사용량 측정

## 문제 해결

### 모델 빌드 실패
- `weights/weights.bin` 파일이 있는지 확인
- `weights/weights_map.json` 파일이 있는지 확인

### Forward pass 실패
- 입력 텐서 shape 확인 (1, 3, 640, 640)
- 메모리 부족 확인

### 출력이 모두 0
- 가중치 로드 확인
- 레이어 실행 순서 확인
