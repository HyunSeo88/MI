# Super Resolution Training Framework

깔끔하고 모듈화된 Super Resolution 모델 학습 프레임워크입니다.

## 📁 프로젝트 구조

```
Experiments/
├── data/                 # 데이터셋 (DIV2K)
├── results/              # 학습 결과물들
├── test_div2k.ipynb     # 데이터셋 테스트 노트북
├── dataset.py           # SR 데이터셋 로더
├── model_loader.py      # 모델 로더
├── engine.py            # 학습/평가 엔진
├── utils.py             # 유틸리티 함수들
├── config.py            # 설정 관리
├── train.py             # 메인 학습 스크립트
└── README.md            # 사용 설명서
```

## 🚀 빠른 시작

### 1. 기본 EDSR 모델 학습
```bash
python train.py --model EDSR --scale 2 --epochs 50
```

### 2. RCAN 모델로 4배 확대 학습
```bash
python train.py --model RCAN --scale 4 --batch-size 8 --epochs 100
```

### 3. 커스텀 설정으로 학습
```bash
python train.py \
    --model EDSR \
    --scale 2 \
    --batch-size 16 \
    --epochs 200 \
    --lr 1e-5 \
    --optimizer AdamW \
    --scheduler CosineAnnealingLR \
    --loss L1 \
    --experiment-name "EDSR_experiment_v1"
```

### 4. 훈련 재개
```bash
python train.py --resume-from ./results/EDSR_x2_20231201_120000/checkpoints/last.pth
```

## ⚙️ 주요 매개변수

### 모델 관련
- `--model`: 모델 종류 (EDSR, RCAN, ESRGAN, SwinIR)
- `--scale`: 확대 비율 (2, 3, 4, 8)
- `--model-params`: JSON 형태의 모델 특수 파라미터

### 데이터 관련
- `--data-path`: 데이터셋 경로 (기본: ./data/DV2K)
- `--patch-size`: HR 패치 크기 (기본: 96)
- `--batch-size`: 배치 크기 (기본: 16)
- `--num-workers`: 데이터 로더 워커 수 (기본: 0)

### 학습 관련
- `--epochs`: 학습 에포크 수 (기본: 100)
- `--lr`: 학습률 (기본: 1e-4)
- `--weight-decay`: 가중치 감쇠 (기본: 1e-4)
- `--loss`: 손실 함수 (L1, L2, MSE, Huber)

### 옵티마이저 관련
- `--optimizer`: 옵티마이저 (Adam, AdamW, SGD)
- `--scheduler`: 스케줄러 (CosineAnnealingLR, StepLR, MultiStepLR, None)

### 기타
- `--device`: 디바이스 (auto, cpu, cuda)
- `--experiment-name`: 실험 이름
- `--save-every`: 체크포인트 저장 주기 (기본: 10 에포크)
- `--eval-every`: 평가 주기 (기본: 1 에포크)

## 📊 지원되는 모델

### 1. EDSR (Enhanced Deep Residual Networks)
```bash
python train.py --model EDSR --scale 2
```

**기본 파라미터:**
- n_colors: 3
- n_feats: 64
- kernel_size: 3
- n_resblocks: 16
- res_scale: 1.0

### 2. RCAN (Residual Channel Attention Networks)
```bash
python train.py --model RCAN --scale 2
```

**기본 파라미터:**
- n_colors: 3
- n_feats: 64
- kernel_size: 3
- n_resblocks: 20

### 3. 커스텀 모델 파라미터
```bash
python train.py \
    --model EDSR \
    --model-params '{"n_feats": 128, "n_resblocks": 32}'
```

## 📈 학습 모니터링

### 1. 로그 확인
```bash
tail -f ./results/EDSR_x2_20231201_120000/logs/training.log
```

### 2. 메트릭 확인
학습 완료 후 다음 파일들이 생성됩니다:
- `metrics.csv`: 에포크별 메트릭 데이터
- `training_curves.png`: 학습 곡선 그래프
- `config.json`: 사용된 설정

### 3. 샘플 이미지 확인
`./results/{experiment}/images/` 폴더에서 10 에포크마다 생성되는 샘플 이미지들을 확인할 수 있습니다.

## 🔧 문제 해결

### 1. CUDA 메모리 부족
```bash
# 배치 크기 감소
python train.py --batch-size 8

# 패치 크기 감소
python train.py --patch-size 64
```

### 2. Windows 환경에서 multiprocessing 오류
```bash
# num_workers를 0으로 설정 (기본값)
python train.py --num-workers 0
```

### 3. 데이터셋 경로 문제
```bash
# 절대 경로 사용
python train.py --data-path "C:/path/to/your/DIV2K"
```

## 📝 사용 예시

### 실험 1: 기본 EDSR 학습
```bash
python train.py \
    --model EDSR \
    --scale 2 \
    --epochs 100 \
    --experiment-name "EDSR_baseline"
```

### 실험 2: 큰 RCAN 모델 학습
```bash
python train.py \
    --model RCAN \
    --scale 4 \
    --batch-size 8 \
    --epochs 200 \
    --lr 5e-5 \
    --model-params '{"n_feats": 128}' \
    --experiment-name "RCAN_large_x4"
```

### 실험 3: 빠른 프로토타이핑
```bash
python train.py \
    --model EDSR \
    --epochs 10 \
    --eval-every 2 \
    --save-every 5 \
    --experiment-name "quick_test"
```

## 📂 결과물 구조

학습 완료 후 다음과 같은 구조로 결과물이 저장됩니다:

```
results/EDSR_x2_20231201_120000/
├── checkpoints/
│   ├── best.pth          # 최고 성능 모델
│   ├── last.pth          # 마지막 에포크 모델
│   └── epoch_*.pth       # 정기 체크포인트
├── images/
│   └── samples_epoch_*.png  # 샘플 이미지들
├── logs/
│   └── training.log      # 학습 로그
├── config.json           # 사용된 설정
├── metrics.csv           # 메트릭 데이터
└── training_curves.png   # 학습 곡선
```

## 🔄 모델 로딩 및 추론

```python
import torch
from model_loader import create_model
from utils import load_model

# 모델 생성
model = create_model('EDSR', scale=2)

# 체크포인트 로드
load_model('./results/EDSR_x2_20231201_120000/checkpoints/best.pth', model)

# 추론 모드
model.eval()
with torch.no_grad():
    sr_img = model(lr_img)
```

## 🆘 도움말

모든 사용 가능한 옵션 확인:
```bash
python train.py --help
```

## 📊 성능 메트릭

이 프레임워크는 다음 메트릭들을 자동으로 계산합니다:
- **Loss**: 학습 손실 (L1, L2, MSE, Huber)
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index

## 🎯 Best Practices

1. **시작은 작은 모델과 짧은 에포크로**: 파이프라인이 정상 작동하는지 확인
2. **배치 크기 조정**: GPU 메모리에 맞게 조정
3. **정기적인 체크포인트 저장**: `--save-every`로 주기 설정
4. **실험 이름 지정**: 여러 실험을 구분하기 위해 명확한 이름 사용
5. **학습률 스케줄링**: 대부분의 경우 CosineAnnealingLR이 좋은 성능

---

**Happy Training! 🚀** 