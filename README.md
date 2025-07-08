# 🤖 모델 실습 프레임워크 (Model Practice Framework)

딥러닝 모델 학습을 위한 범용 프레임워크입니다. 도메인별로 구조화되어 있어 Super-Resolution, NLP, Signal Processing 등 다양한 분야의 모델을 체계적으로 실습할 수 있습니다.

## ✨ 주요 특징

- **🔧 도메인별 구조화**: SR, NLP, SignalProcessing 도메인으로 명확히 분리
- **⚙️ 통합 훈련 엔진**: 모든 모델에 공통으로 사용되는 Trainer 클래스
- **📊 자동 로깅**: TensorBoard를 통한 실시간 훈련 모니터링
- **💾 체크포인트 시스템**: 훈련 중단/재개 지원
- **🎯 Easy Configuration**: YAML 기반 설정 관리
- **🚀 확장성**: 새로운 모델과 도메인 추가 용이

## 📁 프로젝트 구조

```
모델_실습(demo)/
├── engine.py                 # 공통 훈련/검증 엔진 (Trainer 클래스)
├── train.py                  # CLI 진입점 (메인 실행 스크립트)
├── utils.py                  # 공통 유틸리티 함수들
├── configs/                  # 설정 파일들
│   ├── common.yaml           # 공통 설정 (디바이스, 로깅 등)
│   ├── SR.yaml               # Super-Resolution 설정
│   ├── NLP.yaml              # 자연어처리 설정
│   └── SignalProcessing.yaml # 신호처리 설정
│
├── SR/                       # Super-Resolution 도메인
│   ├── models/               # 모델 정의들
│   │   ├── __init__.py
│   │   ├── EDSR.py          # Enhanced Deep Residual Network
│   │   ├── RCAN.py          # Residual Channel Attention Network  
│   │   └── SRGAN.py         # Super-Resolution GAN
│   ├── data/
│   │   └── preprocess.py    # SR 데이터셋 클래스
│   └── experiments/         # 실험 실행 스크립트들
│       ├── run_edsr.py
│       ├── run_rcan.py
│       └── run_srgan.py
│
├── NLP/                     # 자연어처리 도메인 (확장 준비)
│   ├── models/__init__.py
│   ├── data/
│   └── experiments/
│
└── SignalProcessing/        # 신호처리 도메인 (확장 준비)
    ├── models/__init__.py
    ├── data/
    └── experiments/
```

## 🛠️ 설치 및 환경 설정

### 1. 요구사항

```bash
Python >= 3.10
PyTorch >= 1.9.0
torchvision
tensorboard
scikit-image
pillow
pyyaml
```

### 2. 환경 설정

```bash
# 1. 저장소 클론
git clone <repository-url>
cd 모델\ 실습\(demo\)

# 2. 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 의존성 설치
pip install torch torchvision tensorboard scikit-image pillow pyyaml
```

## 🚀 빠른 시작

### 1. SR 모델 훈련 (EDSR 예시)

```bash
# 방법 1: 직접 실행
python train.py --domain SR --model edsr --config configs/SR.yaml

# 방법 2: 실험 스크립트 사용
python SR/experiments/run_edsr.py
```

### 2. 데이터 준비

SR 모델 훈련을 위해 이미지 데이터를 준비합니다:

```
data/
└── DIV2K/           # 또는 원하는 데이터셋 이름
    ├── train/       # 훈련용 고해상도 이미지들
    │   ├── img1.png
    │   ├── img2.jpg
    │   └── ...
    └── val/         # 검증용 고해상도 이미지들
        ├── img1.png
        ├── img2.jpg
        └── ...
```

## 📋 구현된 모델들

### Super-Resolution (SR) 도메인

#### 1. **EDSR (Enhanced Deep Residual Networks)**
- **특징**: 잔차 블록 기반의 단순하고 효과적인 구조
- **설정**: `configs/SR.yaml`의 `edsr` 섹션
- **실행**: `python SR/experiments/run_edsr.py`

#### 2. **RCAN (Residual Channel Attention Networks)**
- **특징**: 채널 어텐션 메커니즘을 활용한 고성능 모델
- **설정**: `configs/SR.yaml`의 `rcan` 섹션  
- **실행**: `python SR/experiments/run_rcan.py`

#### 3. **SRGAN (Super-Resolution GAN)**
- **특징**: GAN 기반의 지각적 품질 향상 모델
- **설정**: `configs/SR.yaml`의 `srgan` 섹션
- **실행**: `python SR/experiments/run_srgan.py`

## ⚙️ 설정 파일 가이드

### configs/SR.yaml 예시

```yaml
# 기본 훈련 설정
epochs: 100
batch_size: 16
device: 'cuda'
data_path: 'data/DIV2K'

# EDSR 모델 설정
edsr:
  n_colors: 3
  n_feats: 64
  kernel_size: 3
  n_resblocks: 16
  scale: 4
  lr: 1e-4

# RCAN 모델 설정
rcan:
  in_channels: 3
  n_feats: 64
  num_rg: 10
  num_rcab: 20
  scale: 4
  lr: 1e-4

# SRGAN 모델 설정
srgan:
  lambda_adv: 1e-3
  lrG: 1e-4
  lrD: 1e-4
```

## 📊 모니터링 및 로깅

### TensorBoard 실행

```bash
# 로그 확인
tensorboard --logdir logs

# 브라우저에서 http://localhost:6006 접속
```

### 체크포인트 관리

```bash
# 체크포인트는 자동으로 checkpoints/ 폴더에 저장됩니다
checkpoints/
├── edsr_epoch50.pth
├── rcan_epoch100.pth
└── srgan_epoch75.pth
```

## 🔧 고급 사용법

### 1. 커스텀 설정으로 실행

```bash
# 사용자 정의 설정 파일 사용
python train.py --domain SR --model edsr --config my_config.yaml
```

### 2. 체크포인트에서 훈련 재개

```python
# train.py 수정 또는 새 스크립트에서
trainer = Trainer(model, dataloaders, cfg)
start_epoch = trainer.load_checkpoint('checkpoints/edsr_epoch50.pth')
trainer.train(num_epochs=100, start_epoch=start_epoch+1)
```

### 3. 새로운 모델 추가

1. **모델 클래스 작성** (`SR/models/new_model.py`)
```python
class NEWMODEL(nn.Module):
    def __init__(self, cfg):
        # 모델 초기화
        pass
    
    def forward(self, x):
        # 순전파 정의
        pass
    
    def compute_loss(self, output, target):
        # 손실 함수 정의
        pass
    
    def configure_optimizers(self, cfg):
        # 옵티마이저 설정
        return [optimizer], [scheduler]
```

2. **__init__.py 업데이트**
```python
from .NEWMODEL import NEWMODEL
__all__ = ['EDSR', 'RCAN', 'SRGAN', 'NEWMODEL']
```

3. **설정 파일에 추가** (`configs/SR.yaml`)
```yaml
newmodel:
  param1: value1
  param2: value2
  lr: 1e-4
```

4. **실험 스크립트 생성** (`SR/experiments/run_newmodel.py`)

## 🔍 트러블슈팅

### 1. CUDA 관련 문제
```
Error: CUDA out of memory
→ batch_size를 줄이거나 configs/SR.yaml에서 device: 'cpu' 설정
```

### 2. 데이터 로딩 문제
```
Error: No images found
→ configs/SR.yaml의 data_path 경로 확인
→ 이미지 파일이 train/, val/ 폴더에 있는지 확인
```

### 3. 모듈 Import 오류
```
ModuleNotFoundError: No module named 'SR.models'
→ 프로젝트 루트 디렉토리에서 실행하는지 확인
→ __init__.py 파일들이 존재하는지 확인
```

## 🎯 성능 지표

훈련 중 모니터링되는 지표들:

- **Loss**: 훈련/검증 손실
- **PSNR**: Peak Signal-to-Noise Ratio (dB)
- **SSIM**: Structural Similarity Index

## 📈 확장 계획

### 단기 목표
- [ ] NLP 도메인 모델 추가 (BERT, GPT 등)
- [ ] SignalProcessing 도메인 모델 추가 (CNN-EEG 등)
- [ ] 추론(inference) 스크립트 추가
- [ ] 모델 비교 및 벤치마크 도구

### 장기 목표
- [ ] Computer Vision 도메인 추가
- [ ] 분산 훈련 지원
- [ ] AutoML 기능 통합
- [ ] 웹 인터페이스 개발

## 📝 라이선스

이 프로젝트는 교육 목적으로 제작되었습니다.

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

📧 **문의사항이 있으시면 언제든 연락주세요!**
