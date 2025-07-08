## 구조
========================================================================================
모델_실습/                     # 최상위 프로젝트 디렉터리
├── engine.py                 # 공통 학습/검증 루프 정의 (Trainer)
├── train.py                  # CLI 진입점 (도메인, 모델, config 인자 파싱)
├── utils.py                  # 공통 유틸: 데이터로더, 로깅, 시각화 등
├── configs/                  # 하이퍼파라미터·데이터 경로 설정
│   ├── common.yaml           # 전체 공통 설정
│   ├── SR.yaml               # Super-Resolution 전용 설정
│   ├── NLP.yaml              # 자연어처리 전용 설정
│   └── SignalProcessing.yaml # 신호처리 전용 설정
│
├── SR/                       # Super-Resolution 실습 공간
│   ├── models/               # 네트워크 정의만 (forward + 훅 메서드)
│   │   ├── __init__.py
│   │   ├── edsr.py           # EDSR 모델 (compute_loss, configure_optimizers 포함)
│   │   └── srgan.py          # SRGAN 모델 (compute_loss, configure_optimizers 포함)
│   │
│   ├── data/                 # 데이터 다운로드·전처리 스크립트
│   │   └── preprocess.py
│   │
│   └── experiments/          # 각 모델 실행용 래퍼 스크립트
│       ├── run_edsr.py       # train.py 호출, --domain SR --model edsr 지정
│       └── run_srgan.py      # train.py 호출, --domain SR --model srgan 지정
│
├── NLP/                      # 자연어처리 실습 공간 (유사 구조)
│   ├── models/
│   │   └── bert_classifier.py
│   ├── data/
│   │   └── preprocess.py
│   └── experiments/
│       └── run_bert.py
│
└── SignalProcessing/         # 신호처리 실습 공간 (유사 구조)
    ├── models/
    │   └── cnn_eeg.py
    ├── data/
    │   └── preprocess.py
    └── experiments/
        └── run_cnn_eeg.py
========================================================================================