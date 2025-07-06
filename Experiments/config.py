"""
Configuration Management for Super Resolution Training
SR 학습을 위한 설정 관리
"""

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Optional, List


@dataclass
class TrainingConfig:
    """학습 설정"""
    # 모델 설정
    model_name: str = 'EDSR'
    scale: int = 2
    
    # 데이터 설정
    data_path: str = './data/DIV2K'
    patch_size: int = 96
    batch_size: int = 16
    num_workers: int = 0  # Windows 호환성을 위해 0으로 기본 설정
    
    # 학습 설정
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    loss_type: str = 'L1'  # L1, L2, MSE, Huber
    
    # 옵티마이저 설정
    optimizer: str = 'Adam'  # Adam, AdamW, SGD
    scheduler: str = 'CosineAnnealingLR'  # CosineAnnealingLR, StepLR, MultiStepLR, None
    
    # 메트릭 설정
    use_psnr: bool = True
    use_ssim: bool = True
    
    # 기타 설정
    device: str = 'auto'  # auto, cpu, cuda
    experiment_name: Optional[str] = None
    resume_from: Optional[str] = None
    save_every: int = 10  # 체크포인트 저장 주기
    eval_every: int = 1   # 평가 주기
    
    # 로깅 설정
    log_level: str = 'INFO'
    save_images: bool = True
    save_curves: bool = True
    
    # 모델별 특수 설정
    model_params: dict = None
    
    def __post_init__(self):
        if self.model_params is None:
            self.model_params = {}
        
        if self.experiment_name is None:
            self.experiment_name = f"{self.model_name}_x{self.scale}"


def get_default_model_params(model_name: str, scale: int = 2) -> dict:
    """
    모델별 기본 파라미터 반환
    
    Args:
        model_name (str): 모델 이름
        scale (int): 확대 비율
        
    Returns:
        dict: 기본 파라미터
    """
    params = {
        'EDSR': {
            'n_colors': 3,
            'n_feats': 64,
            'kernel_size': 3,
            'n_resblocks': 16,
            'scale': scale,
            'res_scale': 1.0
        },
        'RCAN': {
            'n_colors': 3,
            'n_feats': 64,
            'kernel_size': 3,
            'n_resblocks': 20,
            'scale': scale
        },
        'ESRGAN': {
            'n_colors': 3,
            'n_feats': 64,
            'kernel_size': 3,
            'n_basic_blocks': 23,
            'scale': scale
        },
        'SwinIR': {
            'img_size': 64,
            'patch_size': 1,
            'in_chans': 3,
            'embed_dim': 96,
            'depths': [6, 6, 6, 6],
            'num_heads': [6, 6, 6, 6],
            'window_size': 7,
            'mlp_ratio': 4.0,
            'upscale': scale
        }
    }
    
    return params.get(model_name, {'scale': scale})


def create_argument_parser():
    """
    명령행 인수 파서 생성
    
    Returns:
        argparse.ArgumentParser: 파서
    """
    parser = argparse.ArgumentParser(
        description='Super Resolution Model Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 모델 관련
    parser.add_argument('--model', type=str, default='EDSR',
                       choices=['EDSR', 'RCAN', 'ESRGAN', 'SwinIR'],
                       help='Model architecture')
    parser.add_argument('--scale', type=int, default=2,
                       choices=[2, 3, 4, 8],
                       help='Super resolution scale factor')
    
    # 데이터 관련
    parser.add_argument('--data-path', type=str, default='./data/DIV2K',
                       help='Path to dataset')
    parser.add_argument('--patch-size', type=int, default=96,
                       help='HR patch size for training')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='Number of data loading workers')
    
    # 학습 관련
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', '--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--loss', type=str, default='L1',
                       choices=['L1', 'L2', 'MSE', 'Huber'],
                       help='Loss function')
    
    # 옵티마이저 관련
    parser.add_argument('--optimizer', type=str, default='Adam',
                       choices=['Adam', 'AdamW', 'SGD'],
                       help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='CosineAnnealingLR',
                       choices=['CosineAnnealingLR', 'StepLR', 'MultiStepLR', 'None'],
                       help='Learning rate scheduler')
    
    # 메트릭 관련
    parser.add_argument('--no-psnr', action='store_true',
                       help='Disable PSNR calculation')
    parser.add_argument('--no-ssim', action='store_true',
                       help='Disable SSIM calculation')
    
    # 기타
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Resume training from checkpoint')
    parser.add_argument('--save-every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--eval-every', type=int, default=1,
                       help='Evaluate every N epochs')
    
    # 로깅 관련
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--no-save-images', action='store_true',
                       help='Disable saving sample images')
    parser.add_argument('--no-save-curves', action='store_true',
                       help='Disable saving training curves')
    
    # 모델별 파라미터 (JSON 형태로 입력)
    parser.add_argument('--model-params', type=str, default=None,
                       help='Model parameters as JSON string')
    
    return parser


def parse_config_from_args(args) -> TrainingConfig:
    """
    명령행 인수로부터 설정 객체 생성
    
    Args:
        args: argparse 결과
        
    Returns:
        TrainingConfig: 설정 객체
    """
    # 모델별 기본 파라미터 가져오기
    model_params = get_default_model_params(args.model, args.scale)
    
    # 사용자 정의 파라미터가 있다면 오버라이드
    if args.model_params:
        try:
            custom_params = json.loads(args.model_params)
            model_params.update(custom_params)
        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON in --model-params: {args.model_params}")
    
    config = TrainingConfig(
        model_name=args.model,
        scale=args.scale,
        data_path=args.data_path,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        loss_type=args.loss,
        optimizer=args.optimizer,
        scheduler=args.scheduler if args.scheduler != 'None' else None,
        use_psnr=not args.no_psnr,
        use_ssim=not args.no_ssim,
        device=args.device,
        experiment_name=args.experiment_name,
        resume_from=args.resume_from,
        save_every=args.save_every,
        eval_every=args.eval_every,
        log_level=args.log_level,
        save_images=not args.no_save_images,
        save_curves=not args.no_save_curves,
        model_params=model_params
    )
    
    return config


def save_config(config: TrainingConfig, filepath: str):
    """
    설정을 JSON 파일로 저장
    
    Args:
        config (TrainingConfig): 설정 객체
        filepath (str): 저장 경로
    """
    config_dict = asdict(config)
    
    # 디렉토리 생성
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)
    
    print(f"Config saved to: {filepath}")


def load_config(filepath: str) -> TrainingConfig:
    """
    JSON 파일로부터 설정 로드
    
    Args:
        filepath (str): 설정 파일 경로
        
    Returns:
        TrainingConfig: 설정 객체
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Config file not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    return TrainingConfig(**config_dict)


def print_config(config: TrainingConfig):
    """
    설정 정보 출력
    
    Args:
        config (TrainingConfig): 설정 객체
    """
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    
    print(f"📋 Experiment: {config.experiment_name}")
    print(f"🏗️  Model: {config.model_name} (x{config.scale})")
    print(f"📊 Dataset: {config.data_path}")
    print(f"🎯 Patch Size: {config.patch_size}")
    print(f"📦 Batch Size: {config.batch_size}")
    print(f"🔄 Epochs: {config.epochs}")
    print(f"📈 Learning Rate: {config.learning_rate}")
    print(f"🔧 Optimizer: {config.optimizer}")
    print(f"📉 Scheduler: {config.scheduler}")
    print(f"💾 Loss Function: {config.loss_type}")
    print(f"📏 Metrics: PSNR={config.use_psnr}, SSIM={config.use_ssim}")
    print(f"⚙️  Device: {config.device}")
    
    if config.model_params:
        print(f"\n🏗️  Model Parameters:")
        for key, value in config.model_params.items():
            print(f"   {key}: {value}")
    
    print("="*60)


def validate_config(config: TrainingConfig) -> List[str]:
    """
    설정 검증
    
    Args:
        config (TrainingConfig): 설정 객체
        
    Returns:
        List[str]: 에러 메시지 리스트 (비어있으면 정상)
    """
    errors = []
    
    # 필수 경로 검증
    if not os.path.exists(config.data_path):
        errors.append(f"Data path not found: {config.data_path}")
    
    # 값 범위 검증
    if config.batch_size <= 0:
        errors.append("Batch size must be positive")
    
    if config.epochs <= 0:
        errors.append("Epochs must be positive")
    
    if config.learning_rate <= 0:
        errors.append("Learning rate must be positive")
    
    if config.scale not in [2, 3, 4, 8]:
        errors.append("Scale must be one of [2, 3, 4, 8]")
    
    if config.patch_size <= 0:
        errors.append("Patch size must be positive")
    
    # Resume 파일 검증
    if config.resume_from and not os.path.exists(config.resume_from):
        errors.append(f"Resume checkpoint not found: {config.resume_from}")
    
    return errors


if __name__ == "__main__":
    # 테스트 코드
    print("Configuration Management Test")
    
    # 기본 설정 생성
    config = TrainingConfig()
    print_config(config)
    
    # 설정 검증
    errors = validate_config(config)
    if errors:
        print("\n❌ Configuration errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\n✅ Configuration is valid")
    
    # 명령행 파서 테스트
    parser = create_argument_parser()
    
    # 기본 모델 파라미터 출력
    print("\n📋 Default Model Parameters:")
    for model in ['EDSR', 'RCAN', 'ESRGAN', 'SwinIR']:
        params = get_default_model_params(model, scale=2)
        print(f"  {model}: {params}")
    
    print("\n🚀 Use this module with train.py for full functionality!") 