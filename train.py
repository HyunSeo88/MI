import argparse
import yaml
import importlib
from engine import Trainer
from utils import get_dataloaders

def parse_args():
    """
    - 실험 단계에서 어떤 도메인에서, 어떤 모델을, 어떤 설정 파일로 실험을 돌릴지에 대한 최고 레벨의 인자만 처리한다.
    
    """
    parser = argparse.ArgumentParser(description="훈련 스크립트")
    parser.add_argument(
        "--domain", required=True,
        choices=["SR","NLP","SignalProcessing"],
        help="도메인 이름"
    )
    parser.add_argument(
        "--model", required=True,
        help="모델 이름"
    )
    parser.add_argument(
        "--config", required=True,
        help="YAML 설정 파일 경로"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # 1) 설정 불러오기
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    # cfg를 dot notation으로 접근 가능한 객체로 변환
    class Config:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    setattr(self, key, Config(value))
                else:
                    setattr(self, key, value)
    
    # 도메인별 설정 생성        
    domain_cfg = Config(cfg)
        
    # 2) 데이터로더 준비
    dataloaders = get_dataloaders(cfg)
    
    # 3) 모델 동적 로딩
    module_path = f"{args.domain}.models.{args.model}"
    module = importlib.import_module(module_path)
        # 여기서 ModelClass 변수는 SRGAN 클래스 정의 자체를 가리키게 된다.
    ModelClass = getattr(module, args.model.upper())    # args.model 값인 srgan을 대문자로 바꾼다. upper함 + getattr는 첫번째 인자에서로 받은 객체에서 두번째 인자로 받은 문자열과 똑같은 이름의 속성을 찾아서 반환한다.
    
    # 모델별 설정 가져오기
    model_cfg = cfg.get(args.model, {})
    model = ModelClass(model_cfg)
    
    # 4) Trainer 생성 및 학습 실행
    domain_cfg.model_name = args.model  # 로깅을 위한 모델명 추가
    trainer = Trainer(model, dataloaders, domain_cfg)
    trainer.train(cfg.get("epochs", 100))
    
# 이 코드가 없으면, 다른 코드 파일에서 이 코드를 import할 때마다 이 코드가 실행되어 버린다.
if __name__=="__main__":
    main()
        
        
    """
    파이썬은 모든 .py 파일을 실행할 때 내부적으로 __name__ 이라는 변수를 설정한다. 이 변수의 값은 파일이 어떻게 실행되느냐에 따라 달라진다.
        1. 직접 실행될 때
            - 터미널에서 "python my_script.py"와 같이 직접 실행하면, 파이썬은 해당 파일의 __name__ 변수에 "__main__"이라는 특별한 문자열을 할당한다.
        2. 모듈로 임포트될 때
            - 다른 파일에서 import my_script와 같이 임포트하면, my_script.py 파일의 __name__ 변수에는 모듈 이름인 my_script가 할당된다.
    """