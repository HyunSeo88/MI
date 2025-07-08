import sys
import os

# train.py를 사용하기 위해 상위 디렉토리를 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

if __name__ == "__main__":
    # train.py를 직접 실행하는 것과 동일한 효과
    sys.argv = [
        'train.py',
        '--domain', 'SR', 
        '--model', 'edsr',
        '--config', 'configs/SR.yaml'
    ]
    
    # train.py의 main 함수 실행
    import train
    train.main() 