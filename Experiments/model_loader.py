"""
Model Loader Module
Super Resolution 모델들을 로드하는 기능
"""

import os
import sys
import torch
import torch.nn as nn
import nbformat
from nbconvert import PythonExporter
from pathlib import Path


def load_notebook_as_module(notebook_path):
    """
    주피터 노트북을 Python 모듈로 로드
    
    Args:
        notebook_path (str): 노트북 파일 경로
        
    Returns:
        module: 로드된 모듈
    """
    if not os.path.exists(notebook_path):
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")
    
    # 노트북 읽기
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Python 코드로 변환
    exporter = PythonExporter()
    source, _ = exporter.from_notebook_node(nb)
    
    # 모듈 생성 및 실행
    module = type(sys)('module')
    exec(source, module.__dict__)
    
    return module


class ModelLoader:
    """SR 모델 로더"""
    
    def __init__(self, model_dir="../SR"):
        """
        Args:
            model_dir (str): 모델 디렉토리 경로
        """
        self.model_dir = Path(model_dir)
        self.available_models = self._scan_models()
        
    def _scan_models(self):
        """사용 가능한 모델들을 스캔"""
        models = {}
        
        # CNN-based models
        cnn_dir = self.model_dir / "CNN-based"
        if cnn_dir.exists():
            for model_file in cnn_dir.glob("*.ipynb"):
                model_name = model_file.stem
                models[model_name] = {
                    'path': str(model_file),
                    'type': 'CNN-based',
                    'class_name': model_name  # 보통 파일명과 클래스명이 같음
                }
        
        # GAN-based models
        gan_dir = self.model_dir / "GAN-based"
        if gan_dir.exists():
            for model_file in gan_dir.glob("*.ipynb"):
                model_name = model_file.stem
                models[model_name] = {
                    'path': str(model_file),
                    'type': 'GAN-based',
                    'class_name': model_name
                }
        
        # Transformer-based models
        transformer_dir = self.model_dir / "Transformer-based"
        if transformer_dir.exists():
            for model_file in transformer_dir.glob("*.ipynb"):
                model_name = model_file.stem
                models[model_name] = {
                    'path': str(model_file),
                    'type': 'Transformer-based',
                    'class_name': model_name
                }
        
        return models
    
    def list_models(self):
        """사용 가능한 모델 목록 출력"""
        print("Available Super Resolution Models:")
        print("=" * 50)
        
        for model_name, info in self.available_models.items():
            print(f"• {model_name}")
            print(f"  Type: {info['type']}")
            print(f"  Path: {info['path']}")
            print(f"  Class: {info['class_name']}")
            print()
    
    def load_model(self, model_name, **kwargs):
        """
        모델 로드
        
        Args:
            model_name (str): 모델 이름
            **kwargs: 모델 생성 파라미터
            
        Returns:
            model: 로드된 모델
        """
        if model_name not in self.available_models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.available_models.keys())}")
        
        model_info = self.available_models[model_name]
        model_path = model_info['path']
        class_name = model_info['class_name']
        
        print(f"Loading model: {model_name}")
        print(f"From: {model_path}")
        
        try:
            # 노트북 모듈로 로드
            module = load_notebook_as_module(model_path)
            
            # 모델 클래스 가져오기
            if hasattr(module, class_name):
                model_class = getattr(module, class_name)
            else:
                raise AttributeError(f"Class '{class_name}' not found in {model_path}")
            
            # 모델 인스턴스 생성
            model = model_class(**kwargs)
            
            print(f"✓ Model {model_name} loaded successfully")
            return model
            
        except Exception as e:
            print(f"✗ Failed to load model {model_name}: {e}")
            raise
    
    def get_model_info(self, model_name):
        """모델 정보 가져오기"""
        if model_name not in self.available_models:
            return None
        return self.available_models[model_name]


def get_model_configs():
    """사전 정의된 모델 설정들"""
    configs = {
        'EDSR': {
            'n_colors': 3,
            'n_feats': 64,
            'kernel_size': 3,
            'n_resblocks': 16,
            'scale': 2,
            'res_scale': 1.0
        },
        'RCAN': {
            'in_channels': 3,
            'n_feats': 64,
            'kernel_size': 3,
            'num_rg': 10,
            'num_rcab': 20,
            'scale': 2
        },
        'ESRGAN': {
            'n_colors': 3,
            'n_feats': 64,
            'kernel_size': 3,
            'n_basic_blocks': 23,
            'scale': 2
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
            'upscale': 2
        }
    }
    
    return configs


def create_model(model_name, **kwargs):
    """
    모델 생성 헬퍼 함수
    
    Args:
        model_name (str): 모델 이름
        **kwargs: 모델 파라미터 (scale 포함)
        
    Returns:
        model: 생성된 모델
    """
    loader = ModelLoader()
    configs = get_model_configs()
    
    if model_name in configs:
        config = configs[model_name].copy()
        config.update(kwargs)  # kwargs로 덮어쓰기 (scale 포함)
        
        model = loader.load_model(model_name, **config)
        return model
    else:
        # 기본 파라미터로 모델 생성
        default_config = {
            'n_colors': 3,
            'scale': 2,  # 기본 scale
        }
        default_config.update(kwargs)  # kwargs로 덮어쓰기
        
        model = loader.load_model(model_name, **default_config)
        return model


def count_parameters(model):
    """모델 파라미터 개수 계산"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def print_model_info(model, model_name="Model"):
    """모델 정보 출력"""
    params = count_parameters(model)
    
    print(f"\n{model_name} Information:")
    print("=" * 50)
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Non-trainable parameters: {params['non_trainable']:,}")
    print(f"Model size: {params['total'] * 4 / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    # 테스트 코드
    loader = ModelLoader()
    loader.list_models()
    
    # 모델 설정 출력
    print("\nPredefined Model Configurations:")
    print("=" * 50)
    configs = get_model_configs()
    for name, config in configs.items():
        print(f"{name}: {config}")
    
    # 모델 로드 테스트 (EDSR이 있다면)
    try:
        model = create_model('EDSR', scale=2)
        print_model_info(model, 'EDSR')
    except Exception as e:
        print(f"Model loading test failed: {e}") 