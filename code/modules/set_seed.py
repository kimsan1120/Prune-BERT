import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """
    모든 라이브러리의 랜덤 시드를 고정하여 재현성을 보장하는 함수.

    Args:
        seed (int, optional): 고정할 랜덤 시드 값. 기본값은 42.
    """
    random.seed(seed)  # Python 기본 랜덤 시드 설정
    np.random.seed(seed)  # NumPy 랜덤 시드 설정
    torch.manual_seed(seed)  # PyTorch 랜덤 시드 설정
    torch.cuda.manual_seed_all(seed)  # 모든 GPU에서 동일한 시드 설정
    torch.backends.cudnn.deterministic = True  # CuDNN에서 결정적 연산 보장
    torch.backends.cudnn.benchmark = False  # 성능 최적화를 비활성화하여 재현성 유지

    print(f"[INFO] Random seed set to {seed}")
