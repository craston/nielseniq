import torch
import random
import numpy as np


def set_global_seed(seed: int):
    """
    Set the global seed for reproducibility in PyTorch, NumPy, and Python's built-in random.

    Args:
        seed (int): The seed value.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Ensure all CUDA operations are deterministic
    np.random.seed(seed)
    random.seed(seed)

    # Ensure deterministic behavior in cuDNN-based models
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
