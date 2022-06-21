import random
import yaml

import numpy as np
import torch


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def load_config(config_file_path: str):
    with open(config_file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
