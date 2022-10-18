from typing import List, Tuple
from pathlib import Path
import torchaudio
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from src.utils import project_root

class Golos(Dataset):
    _sample_rate = 8000
    def __init__(self, path: Path):
        pass

    def __getitem__(self, index: int):
        pass

    def __len__(self):
        pass