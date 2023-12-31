import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from einops import rearrange, repeat, einsum
from tqdm import tqdm
import math
import os
from dataclasses import dataclass
import json
from typing import Optional, Union

from transformers import AutoTokenizer

@dataclass
class ModelArgs:
    d_model: int,
    n_layers: int,
    vocab_size: int,
    d_state: int = 16,
    expand: int = 4,
    dt_rank: Union[int, str] = "auto",
    d_conv:int = 4,
    pad_vocab_size_multiple: int = 8,
    conv_bias: bool = True,
    bias: bool = False
    def __post_init__(self):
        self.d_inner = int(self.d_model * self.expand)
        if self.dt_rank == "auto":
            self.dt_rank = self.d_model // 16
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += self.pad_vocab_size_multiple - self.vocab_size % self.pad_vocab_size_multiple





device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print('Using device:', device)

