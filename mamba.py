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
    d_model: int
    n_layers: int
    vocab_size: int
    d_state: int = 16
    expand: int = 4
    dt_rank: Union[int, str] = "auto"
    d_conv:int = 4
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    def __post_init__(self):
        self.d_inner = int(self.d_model * self.expand)
        if self.dt_rank == "auto":
            self.dt_rank = self.d_model // 16
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += self.pad_vocab_size_multiple - self.vocab_size % self.pad_vocab_size_multiple

class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.layers = nn.ModuleList([MambaBlock(args) for _ in range(args.n_layers)])  
        self.norm = RMSNorm(args.d_model)
        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)
    
    @staticmethod
    def from_pretrained(model_name: str):
        from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
        from transformers.utils import cached_file
        def load_config_hf(path_to_config_file):
            resolved_archive_file = cached_file(model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
            return json.load(open(resolved_archive_file))
        
        def load_state_dict_hf(model_name, device=None, dtypes=None):
            resolved_archive_file = cached_file(model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)
            state_dict = torch.load(resolved_archive_file, weights_only=True, map_location=device, mmap=True)
            return state_dict

        config_file = load_config_hf(model_name)
        args = ModelArgs(
            d_model=config_file["d_model"],
            n_layers=config_file["n_layer"],
            vocab_size=config_file["vocab_size"],
        )
        model= Mamba(args)
        state_dict = load_state_dict_hf(model_name)
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace("backbone.", "")
            new_state_dict[new_key] = state_dict[key]

        model.load_state_dict(new_state_dict)
        return model


class MambaBlock(nn.Module):
    pass

class RMSNorm(nn.Module):
    pass



device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print('Using device:', device)

