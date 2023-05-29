"""
    Copyright 2019 Tae Hwan Jung
    ALBERT Implementation with forking
    Clean Pytorch Code from https://github.com/dhlee347/pytorchic-bert
"""

""" FourierFormer Model Classes & Config Class """

import math
import json
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import split_last, merge_last

from fourier_attention import FourierAttention


class Config(NamedTuple):
    "Configuration for BERT model"
    vocab_size: int = None # Size of Vocabulary
    hidden: int = 768 # Dimension of Hidden Layer in FourierFormer Encoder
    hidden_ff: int = 768*4 # Dimension of Intermediate Layers in Positionwise Feedforward Net
    embedding: int = 128 # Factorized embedding parameterization
    p_drop_hidden: float = 0.1 # finetune dropout
    n_layers: int = 12 # Numher of Hidden Layers
    n_heads: int = 768//64 # Numher of Heads in Multi-Headed Attention Layers
    #activ_fn: str = "gelu" # Non-linear Activation Function Type in Hidden Layers
    max_len: int = 512 # Maximum Length for Positional Embeddings
    n_segments: int = 2 # Number of Sentence Segments
    r: float = 1.125

    @classmethod
    def from_json(cls, file):
        return cls(**json.load(open(file, "r")))


def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg.hidden))
        self.beta  = nn.Parameter(torch.zeros(cfg.hidden))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    "The embedding module from word, position and token_type embeddings."
    def __init__(self, cfg):
        super().__init__()
        # Original BERT Embedding
        # self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.hidden) # token embedding

        # factorized embedding
        self.tok_embed1 = nn.Embedding(cfg.vocab_size, cfg.embedding)
        self.tok_embed2 = nn.Linear(cfg.embedding, cfg.hidden)

        self.pos_embed = nn.Embedding(cfg.max_len, cfg.hidden) # position embedding
        self.seg_embed = nn.Embedding(cfg.n_segments, cfg.hidden) # segment(token type) embedding

        self.norm = LayerNorm(cfg)
        # self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.int64, device=x.device)
        pos = pos.unsqueeze(0).expand_as(x) # (S,) -> (B, S)

        # factorized embedding
        e = self.tok_embed1(x)
        e = self.tok_embed2(e)
        e = e + self.pos_embed(pos) + self.seg_embed(seg)
        #return self.drop(self.norm(e))
        return self.norm(e)


class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden, cfg.hidden_ff)
        self.fc2 = nn.Linear(cfg.hidden_ff, cfg.hidden)
        #self.activ = lambda x: activ_fn(cfg.activ_fn, x)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(gelu(self.fc1(x)))


class FourierBlock(nn.Module):
    """ FourierFormer Block """
    def __init__(self, cfg):
        super().__init__()
        self.attn = FourierAttention(cfg)
        self.proj = nn.Linear(cfg.hidden, cfg.hidden)
        self.norm1 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x, mask):
        h = self.attn(x, mask)
        h = self.norm1(x + self.drop(self.proj(h)))
        h = self.norm2(h + self.drop(self.pwff(h)))
        return h


class FourierFormer(nn.Module):
    """ FourierFormer with Self-Attentive Blocks"""
    def __init__(self, cfg):
        super().__init__()
        self.embed = Embeddings(cfg)
        # Original BERT not used parameter-sharing strategies
        # self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])

        # To used parameter-sharing strategies
        self.n_layers = cfg.n_layers
        self.block = FourierBlock(cfg)

    def forward(self, x, seg, mask):
        h = self.embed(x, seg)
        with torch.no_grad():
            for _ in range(self.n_layers):
                h = self.block(h, mask)

        return h


if __name__ == "__main__":
    model_cfg = Config.from_json('./config/albert_base.json')
    model = FourierFormer(model_cfg)
    pytorch_total_params = sum(p.numel() for p in model.parameters())/1000000
    print("Discriminator    {:2.3f}M".format(pytorch_total_params))
    model_cfg = Config.from_json('./config/generator_base.json')
    model = FourierFormer(model_cfg)
    pytorch_total_params = sum(p.numel() for p in model.parameters())/1000000
    print("Generator        {:2.3f}M".format(pytorch_total_params))
