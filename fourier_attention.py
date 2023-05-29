from typing import Any
import numpy as np
import torch
import torch.nn as nn
import math

import torch.nn.functional as F

from utils import split_last, merge_last

class FourierAttentionKernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, r):
        ctx.save_for_backward(q, k, r)
        tmp = torch.sinc(r * (q - k))
        y = torch.pow(tmp, 4)
        return torch.prod(y, dim=-1)
    
    @staticmethod
    def backward(ctx: Any, grad_output):
        q, k, r = ctx.saved_tensors
        tmp = r * (q - k)
        y1 = torch.sinc(tmp)
        y1 = 4 * torch.pow(y1, 3)
        y2 = torch.div(torch.cos(tmp) - torch.sinc(tmp),tmp) 
        return grad_output * torch.prod(y1 * y2, dim=-1)
    



class FourierAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_k = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_v = nn.Linear(cfg.hidden, cfg.hidden)
        # self.drop = nn.Dropout(cfg.p_drop_attn)
        self.scores = None # for visualization
        self.n_heads = cfg.n_heads
        self.r = cfg.r
    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
       # (B, H, S, W) , (B, H, W, S) |-> (B, H, S, S) -softmax-> (B, H, S, S)
        kernel = FourierAttentionKernel.apply
        B, H, S, W = q.shape
        weights = torch.zeros((B, H, S, S), device=x.device)
        r = torch.tensor(self.r, device=x.device)
        for l in range(S):
            for i in range(S):
                weights[:,:,l,i] = kernel(q[:,:,l], k[:,:,i], r)
        if mask is not None:
            mask = mask[:, None, None, :].float()
            weights -= 10000.0 * (1.0 - mask)
        #scores = self.drop(F.softmax(scores, dim=-1))
        weights = F.softmax(weights, dim=-1)
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        y = (weights @ v)
        weight_sum = weights.sum(dim=-1)
        for w in range(W):
            y[:,:,:,w] = torch.div(y[:,:,:,w],weight_sum)        
        y = y.transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(y, 2)
        self.scores = weights
        return h
