"""Transformer-encoder variant of HitSpotter -- same per-frame input/output
contract (feature vector in, hit-logit + shot-logit per frame out) as the
TCN-based hit_model.py, but with global self-attention instead of local
dilated convolution. Genuinely different inductive bias from the TCN family
(v3/v4/seeds): attention can directly relate a frame to any other frame in
the sequence (e.g. a swing's windup starting well before contact), whereas
the TCN only sees a fixed local receptive field. Real architectural
diversity, useful for ensembling regardless of whether it wins alone.
"""
import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.shape[1]]


class HitSpotterTransformer(nn.Module):
    def __init__(self, in_dim=360, d_model=192, n_heads=6, n_layers=4, n_shot_types=12, dim_ff=384):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            batch_first=True, dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.hit_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.shot_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, n_shot_types))

    def forward(self, x, pad_mask=None):
        """x: (B, T, in_dim). pad_mask: (B, T) bool, True = PADDED (ignore).
        Returns hit_logits (B, T), shot_logits (B, T, n_shot_types)."""
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        hit_logits = self.hit_head(h).squeeze(-1)
        shot_logits = self.shot_head(h)
        return hit_logits, shot_logits
