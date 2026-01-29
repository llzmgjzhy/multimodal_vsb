import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import os


class SetEncoderTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = config.pulse_len
        self.d_model = config.d_model
        self.nhead = config.n_heads
        self.num_layers = config.n_layers
        self.dropout = config.dropout

        self.token_proj = nn.Linear(self.input_dim, self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        # 无位置编码，无 patch embedding
        # 输出集合级表征
        self.pool = nn.AdaptiveAvgPool1d(1)  # [B, N, D] → [B, D]

        self.seed_vectors = nn.Parameter(torch.randn(1, 6, self.d_model))
        self.mha = nn.MultiheadAttention(self.d_model, self.nhead, batch_first=True)

        self.proj_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.GELU(),
            nn.BatchNorm1d(self.d_model * 2),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.BatchNorm1d(self.d_model),
        )

    def forward(self, x, pretrain=True):  # [B, N, 30]
        x = self.token_proj(x)  # → [B, N, d_model]
        z = self.encoder(x)  # → [B, N, d_model]
        seed = self.seed_vectors.expand(z.size(0), -1, -1)  # → [B, k, d_model]
        z, _ = self.mha(seed, z, z)  # → [B, k, d_model]
        z = self.pool(z.transpose(1, 2)).squeeze(-1)  # → [B, d_model]
        if pretrain:
            z = self.proj_head(z)  # → [B, d_model]
        return z


class DownStreamClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = SetEncoderTransformer(config)
        # frozen encoder
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        # self.encoder.eval()

        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1)
        )

    def forward(self, x):  # [B, N, 30]
        z = self.encoder(x, pretrain=False)  # → [B, d_model]
        logits = self.classifier(z)  # → [B, num_classes]
        return logits.squeeze(-1)  # [B]
