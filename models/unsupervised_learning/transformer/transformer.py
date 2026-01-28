import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F


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

    def forward(self, x):  # [B, N, 30]
        x = self.token_proj(x)  # → [B, N, d_model]
        z = self.encoder(x)  # → [B, N, d_model]
        z = z.mean(dim=1)  # 或 self.pool(z.transpose(1,2)).squeeze(-1)
        return z
