import torch.nn as nn
import torch


class PulseConvModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.pulse_len, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
        )

        self.conv = nn.Sequential(
            nn.Conv1d(config.d_model, config.d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(config.d_model, config.d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.pool = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Linear(config.d_model, 1)

    def forward(self, x):
        """
        x: [B, 160, 30]
        """
        tokens = self.mlp(x)  # [B, 160, d_model]
        tokens = tokens.transpose(1, 2)  # [B, d_model, 160]

        feat = self.conv(tokens)  # [B, d_model, 160]
        feat = self.pool(feat).squeeze(-1)  # [B, d_model]

        out = self.head(feat)  # [B, 1]
        return out.squeeze(-1)
