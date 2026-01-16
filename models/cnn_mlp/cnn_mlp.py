import torch.nn as nn
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F


class PulseEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pulse_len = config.pulse_len
        self.d_model = config.d_model
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * config.pulse_len, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
        )
        self.head = nn.Linear(config.d_model, 1)

    def forward(self, x):
        """
        x: [B, N, 30]
        """
        B, N, L = x.shape
        x = x.view(B * N, 1, L)  # [BN, 1, 30]
        x = self.conv(x)  # [BN, 32, 30]
        x = x.flatten(1)  # [BN, 32*30]
        z = self.fc(x)  # [BN, d_model]
        z = z.view(B, N, -1)  # [B, N, d_model]
        z = z.mean(dim=1)  # [B, d_model]
        z = self.head(z)  # [B,  1]
        return z.squeeze(-1)  # [B]
