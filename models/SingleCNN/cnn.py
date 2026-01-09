import torch
import torch.nn as nn
import torch.nn.functional as F


class SinglePulseCNN(nn.Module):
    def __init__(self, config):
        super(SinglePulseCNN, self).__init__()
        self.num_classes = 1
        self.pulse_len = config.pulse_len
        self.pulse_num = config.pulse_num

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        # x.shape [b, 160, 30]
        B, L, M = x.shape
        x = x.reshape(-1, 1, M)  # [b*160, 1, 30]
        x = self.conv(x)  # [b*160, 64, 30]
        x = x.mean(dim=-1)  # [b*160, 64]
        x = x.reshape(B, L, 64)  # [b, 160, 64]
        x = self.head(x)  # [b, 160, 1]
        x = x.mean(dim=1, keepdim=False)  # [b, 1]

        return x.squeeze(-1)
