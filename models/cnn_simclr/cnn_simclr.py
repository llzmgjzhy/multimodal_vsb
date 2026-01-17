import torch
import torch.nn as nn
import torch.nn.functional as F

class PulseEncoder(nn.Module):
    def __init__(self, input_dim=1, feature_dim=64):
        super(PulseEncoder, self).__init__()
        # 针对长度30的微型脉冲设计
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2), # 长度 30 -> 15
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # 转化为向量
        )
        self.fc = nn.Linear(32, feature_dim)

    def forward(self, x):
        # x shape: [batch, 1, 30]
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        return self.fc(out) # 输出嵌入向量