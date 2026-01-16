import torch
import torch.nn as nn
import torch.nn.functional as F


class PulseEncoder(nn.Module):
    def __init__(self, pulse_len=30, d_model=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pulse_len, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x):
        # x: [B, N, 30]
        return self.net(x)  # [B, N, d_model]


class DeepClustering(nn.Module):
    def __init__(self, d_model=64, K=6, temp=0.1):
        super().__init__()
        self.K = K
        self.temp = temp

        self.centroids = nn.Parameter(torch.randn(K, d_model))
        nn.init.xavier_uniform_(self.centroids)

    def forward(self, z):
        """
        z: [B, N, d_model]
        """
        z = F.normalize(z, dim=-1)
        c = F.normalize(self.centroids, dim=-1)

        # similarity
        sim = torch.einsum("bnd,kd->bnk", z, c)

        # soft assignment
        assign = F.softmax(sim / self.temp, dim=-1)

        return assign, sim


class SimpleDeepClusteringModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pulse_len = config.pulse_len
        self.d_model = config.d_model
        self.K = 6
        self.encoder = PulseEncoder(config.pulse_len, config.d_model)
        self.cluster = DeepClustering(config.d_model, self.K)

    def forward(self, x):
        """
        x: [B, 160, 30]
        """
        z = self.encoder(x)
        assign, sim = self.cluster(z)
        return assign
