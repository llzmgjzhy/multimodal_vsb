import torch
import torch.nn as nn
import torch.nn.functional as F


class PulseEncoder(nn.Module):
    def __init__(self, in_dim=30, hidden=64, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        # x: [B, N, L]
        return self.net(x)  # [B, N, D]


class PrototypeStats(nn.Module):
    def __init__(self, dim, num_proto=6, temp=0.1):
        super().__init__()
        self.K = num_proto
        self.temp = temp
        self.prototypes = nn.Parameter(torch.randn(num_proto, dim))

    def forward(self, z):
        """
        z: [B, N, D]
        """
        B, N, D = z.shape

        # [B, N, K]
        logits = torch.einsum("bnd,kd->bnk", z, self.prototypes)
        assign = torch.softmax(logits / self.temp, dim=-1)

        # soft count
        count = assign.sum(dim=1) + 1e-6  # [B, K]

        # mean embedding
        mean = torch.einsum("bnk,bnd->bkd", assign, z) / count.unsqueeze(-1)

        # dispersion
        diff = z.unsqueeze(2) - self.prototypes.unsqueeze(0).unsqueeze(0)
        dist2 = diff.pow(2).sum(-1)    # [B, N, K]
        var = torch.einsum("bnk,bnk->bk", assign, dist2)
        var = var / count

        return {
            "assign": assign,  # [B, N, K]
            "count": count,  # [B, K]
            "mean": mean,  # [B, K, D]
            "var": var,  # [B, K]
        }


class PrototypeAggregator(nn.Module):
    def __init__(self, dim, num_proto):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(num_proto * (dim + 2), 128), nn.GELU())

    def forward(self, stats):
        B, K, D = stats["mean"].shape

        feat = torch.cat(
            [stats["count"], stats["mean"].reshape(B, K * D), stats["var"]], dim=1
        )

        return self.fc(feat)


class DL_SOTA_PrototypeNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        pulse_len = config.pulse_len
        dim = config.d_model
        K = 6
        self.encoder = PulseEncoder(pulse_len, 64, dim)
        self.proto = PrototypeStats(dim, K)
        self.agg = PrototypeAggregator(dim, K)
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        z = self.encoder(x)  # [B, N, D]
        stats = self.proto(z)
        feat = self.agg(stats)
        out = self.head(feat)
        return stats['var']
