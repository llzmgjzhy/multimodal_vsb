import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unsupervised_learning.prototype.prototype_stats import DL_SOTA_PrototypeNet


class PulseEncoder(nn.Module):
    def __init__(self, in_dim=30, hidden_dim=64, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        # x: [B, N, 30]
        return self.net(x)  # [B, N, d]


class ProtoCrossAttention(nn.Module):
    def __init__(self, dim=128, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )

    def forward(self, z, prototypes):
        # z: [B, N, d]
        # prototypes: [K, d] → expand to batch

        B = z.size(0)
        proto = prototypes.unsqueeze(0).expand(B, -1, -1)

        attn_out, _ = self.attn(query=z, key=proto, value=proto)

        return z + attn_out


class PrototypeLayer(nn.Module):
    def __init__(self, num_proto=8, dim=128, temperature=0.1):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_proto, dim))
        self.temperature = temperature

    def forward(self, z):
        z_norm = F.normalize(z, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)

        logits = torch.einsum("bnd,kd->bnk", z_norm, p_norm)
        assign = F.softmax(logits / self.temperature, dim=-1)

        return assign, self.prototypes


class ProtoEnrichMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_dim = config.pulse_len
        d = config.d_model
        num_proto = 6
        self.stage1_model = DL_SOTA_PrototypeNet(config)
        # self.stage1_model.eval()
        # for param in self.stage1_model.parameters():
        #     param.requires_grad = False

        self.fuse = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.LayerNorm(d),
            nn.GELU(),
        )
        self.head = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        z = self.stage1_model.encoder(x)  # [B,N,d]
        stats = self.stage1_model.proto(z)
        assign = stats["assign"]  # [B,N,K]
        protos = stats["prototypes"]  # [K,D]

        proto_ctx = torch.einsum("bnk,kd->bnd", assign, protos)

        z = self.fuse(torch.cat([z, proto_ctx], dim=-1))
        z = z.mean(dim=1)

        out = self.head(z)
        return out.squeeze(-1)


class ProtoModelB3(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_dim = config.pulse_len
        d_model = config.d_model
        num_proto = 6
        self.use_cls_token = True
        self.encoder = PulseEncoder(in_dim, 64, d_model)
        self.prototypes = nn.Parameter(torch.randn(num_proto, d_model))

        self.proto_attn = ProtoCrossAttention(d_model)

        self.encoder_blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True),
            num_layers=2,
        )
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model))

        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        B, N, _ = x.shape
        z = self.encoder(x)  # [B,N,d]
        z = self.proto_attn(z, self.prototypes)
        if self.use_cls_token:
            cls = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
            z = torch.cat([cls, z], dim=1)  # [B, 161, d_model]
        z = self.encoder_blocks(z)

        if self.use_cls_token:
            z = z[:, 0]  # [B, d_model]
        else:
            z = z.mean(dim=1)  # mean pooling
        out = self.head(z)

        return out.squeeze(-1)
