import torch.nn as nn
import torch


class PulseMLP(nn.Module):
    def __init__(self, pulse_len=30, d_model=64, hidden=128):
        super().__init__()

        self.in_proj = nn.Sequential(
            nn.LayerNorm(pulse_len),
            nn.Linear(pulse_len, hidden),
            nn.GELU(),
        )

        self.block = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x):
        # x: [B,160,30]
        h = self.in_proj(x)  # [B,160,hidden]
        h = self.block(h)  # residual
        h = self.out_proj(h)  # [B,160,d_model]
        return h


class PulseTransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = 2 if config.task == "classification" else 1
        self.pulse_len = config.pulse_len
        self.pulse_num = config.pulse_num
        self.use_cls_token = True

        # pulse → token
        self.mlp = PulseMLP(
            pulse_len=config.pulse_len,
            d_model=config.d_model,
            hidden=config.d_model * 2,
        )

        # positional embedding (pulse index)
        self.pos_embed = nn.Parameter(
            torch.randn(
                1,
                self.pulse_num + (1 if self.use_cls_token else 0),
                config.d_model,
            )
        )

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=4,
            dim_feedforward=256,
            dropout=config.dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1,
        )

        self.head = nn.Linear(config.d_model, 1)

    def forward(self, x):
        """
        x: [B, 160, 30]
        """
        B, N, _ = x.shape

        tokens = self.mlp(x)  # [B, 160, d_model]

        if self.use_cls_token:
            cls = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
            tokens = torch.cat([cls, tokens], dim=1)  # [B, 161, d_model]

        tokens = tokens + self.pos_embed[:, : tokens.size(1)]

        tokens = self.transformer(tokens)  # [B, N(+1), d_model]

        if self.use_cls_token:
            feat = tokens[:, 0]  # [B, d_model]
        else:
            feat = tokens.mean(dim=1)  # mean pooling

        out = self.head(feat)  # [B, 1]
        return out.squeeze(-1)
