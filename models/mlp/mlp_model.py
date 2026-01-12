from typing import Optional
import torch.nn as nn
from models.one_fits_all.embed import DataEmbedding


class MLP_Model(nn.Module):

    def __init__(self, config):
        super(MLP_Model, self).__init__()
        self.num_classes = 1
        self.pulse_len = config.pulse_len
        self.pulse_num = config.pulse_num

        self.net = nn.Sequential(
            nn.LayerNorm(config.pulse_len),
            nn.Linear(config.pulse_len, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, self.num_classes),
        )

        self.out_layer = nn.Linear(config.pulse_num * config.d_model, self.num_classes)

    def forward(self, x_enc):
        B, L, M = x_enc.shape  # [b, 160, 30]
        outputs = self.net(x_enc)  # [b, 160, 1]
        # mean pooling over the pulse_num dimension
        outputs = outputs.reshape(B, self.pulse_num).mean(
            dim=1, keepdim=False
        )  # [b, 1]

        return outputs.squeeze(-1)
