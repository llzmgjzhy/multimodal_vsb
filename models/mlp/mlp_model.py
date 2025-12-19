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
            nn.Linear(config.pulse_len, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
        )

        self.out_layer = nn.Linear(config.pulse_num * config.d_model, self.num_classes)

    def forward(self, x_enc):
        B, L, M = x_enc.shape
        outputs = self.net(x_enc)
        outputs = outputs.reshape(B, -1)

        outputs = self.out_layer(outputs)

        return outputs.squeeze(-1)
