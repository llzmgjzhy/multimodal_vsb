from typing import Optional
import torch.nn as nn


class Linear_Model(nn.Module):

    def __init__(self, config):
        super(Linear_Model, self).__init__()
        self.num_classes = 1
        self.pulse_len = config.pulse_len
        self.pulse_num = config.pulse_num

        self.ln_proj = nn.LayerNorm(self.pulse_len * self.pulse_num)
        self.out_layer = nn.Linear(self.pulse_len * self.pulse_num)

    def forward(self, x_enc):
        B, L, M = x_enc.shape

        outputs = x_enc.reshape(B, -1)
        outputs = self.ln_proj(outputs)
        outputs = self.out_layer(outputs)

        return outputs.squeeze(-1)
