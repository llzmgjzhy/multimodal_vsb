from typing import Optional
import torch.nn as nn
from models.one_fits_all.embed import DataEmbedding


class Linear_Model(nn.Module):

    def __init__(self, config):
        super(Linear_Model, self).__init__()
        self.num_classes = 1
        self.pulse_len = config.pulse_len
        self.pulse_num = config.pulse_num

        self.enc_embedding = DataEmbedding(
            config.pulse_len,
            config.d_model,
            dropout=config.dropout,
        )
        self.ln_proj = nn.LayerNorm(config.pulse_num * config.d_model)
        self.out_layer = nn.Linear(config.pulse_num * config.d_model, self.num_classes)

    def forward(self, x_enc):
        B, C, L, M = x_enc.shape
        outputs = self.enc_embedding(x_enc, None)
        outputs = outputs.reshape(B, -1)
        outputs = self.ln_proj(outputs)
        outputs = self.out_layer(outputs)

        return outputs.squeeze(-1)
