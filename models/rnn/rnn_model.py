from typing import Optional
import torch.nn as nn
import torch


class RNN_Model(nn.Module):

    def __init__(self, config):
        super(RNN_Model, self).__init__()
        self.num_classes = 2 if config.task == "classification" else 1
        self.pulse_len = config.pulse_len
        self.pulse_num = config.pulse_num

        self.mlp = nn.Sequential(
            nn.Linear(config.pulse_len, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
        )

        self.rnn = nn.LSTM(
            input_size=config.d_model,
            hidden_size=config.d_model * 2,
            num_layers=1,
            batch_first=True,
            dropout=config.dropout,
            bidirectional=True,
        )

        self.head = nn.Linear(config.d_model * 2, self.num_classes)

    def forward(self, x_enc):
        B, L, M = x_enc.shape
        x_enc = self.mlp(x_enc)

        outputs, (h_n, c_n) = self.rnn(x_enc)

        h_last = h_n[-1]  # [B, hid_dim]

        outputs = self.head(h_last)

        return outputs.squeeze(-1)
