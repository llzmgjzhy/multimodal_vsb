from typing import Optional
import torch.nn as nn
import torch


class RNN_Model(nn.Module):

    def __init__(self, config):
        super(RNN_Model, self).__init__()
        self.num_classes = 1
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
            hidden_size=config.d_model,
            num_layers=1,
            batch_first=True,
            dropout=config.dropout,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(2 * config.d_model, 2 * config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(2 * config.d_model, self.num_classes),
        )

    def forward(self, x_enc):
        B, L, M = x_enc.shape
        x_enc = self.mlp(x_enc)

        outputs, (h_n, c_n) = self.rnn(x_enc)

        h_forward = h_n[-2]  # [B, d_model]
        h_backward = h_n[-1]  # [B, d_model]

        outputs = torch.cat((h_forward, h_backward), dim=1)  # [B, 2*d_model]

        outputs = self.classifier(outputs)

        return outputs.squeeze(-1)
