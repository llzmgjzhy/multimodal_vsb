import torch.nn as nn
import torch


class PulseGRUModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = 2 if config.task == "classification" else 1
        self.pulse_len = config.pulse_len
        self.pulse_num = config.pulse_num

        self.mlp = nn.Sequential(
            nn.Linear(config.pulse_len, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
        )

        self.gru = nn.GRU(
            input_size=config.d_model,
            hidden_size=config.d_model * 2,
            batch_first=True,
            bidirectional=False,
        )

        self.head = nn.Linear(config.d_model * 2, 1)

    def forward(self, x):
        """
        x: [B, 160, 30]
        """
        tokens = self.mlp(x)  # [B, 160, d_model]
        _, h_n = self.gru(tokens)  # h_n: [1, B, hidden_dim]
        h_n = h_n.squeeze(0)  # [B, hidden_dim]
        out = self.head(h_n)  # [B, 1]
        return out.squeeze(-1)
