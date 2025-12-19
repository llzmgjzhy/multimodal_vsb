from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from einops import rearrange
from models.one_fits_all.embed import DataEmbedding, DataEmbedding_wo_time


class gpt4ts(nn.Module):

    def __init__(self, config):
        super(gpt4ts, self).__init__()
        self.gpt_layers = 6
        self.num_classes = 1
        self.d_model = config.d_model

        self.patch_num = config.pulse_num

        self.enc_embedding = DataEmbedding(
            config.pulse_len,
            config.d_model,
            dropout=config.dropout,
        )

        self.gpt2 = GPT2Model.from_pretrained(
            "gpt2", output_attentions=True, output_hidden_states=True
        )
        self.gpt2.h = self.gpt2.h[: self.gpt_layers]

        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if "ln" in name or "wpe" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        device = torch.device("cuda:{}".format(0))
        self.gpt2.to(device=device)

        self.act = F.gelu

        self.ln_proj = nn.LayerNorm(config.d_model * self.patch_num)

        self.out_layer = nn.Linear(config.d_model * self.patch_num, self.num_classes)

    def forward(self, x_enc):
        B, L, M = x_enc.shape

        outputs = self.enc_embedding(x_enc, None)
        # outputs shape: (B, L, d_model)

        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state

        outputs = self.act(outputs).reshape(B, -1)
        outputs = self.ln_proj(outputs)
        outputs = self.out_layer(outputs)

        return outputs.squeeze(-1)
