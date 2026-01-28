__all__ = ["PatchTST"]

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from .layers.PatchTST_backbone import PatchTST_backbone


class Model(nn.Module):
    def __init__(self, configs):

        super().__init__()
        max_seq_len: Optional[int] = 1024
        d_k: Optional[int] = None
        d_v: Optional[int] = None
        norm: str = "BatchNorm"
        attn_dropout: float = 0.0
        act: str = "gelu"
        key_padding_mask: bool = "auto"
        padding_var: Optional[int] = None
        attn_mask: Optional[Tensor] = None
        res_attention: bool = True
        pre_norm: bool = False
        store_attn: bool = False
        pe: str = None
        learn_pe: bool = True
        pretrain_head: bool = False
        head_type = "flatten"
        verbose: bool = False

        # load parameters
        c_in = 1
        context_window = 5000
        target_window = 1

        n_layers = configs.n_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout

        patch_len = configs.pulse_len
        stride = configs.pulse_len

        revin = False

        self.model = PatchTST_backbone(
            c_in=c_in,
            context_window=context_window,
            target_window=target_window,
            patch_len=patch_len,
            stride=stride,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            act=act,
            key_padding_mask=key_padding_mask,
            padding_var=padding_var,
            attn_mask=attn_mask,
            res_attention=res_attention,
            pre_norm=pre_norm,
            store_attn=store_attn,
            pe=pe,
            learn_pe=learn_pe,
            pretrain_head=pretrain_head,
            head_type=head_type,
            verbose=verbose,
        )

    def forward(self, x):  # x: [Batch, Pulse num, Pulse length]
        x = x.unsqueeze(1)  # x: [Batch, Channel=1, Pulse num, Pulse length]
        x = self.model(x)  # x: [Batch, Channel=1, Output length]
        x = x.squeeze()  # x: [Batch,]
        return x
