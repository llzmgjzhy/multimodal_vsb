import torch.nn as nn
from torch.nn import functional as F
from torchvision.ops.focal_loss import sigmoid_focal_loss
import torch


def get_loss_module(config):

    loss_type = config.loss

    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(reduction="none")

    if loss_type == "mae":
        return nn.L1Loss(reduction="none")

    if loss_type == "mse":
        return nn.MSELoss(reduction="none")

    if loss_type == "bce":
        return nn.BCEWithLogitsLoss(reduction="none")

    if loss_type == "focal":
        return lambda inp, target: sigmoid_focal_loss(inp, target, reduction="none")

    if loss_type == "cluster":
        return lambda stats, traget: (
            stats["var"].mean()
            + 0.2
            * F.kl_div(
                (stats["assign"].mean(dim=(0, 1)) + 1e-8).log(),
                torch.full_like(
                    stats["assign"].mean(dim=(0, 1)), 1.0 / stats["assign"].size(-1)
                ),
                reduction="batchmean",
            )
        )

    else:
        raise ValueError(f"Loss module for '{loss_type}' does not exist")
