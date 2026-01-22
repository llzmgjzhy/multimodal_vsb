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
        return lambda assign, traget: assign['var'].mean()

    else:
        raise ValueError(f"Loss module for '{loss_type}' does not exist")


def cluster_balance_loss(assign):
    """
    Encourage all clusters to be used
    """
    p = assign.mean(dim=(0, 1))  # [K]
    return torch.sum(p * torch.log(p + 1e-8))


def sharp_assignment_loss(assign):
    return -(assign * torch.log(assign + 1e-8)).sum(dim=-1).mean()
