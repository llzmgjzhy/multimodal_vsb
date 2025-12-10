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

    else:
        raise ValueError(f"Loss module for '{loss_type}' does not exist")
