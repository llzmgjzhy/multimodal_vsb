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
            (stats["assign"] * torch.cdist(stats["z"], stats["prototypes"]).pow(2))
            .sum(-1)
            .mean()
            + 0.2
            * F.kl_div(
                (stats["assign"].mean(dim=(0, 1)) + 1e-8).log(),
                torch.full_like(
                    stats["assign"].mean(dim=(0, 1)), 1.0 / stats["assign"].size(-1)
                ),
                reduction="batchmean",
            )
        )

    if loss_type == "contrastive":
        return contrastive_loss

    else:
        raise ValueError(f"Loss module for '{loss_type}' does not exist")


def contrastive_loss(z1, z2, temperature=0.2):
    """
    z1, z2: [B, D] 两个增广视图的表示
    """
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    z = F.normalize(z, dim=1)  # 单位向量

    # 相似度矩阵
    sim = torch.matmul(z, z.T) / temperature  # [2B, 2B]

    # 正样本对位置：i<->i+B
    labels = torch.arange(B, device=z.device)
    labels = torch.cat([labels + B, labels], dim=0)

    # 遮挡对角线（自己与自己）
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -1e9)

    # 计算交叉熵损失
    loss = F.cross_entropy(sim, labels)
    return loss
