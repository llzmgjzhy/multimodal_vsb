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

    if loss_type == "swav":
        return swav_loss

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


def sinkhorn(scores, eps=0.05, iters=3):
    """
    对输入的原型分数矩阵应用 Sinkhorn-Knopp 算法，返回双随机分配矩阵 Q。
    参数:
        scores: 张量 [B, K]，每个元素为样本与某原型的匹配分数。
        eps: Sinkhorn 正则化参数 (温度的作用类似, 默认0.05).
        iters: 迭代次数 (默认3).
    返回:
        Q: 张量 [B, K]，满足每个样本的分布和为1，每个原型在该批次中被选中的总和均约为1/K。
    """
    # 1. 计算 Q 初始化 = exp(scores / eps)
    Q = torch.exp(scores / eps)  # [B, K]
    # 转置为 [K, B] 方便按行(原型)和按列(样本)归一化
    Q = Q.t().contiguous()  # Q 现在 shape: [K, B]
    B = Q.shape[1]
    K = Q.shape[0]

    # 2. 初始化使每个样本（列）和为1
    Q /= torch.sum(Q, dim=0, keepdim=True)  # 每列归一化之和为1
    # 3. 迭代交替归一化行和列
    for _ in range(iters):
        # 3.1 归一化每一行（原型），使每行和为 1/K
        row_sum = torch.sum(Q, dim=1, keepdim=True)  # [K, 1]
        Q = Q / row_sum  # 每行和归一为1
        Q /= K  # 再除以K，使每行和为1/K
        # 3.2 归一化每一列（样本），使每列和为 1/B
        col_sum = torch.sum(Q, dim=0, keepdim=True)  # [1, B]
        Q = Q / col_sum  # 每列和归一为1
        Q /= B  # 再除以B，使每列和为1/B

    # 4. 最终将 Q 转置回 [B, K]
    Q = Q.t().contiguous()  # [B, K]
    return Q


def swav_loss(
    scores_view1, scores_view2, temperature=0.1, sinkhorn_eps=0.05, sinkhorn_iters=3
):
    """
    计算一对视角的 SwAV 损失。
    参数:
        scores_view1: 张量 [B, K], 视角1的原型匹配分数 (模型输出)
        scores_view2: 张量 [B, K], 视角2的原型匹配分数
        temperature: 温度系数 tau (用于 softmax)，默认0.1
        sinkhorn_eps: Sinkhorn 正则参数 (默认0.05)
        sinkhorn_iters: Sinkhorn 迭代次数 (默认3)
    返回:
        loss: 张量，标量，表示当前批次视角交换的平均损失
    """
    # 1. 计算 Sinkhorn 分配 (软标签 q)，不需要梯度
    with torch.no_grad():
        q_view1 = sinkhorn(
            scores_view1, eps=sinkhorn_eps, iters=sinkhorn_iters
        )  # [B, K]
        q_view2 = sinkhorn(
            scores_view2, eps=sinkhorn_eps, iters=sinkhorn_iters
        )  # [B, K]
        # 注：q_view1 和 q_view2 的每一行和=1, 它们在计算损失时作为目标分布使用

    # 2. 计算预测分布 p (经过温度softmax)，需要梯度用于更新编码器和原型参数
    p_view1 = F.softmax(scores_view1 / temperature, dim=1)  # [B, K]
    p_view2 = F.softmax(scores_view2 / temperature, dim=1)  # [B, K]

    # 3. 计算交换预测损失: -(q1 * log p2 + q2 * log p1) 的平均
    # 使用 sum(dim=1) 将每个样本的多个原型概率的交叉熵加总，然后再平均
    loss = -0.5 * (
        q_view1 * torch.log(p_view2 + 1e-10) + q_view2 * torch.log(p_view1 + 1e-10)
    )
    loss = loss.sum(dim=1).mean(dim=0)
    return loss
