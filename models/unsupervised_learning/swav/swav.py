import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim=30,
        embed_dim=128,
        num_layers=4,
        num_heads=8,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super(TransformerEncoder, self).__init__()
        # 将每个脉冲的特征维度从input_dim投影到embed_dim
        self.input_proj = nn.Linear(input_dim, embed_dim)
        # 定义Transformer编码器层和编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        # 不使用位置编码，因此这里不定义positional encoding

    def forward(self, x):
        """
        参数:
        x: 张量, shape [B, 160, 30], B为批量大小, 160为序列长度, 30为每个脉冲的特征维度
        返回:
        seq_rep: 张量, shape [B, embed_dim], 每个样本的序列全局表示向量
        """
        # 1. 投影输入到embed_dim
        # 输入 x shape: [B, seq_len, input_dim]
        seq_len = x.shape[1]  # 160
        x_emb = self.input_proj(x)  # [B, seq_len, embed_dim]
        # 2. 通过Transformer编码器层（由于batch_first=True，不需要转置张量）
        #    无位置编码，Transformer将基于内容建模序列
        #    TransformerEncoder输出 shape: [B, seq_len, embed_dim]
        enc_output = self.transformer_encoder(x_emb)  # [B, seq_len, embed_dim]
        # 3. 聚合序列输出为全局表示，这里使用平均池化（对序列维度取平均）
        seq_rep = enc_output.mean(dim=1)  # [B, embed_dim]
        return seq_rep


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(ProjectionHead, self).__init__()
        # 两层MLP：第一层Linear + BN + ReLU，第二层Linear直接输出
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # x形状: [B, in_dim]
        x = self.layer1(x)  # [B, hidden_dim], BN在批维度上归一化
        x = self.layer2(x)  # [B, out_dim]
        return x


class PrototypeLayer(nn.Module):
    def __init__(self, in_dim, num_prototypes):
        super(PrototypeLayer, self).__init__()
        # 原型向量作为线性层的权重（无偏置），输出维度=num_prototypes
        self.prototypes = nn.Linear(in_dim, num_prototypes, bias=False)

    def forward(self, x):
        # 计算每个样本与各原型的匹配分数（点积），输出形状 [B, num_prototypes]
        scores = self.prototypes(x)  # [B, K]
        return scores

    def normalize(self):
        # 将原型权重向量归一化为单位长度
        with torch.no_grad():
            w = self.prototypes.weight.data  # 权重形状 [num_prototypes, in_dim]
            # 按行进行L2归一化，每行对应一个原型向量
            w = F.normalize(w, dim=1, p=2)
            self.prototypes.weight.data.copy_(w)


class SwAVModel(nn.Module):
    def __init__(self, config):
        super(SwAVModel, self).__init__()

        self.num_prototypes = 100
        self.embed_dim = config.d_model
        self.hidden_dim = config.d_ff
        self.proj_out_dim = config.d_model
        # 1. 编码器: TransformerEncoder
        self.encoder = TransformerEncoder(
            input_dim=config.pulse_len,
            embed_dim=self.embed_dim,
            num_layers=config.n_layers,
            num_heads=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
        )
        # 2. 投影头: ProjectionHead
        self.projection = ProjectionHead(
            in_dim=self.embed_dim, hidden_dim=self.hidden_dim, out_dim=self.proj_out_dim
        )
        # 3. 原型层: PrototypeLayer
        self.prototypes = PrototypeLayer(
            in_dim=self.proj_out_dim, num_prototypes=self.num_prototypes
        )

    def forward(self, x):
        # 顺序: 编码器 -> 投影 -> 特征归一化 -> 原型匹配分数
        h = self.encoder(x)  # 编码器输出 [B, embed_dim]
        z = self.projection(h)  # 投影头输出 [B, proj_out_dim]
        z = F.normalize(z, dim=1, p=2)  # 单位化特征向量
        scores = self.prototypes(z)  # 原型分数 [B, num_prototypes]
        return scores
