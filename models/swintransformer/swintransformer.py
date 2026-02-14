from transformers import SwinModel, Swinv2Model
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwinFeatureExtractor(nn.Module):
    def __init__(
        self, model_name="microsoft/swin-tiny-patch4-window7-224", pretrained=True
    ):
        super().__init__()
        self.backbone = Swinv2Model.from_pretrained(model_name)
        self.hidden = self.backbone.config.hidden_size  # 768 for tiny
        # SwinModel 输出 last_hidden_state: [B, num_patches, hidden]
        # 我们做 mean pooling 得到 [B, hidden]

    def forward(self, x):
        out = self.backbone(pixel_values=x)  # x: [B,3,224,224]
        feat = out.last_hidden_state.mean(dim=1)  # [B, hidden]
        return feat


class GatedAttnMIL(nn.Module):
    """
    Gated Attention MIL pooling.
    x: [B, K, D]
    returns:
      z: [B, D]
      attn: [B, K]
    """

    def __init__(self, d_model: int, attn_dropout: float = 0.0):
        super().__init__()
        self.v = nn.Linear(d_model, d_model)
        self.u = nn.Linear(d_model, d_model)
        self.w = nn.Linear(d_model, 1)
        self.attn_dropout = (
            nn.Dropout(attn_dropout) if attn_dropout > 0 else nn.Identity()
        )

    def forward(self, x):
        # a: [B, K]
        a = self.w(torch.tanh(self.v(x)) * torch.sigmoid(self.u(x))).squeeze(-1)
        attn = F.softmax(a, dim=1)  # [B, K]
        attn = self.attn_dropout(attn)

        # z: [B, D]
        z = torch.sum(attn.unsqueeze(-1) * x, dim=1)
        return z


class DualImageSwinClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_name = config.model_pretrain
        # e.g., "microsoft/swin-tiny-patch4-window7-224"
        self.pretrained = True
        self.freeze_backbone = False
        self.encoder = SwinFeatureExtractor(self.model_name, pretrained=self.pretrained)
        self.fuse = "concat"  # or 'add'

        if self.fuse == "concat":
            self.fuse_proj = nn.Sequential(
                nn.Linear(self.encoder.hidden * 2, self.encoder.hidden),
                nn.GELU(),
                nn.Dropout(0.1),
            )
        else:
            self.fuse_proj = nn.Identity()

        if self.freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.mil = GatedAttnMIL(
            self.encoder.hidden, attn_dropout=0.1
        )  # simple MIL pooling: mean over instances

        in_dim = self.encoder.hidden  # heat + overlay concat
        # in_dim = self.encoder.hidden
        # self.head = nn.Sequential(
        #     nn.Linear(in_dim, 256),
        #     nn.GELU(),
        #     nn.LayerNorm(256),
        #     nn.Dropout(0.2),
        #     nn.Linear(256, 64),
        #     nn.GELU(),
        #     nn.LayerNorm(64),
        #     nn.Dropout(0.2),
        #     nn.Linear(64, 1),  # logit
        # )
        self.head = nn.Linear(in_dim, 1)  # 简单线性分类头

    def forward(self, x):
        # x 可以是 tuple/list: (x_heat, x_over)
        x_heat, x_over = x
        B, K, C, H, W = x_heat.shape
        # 1) flatten instances: [B*K, C, H, W]
        heat_flat = x_heat.reshape(B * K, C, H, W)
        over_flat = x_over.reshape(B * K, C, H, W)

        # 2) shared backbone feature extraction: [B*K, D]
        h_feat = self.encoder(heat_flat)  # expect [B*K, D]
        o_feat = self.encoder(over_flat)  # [B*K, D]

        # 3) fuse heat/overlay per instance
        if self.fuse == "add":
            inst_feat = h_feat + o_feat  # [B*K, D]
        else:
            inst_feat = torch.cat([h_feat, o_feat], dim=-1)  # [B*K, 2D]
            inst_feat = self.fuse_proj(inst_feat)  # [B*K, D]

        # 4) reshape back to bag: [B, K, D]
        bag_feat = inst_feat.reshape(B, K, self.encoder.hidden)

        # 5) MIL pooling: [B, D]
        z = bag_feat.mean(dim=1)  # [B, D]

        # 6) classifier: [B]
        logit = self.head(z).squeeze(-1)
        return logit
