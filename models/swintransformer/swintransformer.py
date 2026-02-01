from transformers import SwinModel
import torch
import torch.nn as nn


class SwinFeatureExtractor(nn.Module):
    def __init__(
        self, model_name="microsoft/swin-tiny-patch4-window7-224", pretrained=True
    ):
        super().__init__()
        self.backbone = (
            SwinModel.from_pretrained(model_name)
            if pretrained
            else SwinModel.from_pretrained(model_name)
        )
        self.hidden = self.backbone.config.hidden_size  # 768 for tiny
        # SwinModel 输出 last_hidden_state: [B, num_patches, hidden]
        # 我们做 mean pooling 得到 [B, hidden]

    def forward(self, x):
        out = self.backbone(pixel_values=x)  # x: [B,3,224,224]
        feat = out.last_hidden_state.mean(dim=1)  # [B, hidden]
        return feat


class DualImageSwinClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_name = config.model_pretrain
        # e.g., "microsoft/swin-tiny-patch4-window7-224"
        self.pretrained = True
        self.freeze_backbone = True
        self.encoder = SwinFeatureExtractor(self.model_name, pretrained=self.pretrained)

        if self.freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()  # 注意：训练时你仍可以在 forward 外部切换 model.train()

        in_dim = self.encoder.hidden * 2  # heat + overlay concat
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Dropout(0.2),
            nn.Linear(64, 1),  # logit
        )

    def forward(self, x):
        # x 可以是 tuple/list: (x_heat, x_over)
        x_heat, x_over = x
        f1 = self.encoder(x_heat)  # [B,H]
        f2 = self.encoder(x_over)  # [B,H]
        feat = torch.cat([f1, f2], dim=1)
        logit = self.head(feat).squeeze(-1)
        return logit
