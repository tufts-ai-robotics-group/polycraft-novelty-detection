import torch.nn as nn

from polycraft_nov_det.models.dino.vision_transformer import DINOHead, VisionTransformer


class DinoWithHead(nn.Module):
    def __init__(self, dino_backbone: VisionTransformer):
        super().__init__()
        self.backbone = dino_backbone
        self.head = DINOHead()
        self.freeze_upper_blocks()

    def freeze_upper_blocks(self):
        # freeze all but last block of backbone and head
        for name, param in self.backbone.named_parameters():
            if "blocks.11" in name:
                break
            param.requires_grad = False

    def forward(self, x):
        out = self.backbone(x)
        out = self.head(out)
        return out
