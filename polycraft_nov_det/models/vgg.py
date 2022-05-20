import torch
import torch.nn as nn


class VGGPretrained(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
        # remove pretrained head
        self.backbone.classifier = self.backbone.classifier[:6]
        # add custom head
        self.head = nn.Linear(4096, num_classes)

    def forward(self, x):
        out = self.backbone(x)
        out = self.head(out)
        return out
