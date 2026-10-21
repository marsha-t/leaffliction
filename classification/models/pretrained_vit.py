import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class Pretrained_VisionTransformers(nn.Module):
    """
    Implements a pretrained ViT-B/16 for leaf disease classification.

    Uses ImageNet pretrained weights, freezes the backbone,
    and replaces the classification head to predict 8 classes.
    """
    def __init__(self):
        super().__init__()
        weights = ViT_B_16_Weights.DEFAULT
        self.vit = vit_b_16(weights=weights)
        for param in self.vit.parameters():
            param.requires_grad = False
        self.vit.heads.head = nn.Linear(768, 8)

    def forward(self, x):
        """
        Apply foreward pass for pretrained vit
        """
        return self.vit(x)
