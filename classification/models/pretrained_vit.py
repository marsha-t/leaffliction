import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class Pretrained_VisionTransformers(nn.Module):
    """
    Implements a pretrained ViT-B/16 for leaf disease classification.

    Uses ImageNet pretrained weights by default and always freezes the backbone.
    The replacement classification head predicts num_classes classes.
    """
    def __init__(self, num_classes, pretrained=True):
        """Initialise ViT-B/16 and its trainable classification head.

        Args:
            num_classes (int): Number of output classes.
            pretrained (bool): Load ImageNet weights when True. Set False
                when reconstructing a model before loading a checkpoint.
        """
        super().__init__()
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.vit = vit_b_16(weights=weights)
        for param in self.vit.parameters():
            param.requires_grad = False
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Return classification logits with shape (B, num_classes).
        """
        return self.vit(x)
