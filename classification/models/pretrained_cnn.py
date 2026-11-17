import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class PretrainedResNet18(nn.Module):
    """
    Implements ResNet18 for leaf disease classification

    Uses ImageNet pretrained weights by default and always freezes the backbone.
    The replacement classification head predicts num_classes classes.
    Frozen backbone batch-normalisation layers remain in evaluation mode
    during training to preserve their running statistics.
    """

    def __init__(self, num_classes, pretrained=True):
        """
        Initialise ResNet18 and its trainable classification head

        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Load ImageNet weights when True. Set False
                when reconstructing a model before loading a checkpoint
        """
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.resnet = resnet18(weights=weights)

        for param in self.resnet.parameters():
            param.requires_grad = False

        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for module in self.resnet.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def forward(self, x):
        return self.resnet(x)
