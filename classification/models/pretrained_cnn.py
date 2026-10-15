import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class PretrainedResNet18(nn.Module):
    def __init__(self, num_classes, pretrained=True, freeze_backbone=True):
        super().__init__()

        self.freeze_backbone = freeze_backbone
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.resnet = resnet18(weights=weights)

        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False

        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone and mode:
            for module in self.resnet.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def forward(self, x):
        return self.resnet(x)
