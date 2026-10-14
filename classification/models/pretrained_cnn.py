import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class PretrainedResNet18(nn.Module):
    def __init__(self, num_classes, freeze_backbone=True):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT
        self.resnet = resnet18(weights=weights)

        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False

        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
