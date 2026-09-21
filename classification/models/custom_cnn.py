import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Extract features with two convolution, batch normalisation, and ReLU stages

    A final 2-by-2 max-pooling layer reduces the spatial dimensions
    """
    def __init__(self, in_channels, out_channels):
        """
        Initialise the convolutional block

        Args:
            in_channels (int): Number of input feature channels
            out_channels (int): Number of channels produced by both convolutions
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        Extract features and downsample spatial dimensions

        Args:
            x (torch.Tensor): Input with shape (N, in_channels, H, W)

        Returns:
            torch.Tensor: Features with shape
                (N, out_channels, H // 2, W // 2)
        """
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.pool(x)

        return x


class CustomCNN(nn.Module):
    """
    Classify RGB leaf images using four convolutional blocks.

    Global average pooling produces a 256-element feature vector per image.
    A linear layer maps features to unnormalised class scores.
    """
    def __init__(self, num_classes):
        """
        Initialise feature extractor and classification layer

        Args:
            num_classes (int): Number of output classes
        """
        super().__init__()
        self.block1 = ConvBlock(3, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)
        self.block4 = ConvBlock(128, 256)

        # reduce each feature map to 1 value
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        Compute class logits for a batch of RGB images

        Args:
            x (torch.Tensor): Image batch with shape (N, 3, H, W)

        Returns:
            torch.Tensor: Unnormalised class scores with shape (N, num_classes)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)  # [B, 256, 1, 1]
        x = torch.flatten(x, 1)  # [B, 256]
        x = self.classifier(x)
        return x
