from .custom_cnn import CustomCNN
from .pretrained_cnn import PretrainedResNet18
from .custom_vit import VisionTransformer
from .pretrained_vit import Pretrained_VisionTransformers


def create_model(model_type, num_classes, pretrained=True):
    """
    Create and return model based on specified model_type

    Args:
        model_type (str): Type of model to create
        num_classes (int): Number of output classes
        pretrained (bool): Load ImageNet weights for pretrained architectures.
            Ignored for custom architectures.

    Returns:
        nn.Module: The initialized PyTorch model

    Raises:
        ValueError: If the specified model type is not supported
    """
    if model_type == 'custom_cnn':
        return CustomCNN(num_classes)
    if model_type == 'pretrained_cnn':
        return PretrainedResNet18(num_classes, pretrained=pretrained)
    if model_type == 'custom_vit':
        return VisionTransformer(num_classes)
    if model_type == 'pretrained_vit':
        return Pretrained_VisionTransformers(
            num_classes, pretrained=pretrained
        )

    raise ValueError(f'Unknown model type: {model_type}')
