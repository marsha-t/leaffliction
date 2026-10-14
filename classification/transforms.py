from torchvision import transforms
from torchvision.models import ResNet18_Weights


def create_transform(model_type, mean=None, std=None):
    """
    Create image transformation pipeline for the specified model type

    Args:
        model_type (str): Type of model for which to create the transform.
        mean (sequence of float): Per-channel mean values used for
            normalization
        std (sequence of float): Per-channel standard deviation values used for
            normalization.

    Returns:
        transforms.Compose: The image transformation pipeline.

    Raises:
        ValueError: If the specified model type is not supported.
    """
    if model_type == 'custom_cnn':
        return create_custom_cnn_transform(mean, std)
    elif model_type == 'pretrained_cnn':
        return create_pretrained_cnn_transform()
    # elif model_type == 'custom_vit'
    # elif model_type == 'pretrained_vit'
    raise ValueError(f'Unknown model type: {model_type}')


def create_custom_cnn_transform(mean, std):
    """
    Create image transformation pipeline for custom CNN model

    Args:
        mean (sequence of float): Per-channel mean values used for
            normalization
        std (sequence of float): Per-channel standard deviation values used for
            normalization

    Returns:
        transforms.Compose: Transformation pipeline for the custom CNN
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def create_pretrained_cnn_transform():
    weights = ResNet18_Weights.DEFAULT
    return weights.transforms()
