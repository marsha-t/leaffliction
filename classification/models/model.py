from .custom_cnn import CustomCNN
from .custom_vit import VisionTransformer
from .pretrained_vit import Pretrained_VisionTransformers

def create_model(model_type, num_classes):
    """
    Create and return model based on specified model_type

    Args:
        model_type (str): Type of model to create
        num_classes (int): Number of output classes

    Returns:
        nn.Module: The initialized PyTorch model

    Raises:
        ValueError: If the specified model type is not supported
    """
    if model_type == 'custom_cnn':
        return CustomCNN(num_classes)
    # if model_type == 'pretrained_cnn':
    if model_type == 'custom_vit':
        return VisionTransformer(patch_size=16)
    if model_type == 'pretrained_vit':
            return Pretrained_VisionTransformers()
    raise ValueError(f'Unknown model type: {model_type}')
