from .custom_cnn import CustomCNN


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
    # if model_type == 'pretrained_vit':

    raise ValueError(f'Unknown model type: {model_type}')
