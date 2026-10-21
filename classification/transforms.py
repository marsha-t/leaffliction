from torchvision import transforms
from torchvision.models import ResNet18_Weights, ViT_B_16_Weights

from .data_utils import (
    compute_training_stats
)


def get_preprocessing_config(
        model_type,
        train_samples=None,
        class_to_index=None):
    """
    Determine preprocessing configuration for given model:
    - For custom CNN, normalisation statistics are computed from training
        dataset
    - For pretrained models, normalisation statistics are retrieved for the
        given pretrained model

    Args:
        model_type (str): Model
        train_samples: Training samples used to compute statistics
            for custom models
        class_to_index (dict): Mapping of class names to indices

    Returns:
        dict: Preprocessing configuration containing mean and std,
            and the weights version for pretrained models

    Raises:
        ValueError: If the model type is unsupported or required
            training data is missing.
    """
    if model_type == 'custom_cnn' or model_type == "custom_vit":
        if train_samples is None or class_to_index is None:
            raise ValueError('Training data required for custom models')
        mean, std = compute_training_stats(train_samples, class_to_index)
        return {
            'mean': mean.tolist(),
            'std': std.tolist(),
        }
    if model_type == 'pretrained_cnn':
        weights = ResNet18_Weights.DEFAULT
        transform = weights.transforms()
        return {
            'weights': weights.name,
            'mean': list(transform.mean),
            'std': list(transform.std),
        }
    if model_type == 'pretrained_vit':
        weights = ViT_B_16_Weights.DEFAULT
        transform = weights.transforms()

        return {
            'weights': weights.name,
            'mean': list(transform.mean),
            'std': list(transform.std),
        }
    raise ValueError(f'Unknown model type: {model_type}')


def create_transform(model_type, preprocessing):
    """
    Create image transformation pipeline for the specified model type

    Args:
        model_type (str): Type of model for which to create the transform.
        preprocessing (dict): Preprocessing configuration containing
            normalisation statistics and weights version (for pretrained
            models)

    Returns:
        transforms.Compose: The image transformation pipeline

    Raises:
        ValueError: If the specified model type is not supported
    """
    if model_type == 'custom_cnn' or model_type == "custom_vit":
        return create_custom_transform(
            preprocessing['mean'], preprocessing['std']
        )
    elif model_type == 'pretrained_cnn':
        return ResNet18_Weights[preprocessing['weights']].transforms()
    elif model_type == 'pretrained_vit':
        return ViT_B_16_Weights[preprocessing['weights']].transforms()
    raise ValueError(f'Unknown model type: {model_type}')


def create_custom_transform(mean, std):
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
