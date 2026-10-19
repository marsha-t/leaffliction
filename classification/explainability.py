from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def calculate_heat_map(model, input_tensor, predicted_index, model_type):
    """
    Generate a Grad-CAM heatmap for a CNN or ViT

    Args:
        model: Trained model in evaluation mode
        input_tensor: Preprocessed image tensor (C, H, W)
        predicted_index: Class index to explain
        model_type: Model architecture type

    Returns:
        np.ndarray: Grad-CAM heatmap (H, W), normalised to [0, 1]
    """
    if model_type == "custom_cnn":
        target_layers = [model.block4]
        reshape_fn = None

    elif model_type == 'pretrained_cnn':
        target_layers = [model.resnet.layer4[-1]]
        reshape_fn = None

    elif model_type == "custom_vit":
        target_layers = [model.encoders[-1].layer_norm1]
        reshape_fn = reshape_transform

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    device = next(model.parameters()).device
    input_tensor = input_tensor.unsqueeze(0).to(device)
    input_tensor = input_tensor.detach().requires_grad_(True)

    targets = [ClassifierOutputTarget(predicted_index)]

    with GradCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=reshape_fn
    ) as cam:
        heatmap = cam(
            input_tensor=input_tensor,
            targets=targets
        )[0]

    return heatmap
