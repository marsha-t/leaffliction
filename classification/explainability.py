from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import math
import torch.nn.functional as F
import torch


def calculate_attention_rollout(model, image_size=(224, 224)):
    """
    Generate an attention rollout heatmap for a Vision Transformer (ViT).

    Args:
        model: Trained ViT model with stored attention weights
        image_size: Target heatmap dimensions (H, W)

    Returns:
        np.ndarray: Attention rollout heatmap (H, W), normalized to [0, 1]
    """
    attention_matrices = [
        block.last_attention
        for block in model.encoders
    ]
    rollout = None
    for attention in attention_matrices:
        attention = torch.mean(attention, dim=1)
        identity_matrix = torch.eye(
                                    attention.shape[1],
                                    device=attention.device,
                                    dtype=attention.dtype).unsqueeze(0)
        attention = attention + identity_matrix
        attention = attention / torch.sum(attention, dim=-1, keepdim=True)
        if rollout is None:
            rollout = attention
        else:
            rollout = torch.bmm(attention, rollout)
    cls_token = rollout[:, 0, 1:]
    B, N = cls_token.shape
    grid_size = math.isqrt(N)
    assert grid_size**2 == N
    heatmap = cls_token.reshape(B, 1, grid_size, grid_size)
    heatmap = F.interpolate(heatmap, size=image_size, mode="bilinear",
                            align_corners=False)
    heatmap = heatmap[:, 0]
    heatmap = heatmap - heatmap.amin(
        dim=(1, 2),
        keepdim=True
    )
    heatmap = heatmap/(heatmap.amax(dim=(1, 2), keepdim=True) + 1e-8)
    print("attention.shape", heatmap.shape)
    return heatmap.squeeze(0)


def reshape_transform(tensor):
    """
    Reshape ViT patch embeddings into a spatial feature map for Grad-CAM.

    Args:
        tensor: ViT output tensor (B, N+1, C), including the CLS token

    Returns:
        torch.Tensor: Reshaped feature map (B, C, H, W)
    """
    tensor = tensor[:, 1:, :]
    B, N, D = tensor.shape
    grid_size = math.isqrt(N)
    assert grid_size ** 2 == N
    tensor = tensor.reshape(B, grid_size, grid_size, D)
    tensor = tensor.permute(0, 3, 1, 2)
    return tensor


def calculate_heatmap(model, input_tensor, predicted_index, model_type):
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
