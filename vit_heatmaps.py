from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import math
import torch.nn.functional as F
import torch


def calculate_attention_rollout(model, image_size=(224, 224)):
    attention_matrices = [
        block.last_attention
        for block in model.encoders
    ]
    rollout = None
    for attention in attention_matrices:
        attention = attention.mean(dim=1)
        B, N, _ = attention.shape
        identity = torch.eye(
            N,
            device=attention.device,
            dtype=attention.dtype
        ).unsqueeze(0)
        attention = attention + identity
        attention = attention / attention.sum(
            dim=-1,
            keepdim=True
        )
        if rollout is None:
            rollout = attention
        else:
            rollout = torch.bmm(attention, rollout)
    cls_attention = rollout[:, 0, 1:]
    B, num_patches = cls_attention.shape

    grid_size = math.isqrt(num_patches)

    assert grid_size ** 2 == num_patches

    heatmap = cls_attention.reshape(
        B, 1, grid_size, grid_size
    )
    heatmap = F.interpolate(
        heatmap,
        size=image_size,
        mode="bilinear",
        align_corners=False
    )
    heatmap = heatmap[:, 0]
    heatmap = heatmap - heatmap.amin(
        dim=(1, 2),
        keepdim=True
    )
    heatmap = heatmap / (
        heatmap.amax(dim=(1, 2), keepdim=True) + 1e-8
    )
    return heatmap.squeeze(0)


def reshape_transform(tensor):
    tensor = tensor[:, 1:, :]
    B, N, D = tensor.shape
    grid_size = math.isqrt(N)
    assert grid_size ** 2 == N
    tensor = tensor.reshape(B, grid_size, grid_size, D)
    tensor = tensor.permute(0, 3, 1, 2)
    return tensor


def calculate_heat_map(model, input_tensor, predicted_index):
    input_tensor = input_tensor.unsqueeze(0)
    target_layer = [model.encoders[-1].layer_norm1]
    cam = GradCAM(model=model,
                  target_layers=target_layer,
                  reshape_transform=reshape_transform)
    targets = [ClassifierOutputTarget(predicted_index)]
    print("predicted_index ", predicted_index)
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    heatmap = grayscale_cam[0]
    print("gray scale shape is ", grayscale_cam.shape)
    print("heatmap shape is ", heatmap.shape)
    return heatmap