import argparse
from pathlib import Path
import torch
from PIL import Image
import matplotlib.pyplot as plt

from classification.models import create_model
from classification.transforms import create_transform
import numpy as np
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import math
import torch.nn.functional as F


def parse_arguments():
    """
    Parse image path and required checkpoint path from command line

    Returns:
        argparse.Namespace: Parsed prediction arguments
    """
    parser = argparse.ArgumentParser(
        description='Predict the leaf-disease class of an image'
    )
    parser.add_argument(
        'image',
        help='Path to image file'
    )
    parser.add_argument(
        '--checkpoint',
        required=True,
        help='Path to checkpoint file'
    )
    return parser.parse_args()


def load_checkpoint(checkpoint_path, device):
    """
    Reconstruct the model from checkpoint metadata, load its weights,
        move it to the requested device, and enable evaluation mode
    Also load preprocessing configuration

    Args:
        checkpoint_path (str or Path): Path to a training checkpoint
        device (torch.device): Device on which to load and run the model

    Returns:
        dict:
            - model: Restored model in evaluation mode
            - model_type: Model type name
            - preprocessing: Preprocessing configuration
            - transform: Image preprocessing transform
            - index_to_class: Mapping from class indices to class names
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    model = create_model(
        checkpoint['model_type'],
        len(checkpoint['class_to_index']),
        pretrained=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    preprocessing = checkpoint['preprocessing']

    transform = create_transform(
        model_type=checkpoint['model_type'],
        preprocessing=preprocessing
    )

    index_to_class = {
        index: class_name
        for class_name, index in checkpoint['class_to_index'].items()
    }

    return {
        'model': model, 
        'model_type': checkpoint['model_type'],
        'preprocessing': preprocessing,
        'transform': transform,
        'index_to_class': index_to_class
    }


def denormalize_image(tensor, mean, std):
    """
    Reverse channel-wise normalisation for visualisation

    Args:
        tensor: Normalised image tensor of shape (C, H, W)
        mean: Per-channel means with shape (C, )
        std: Per-channel standard deviations with shape (C, )

    Returns:
        Denormalised image tensor of shape (H, W, C) with values in [0, 1]
    """
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)

    # Reshape from (C, ) to (C, 1, 1)
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)

    denorm = tensor * std + mean
    denorm = denorm.clamp(0, 1)

    return denorm.permute(1, 2, 0)


def predict(transformed_image, device, model, index_to_class):
    """
    Predict class of a preprocessed image using a trained model

    Args:
        transformed_image (torch.Tensor): Preprocessed image tensor
        device (torch.device): Device on which to run inference
        model (torch.nn.Module): Trained model in evaluation mode
        index_to_class (dict): Mapping from class indices to class names

    Returns:
        str: Predicted class name
    """
    input_tensor = transformed_image.unsqueeze(0)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        predicted_index = logits.argmax(dim=1).item()

    return index_to_class[predicted_index]


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
    return heatmap[0]


def display_prediction(
    model_type, original_image, denorm_image, predicted_class, model,transformed_image
):
    """
    Display original and transformed images alongside the
    model's predicted class.

    Args:
        model_type (str): Model type name
        original_image (PIL.Image.Image): Original RGB image
        denorm_image (torch.Tensor): Denormalised image tensor
            of shape (H, W, C), with values in [0, 1]
        predicted_class (str): Predicted class name
    """
    fig, axes = plt.subplots(1, 4, figsize=(10, 5))
    fig.suptitle(f'Classification with {model_type}')
    axes[0].imshow(original_image)
    axes[1].imshow(denorm_image)
    fig.subplots_adjust(bottom=0.15)
    for ax in axes:
        ax.set_xlim(0, 256)
        ax.set_ylim(256, 0)
        ax.set_aspect("equal")

    fig.text(
        0.5,
        0.05,
        f"Class predicted: {predicted_class}",
        ha="center",
        va="bottom"
    )
    if model_type == "custom_vit":
        model_image = np.array(
                               original_image.convert("RGB").resize((224, 224))
                               )
        heatmap = calculate_heat_map(
                                     model,
                                     transformed_image
                                    )
        axes[2].imshow(model_image)
        axes[2].imshow(
                        heatmap,
                        cmap="jet",
                        alpha=np.clip(heatmap * 0.6, 0, 0.6),
                        vmin=0,
                        vmax=1
                        )
        rollout_heatmap = calculate_attention_rollout(model)
        print(rollout_heatmap.shape)
        axes[3].imshow(model_image)
        axes[3]
        axes[3].imshow(
                        rollout_heatmap,
                        cmap="jet",
                        alpha=np.clip(heatmap * 0.6, 0, 0.6),
                        vmin=0,
                        vmax=1
                        )
    plt.show()


def reshape_transform(tensor):
    tensor = tensor[:, 1:, :]
    B, N, D = tensor.shape
    grid_size = math.isqrt(N)
    assert grid_size ** 2 == N
    tensor = tensor.reshape(B, grid_size, grid_size, D)
    tensor = tensor.permute(0, 3, 1, 2)
    return tensor
   

def calculate_heat_map(model, input_tensor):
    input_tensor = input_tensor.unsqueeze(0)
    target_layer = [model.encoders[-1].layer_norm1]
    cam = GradCAM(model=model,
                  target_layers=target_layer,
                  reshape_transform=reshape_transform)
    targets = [ClassifierOutputTarget(0)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    heatmap = grayscale_cam[0]
    print("gray scale shape is ", grayscale_cam.shape)
    print("heatmap shape is ",heatmap.shape)
    return heatmap


def draw_heatmap(image, heatmap):
    rgb_image = np.array(
                         image.resize((224, 224)),
                         dtype=np.float32
                        ) / 255.0

    visualization = show_cam_on_image(
        rgb_image,
        heatmap,
        use_rgb=True
    )

    Image.fromarray(visualization).show()



def main():
    """
    Load a checkpoint, run inference on an image, and display
    the original image, preprocessed image, and predicted class.
    """
    args = parse_arguments()
    image_path = Path(args.image).resolve()
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    loaded = load_checkpoint(args.checkpoint, device)

    with Image.open(image_path) as image:
        original_image = image.convert('RGB')
        transformed_image = loaded['transform'](original_image)
        print("the shape of the transformed image",transformed_image.shape)
        denorm_image = denormalize_image(
            transformed_image,
            loaded['preprocessing']['mean'],
            loaded['preprocessing']['std']
        )

    predicted_class = predict(
        transformed_image,
        device,
        loaded['model'],
        loaded['index_to_class']
    )

    display_prediction(
        loaded['model_type'],
        original_image,
        denorm_image,
        predicted_class,
        loaded['model'],
        transformed_image
    )


if __name__ == "__main__":
    main()
