
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image

from predict import load_checkpoint, denormalize_image, predict
from classification.explainability import (
    calculate_heatmap,
    calculate_attention_rollout,
)


IMAGE_PATH = Path("data/Apple/Apple_rust/image (2).JPG")

CHECKPOINTS = {
    "Custom CNN": Path(
        "checkpoints/custom_cnn/baseline_apple/best.pt"
    ),
    "Custom ViT": Path(
        "checkpoints/custom_vit/baseline_apple/best.pt"
    ),
}

OUTPUT_DIR = Path("visualisations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THEMES = {
    "light": {
        "text": "#242424",
    },
    "dark": {
        "text": "#E5E7EB",
    },
}


def get_prediction_visuals(checkpoint_path, original_image, device):
    """Generate prediction and explanation images for one model."""
    loaded = load_checkpoint(checkpoint_path, device)

    transformed = loaded["transform"](original_image)

    denorm = denormalize_image(
        transformed,
        loaded["preprocessing"]["mean"],
        loaded["preprocessing"]["std"],
    ).cpu().numpy()

    predicted_index, predicted_class = predict(
        transformed,
        device,
        loaded["model"],
        loaded["index_to_class"],
    )

    heatmap = calculate_heatmap(
        loaded["model"],
        transformed,
        predicted_index,
        loaded["model_type"],
    )

    gradcam = show_cam_on_image(
        denorm,
        heatmap,
        use_rgb=True,
    )

    rollout = None

    if loaded["model_type"] == "custom_vit":
        rollout_heatmap = calculate_attention_rollout(
            loaded["model"]
        ).detach().cpu().numpy()

        rollout = show_cam_on_image(
            denorm,
            rollout_heatmap,
            use_rgb=True,
        )

    return {
        "predicted_class": predicted_class,
        "gradcam": gradcam,
        "rollout": rollout,
    }


def save_comparison(original, cnn, vit, theme_name):
    """Save a theme-specific figure with a transparent background."""
    text_color = THEMES[theme_name]["text"]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(12, 3.4),
    )

    fig.patch.set_alpha(0)

    images = [
        original,
        cnn["gradcam"],
        vit["gradcam"],
        vit["rollout"],
    ]

    titles = [
        "Original",
        "Custom CNN · Grad-CAM",
        "Custom ViT · Grad-CAM",
        "Custom ViT · Attention rollout",
    ]

    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image)
        ax.set_title(
            title,
            fontsize=11,
            color=text_color,
            pad=12,
        )
        ax.axis("off")
        ax.set_facecolor("none")

    if cnn["predicted_class"] == vit["predicted_class"]:
        prediction_text = (
            f"Both models predicted: "
            f"{cnn['predicted_class'].replace('_', ' ')}"
        )
    else:
        prediction_text = (
            f"CNN: {cnn['predicted_class'].replace('_', ' ')}"
            f"  |  "
            f"ViT: {vit['predicted_class'].replace('_', ' ')}"
        )

    fig.text(
        0.5,
        0.08,
        prediction_text,
        ha="center",
        va="center",
        fontsize=11,
        color=text_color,
    )

    fig.subplots_adjust(
        left=0.01,
        right=0.99,
        top=0.87,
        bottom=0.18,
        wspace=0.12,
    )

    for extension in ("svg", "png"):
        output_path = (
            OUTPUT_DIR
            / f"prediction_comparison_{theme_name}.{extension}"
        )

        fig.savefig(
            output_path,
            dpi=250,
            bbox_inches="tight",
            transparent=True,
        )

        print(f"Saved: {output_path}")

    plt.close(fig)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    with Image.open(IMAGE_PATH) as image:
        original = image.convert("RGB")

    cnn = get_prediction_visuals(
        CHECKPOINTS["Custom CNN"],
        original,
        device,
    )

    vit = get_prediction_visuals(
        CHECKPOINTS["Custom ViT"],
        original,
        device,
    )

    for theme_name in THEMES:
        save_comparison(
            original,
            cnn,
            vit,
            theme_name,
        )

    print(f"CNN prediction: {cnn['predicted_class']}")
    print(f"ViT prediction: {vit['predicted_class']}")


if __name__ == "__main__":
    main()
