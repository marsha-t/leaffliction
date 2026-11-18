#!/usr/bin/env python3
"""Create a portfolio contact sheet from the Apple and Grape datasets.

Run from the Leaffliction project root:
    python visualisations/Plot_Dataset_Examples.py
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageOps

CLASSES = {
    "Apple": [
        ("Apple_Black_rot", "Black rot"),
        ("Apple_healthy", "Healthy"),
        ("Apple_rust", "Rust"),
        ("Apple_scab", "Scab"),
    ],
    "Grape": [
        ("Grape_Black_rot", "Black rot"),
        ("Grape_Esca", "Esca"),
        ("Grape_healthy", "Healthy"),
        ("Grape_spot", "Spot"),
    ],
}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
THEMES = {"light": "#202124", "dark": "#F3F4F6"}


def select_images(data_dir, seed):
    selected = []
    for plant, classes in CLASSES.items():
        for folder, label in classes:
            directory = data_dir / plant / folder
            if not directory.is_dir():
                raise FileNotFoundError(f"Missing class directory: {directory}")
            candidates = sorted(
                path for path in directory.iterdir()
                if path.is_file()
                and path.suffix.lower() in EXTENSIONS
                and "_aug_" not in path.stem.lower()
            )
            if not candidates:
                raise FileNotFoundError(f"No original images found in {directory}")
            # A class-specific RNG ensures stable choices independent of other classes.
            rng = random.Random(f"{seed}:{plant}:{folder}")
            chosen = rng.choice(candidates)
            selected.append((plant, label, chosen))
    return selected


def make_square(image, size=300):
    """Letterbox rather than crop so the entire leaf remains visible."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    background = Image.new("RGB", (size, size), (238, 238, 238))
    background.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return background


def save_sheet(selected, output_dir, theme):
    color = THEMES[theme]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.9))
    fig.patch.set_alpha(0)

    for ax, (plant, label, path) in zip(axes.flat, selected):
        with Image.open(path) as image:
            panel = make_square(image)
        ax.imshow(panel)
        ax.set_title(label, fontsize=12, color=color, pad=9)
        ax.axis("off")
        ax.set_facecolor("none")

    fig.text(0.015, 0.72, "Apple", ha="left", va="center", color=color,
             fontsize=14, fontweight="bold", rotation=90)
    fig.text(0.015, 0.28, "Grape", ha="left", va="center", color=color,
             fontsize=14, fontweight="bold", rotation=90)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.97, bottom=0.03,
                        wspace=0.09, hspace=0.21)

    for extension in ("png", "svg"):
        path = output_dir / f"dataset_examples_{theme}.{extension}"
        fig.savefig(path, dpi=220, bbox_inches="tight", transparent=True)
        print(f"Saved: {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"),
                        help="Dataset root (default: data)")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent,
                        help="Output folder (default: script folder)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for reproducible example selection")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    selected = select_images(args.data, args.seed)
    for plant, label, path in selected:
        print(f"{plant} / {label}: {path}")
    for theme in THEMES:
        save_sheet(selected, args.output, theme)


if __name__ == "__main__":
    main()
