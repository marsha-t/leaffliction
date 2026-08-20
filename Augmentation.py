#! /usr/bin/env python3
from PIL import Image
import argparse
from pathlib import Path

from analysis import analyse_directory, plot_charts
from augmentation.constants import AUGMENTATIONS
from augmentation.dataset import (
    scan_dataset,
    calculate_target,
    create_augmentation_plan,
    execute_augmentation_plan
)
from augmentation.io import save_augmented_image, is_augmented_image


def parse_args():
    """
    Parse command-line arguments

    Returns:
        argparse.Namespace: parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description='Augmentation')
    parser.add_argument('path', help='image or directory path')
    args = parser.parse_args()
    return args


def augment_image(path):
    """
    Apply all supported augmentations to a single image and
    saves results in both the source directory and /augmented_dataset directory

    Args:
        path (Path): Path to the original image

    Raises:
        ValueError: If the input image has already been augmented
    """
    if is_augmented_image(path):
        raise ValueError(
            f"{path.name} appears to already be an augmented image"
        )

    with Image.open(path) as image:
        for name, function in AUGMENTATIONS:
            augmented = function(image)
            # augmented.show()
            save_augmented_image(augmented, path, name)


def augment_directory(root):
    """
    Balance dataset using image augmentation

    Args:
        root (Path): Root directory of the dataset
    """
    dataset = scan_dataset(root)
    targets = calculate_target(dataset)
    plan = create_augmentation_plan(dataset, targets, seed=42)
    execute_augmentation_plan(
        plan,
        data_root=root.parent,
        augmented_root=Path('augmented_directory')
    )
    distribution = analyse_directory(root)
    plot_charts(distribution)


def main():
    try:
        args = parse_args()

        path = Path(args.path)
        if path.is_file():
            augment_image(path)
        elif path.is_dir():
            augment_directory(path)

    except Exception as e:
        print("there is an issue :", e)


if __name__ == "__main__":
    main()
