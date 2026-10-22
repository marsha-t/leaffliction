#! /usr/bin/env python3
from PIL import Image
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from analysis import analyse_directory, plot_charts
from augmentation.constants import AUGMENTATIONS
from augmentation.dataset import (
    scan_dataset,
    calculate_target,
    create_augmentation_plan,
    execute_augmentation_plan,
)
from augmentation.io import (
    save_augmented_image, is_augmented_image, copy_dataset
)


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
    saves results in source directory (not in augmented_dataset)

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
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'Augmentation of {path}')
        axes = axes.flatten()  # Convert to 1D array
        axes[0].imshow(image)
        axes[0].set_title('Original')

        for i, (name, function) in enumerate(AUGMENTATIONS, start=1):
            augmented = function(image)
            axes[i].imshow(augmented)
            axes[i].set_title(name)
            save_augmented_image(augmented, path, name)
        plt.show()


def balance_directory(root):
    """
    Balance dataset using image augmentation
    Augmented images are saved alongside originals.
    A balanced copy of the dataset is saved in augmented_directory

    Args:
        root (Path): Root directory of the dataset
    """
    dataset = scan_dataset(root)
    targets = calculate_target(dataset)
    plan = create_augmentation_plan(dataset, targets, AUGMENTATIONS, seed=42)
    augmented_root = Path('augmented_directory')
    copy_dataset(root, augmented_root)
    execute_augmentation_plan(
        plan,
        AUGMENTATIONS,
        data_root=root.parent,
        augmented_root=augmented_root
    )
    distribution = analyse_directory(augmented_root / root.name)
    plot_charts(distribution)


def main():
    """
    Apply augmentations to an image or balance a dataset from CLI arguments
    """
    try:
        args = parse_args()

        path = Path(args.path)
        if path.is_file():
            augment_image(path)
        elif path.is_dir():
            balance_directory(path)
        else:
            raise FileNotFoundError(f'Invalid path: {path}')

    except Exception as e:
        print("Exception:", e)


if __name__ == "__main__":
    main()
