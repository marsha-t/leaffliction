#! /usr/bin/env python3
from PIL import Image
import argparse
from pathlib import Path
import random

from analysis import validate_directory
from augmentation import (
    flip,
    rotate,
    shear,
    skew,
    crop,
    elastic_distortion,
    grid_distortion
)


AUGMENTATIONS = [
    ('Flip', lambda image: flip(image, 'v')),
    ('Rotate', lambda image: rotate(image, 90)),
    ('Skew', skew),
    ('Shear', lambda image: shear(image, 0.3, horizontal=True)),
    ('Crop', lambda image: crop(image, (0, 0, 100, 100))),
    ('ElasticDistortion', lambda image: elastic_distortion(image)),
    ('GridDistortion', lambda image: grid_distortion(image, 4, 30))
]

def parse_args():
    """
    Parse command-line arguments

    Returns:
        argparse.Namespace: parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description='Augmentation')
    parser.add_argument('image_path', help='image path')
    args = parser.parse_args()
    return args


def save_augmented_image(image, path, name):
    """
    Save augmented image in original data folder and in
        augmented_directory folder

    Args:
        image (Image.Image): augmented image
        path (str): filepath of original image
        name (str): augmentation type

    Raises:
        ValueError if input file does not come from /data
    """
    original_path = Path(path)
    stem = original_path.stem
    suffix = original_path.suffix
    filename = f"{stem}_{name}{suffix}"

    # Save in /data
    input_dir = original_path.parent
    image.save(input_dir / filename)

    # Save in /augmented_directory
    try:
        relative_dir = original_path.parent.relative_to('data')
    except ValueError:
        raise ValueError('Input image must be inside /data')
    output_dir = Path('augmented_directory') / relative_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    image.save(output_dir / filename)


def is_augmented_image(path):
    """
    Determine whether image is an augmented image
    - Augmented images are identified by suffix matching one
    of the supported augmentation names (e.g. '_Flip', '_Rotate')

    Args:
        path (Path): Path to an image

    Returns:
        bool: True if the image appears to be an augmented image,
            False otherwise
    """
    stem = path.stem
    return any(
        stem.endswith(f"_{name[0]}") for name in AUGMENTATIONS
    )


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


def calculate_target(distribution):
    """
    Calculate number of augmented images required for each class

    Args:
        dataset (dict): Dataset information produced by scan_dataset()

    Returns:
        dict: Mapping of class names to the number of additional images
            required for balancing.
    """
    max_value = max(distribution.values())
    target = {}
    for class_name, count in max_value.items():
        target[class_name] = max_value - count
    return target


def execute_augmentation_plan(root, plan):
    """
    Generate augmented images until required number of additional images has
    been generated
    - Original images and augmentation techniques are randomly selected

    Args:
        root (Path): Root directory of the dataset
        plan (dict): Mapping of class names to the number of images to
            generate
    """
    for class_name, target in plan: # TODO update so that images are selected in rounds and augmentations are mindful of existing augmentations done 
        images = list_original_images(root / class_name)
        generated = 0
        while generated < target:
            image = random.choice(images)
            augmentation = random.choice(AUGMENTATIONS)
            augmented = augmentation[1](image)
            save_augmented_image(augmented, root, augmentation)


def scan_dataset(root):
    """
    Validate directory structure, separate original and previously augmented
    images, and records the augmentation history for each original image

    Args:
        root (Path): Root directory of the dataset.

    Returns:
        dict: Dataset information grouped by class, including the
            original images and existing augmentations
    """
    validate_directory(root)


def augment_directory(root):
    """
    Balance dataset using image augmentation

    Args:
        root (Path): Root directory of the dataset
    """
    dataset = scan_dataset(root)
    plan = calculate_target(dataset) 
    execute_augmentation_plan(root, plan)


def main():
    try:
        args = parse_args()

        path = Path(args.image_path)
        if path.is_file():
            augment_image(path)
        elif path.is_dir():
            augment_directory(path)

    except Exception as e:
        print("there is an issue :", e)


if __name__ == "__main__":
    main()
