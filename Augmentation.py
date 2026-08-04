#! /usr/bin/env python3
from PIL import Image
import argparse
from pathlib import Path

from augmentation import crop, shear, distortion


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


def main():
    try:
        args = parse_args()

        augmentations = [
            # ('Flip', flip),
            # ('Rotate', rotate),
            # ('Skew', skew),
            ('Shear', lambda image: shear(image, 0.3, horizontal=True)),
            ('Crop', lambda image: crop(image, (0, 0, 100, 100))),
            ('Distortion', distortion)
        ]

        with Image.open(args.image_path) as image:
            for name, function in augmentations:
                augmented = function(image)
                # augmented.show()
                save_augmented_image(augmented, args.image_path, name)

    except Exception as e:
        print("there is an issue :", e)


if __name__ == "__main__":
    main()
