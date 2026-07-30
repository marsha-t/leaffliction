#! /usr/bin/env python3
from PIL import Image
import argparse
import numpy as np

from augmentation import crop, shear, distortion


def load_img(path):
    """
        load the image

        Args:
            path : image path
        Return:
            array of the image
"""
    img = Image.open(path)
    image_array = np.array(img)
    return (image_array)


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


def main():
    try:
        args = parse_args()
        # img_array = load_img(args.image_path)
        # print(img_array.shape)

        img = Image.open(args.image_path)
        # augmented = crop(img, (0, 0, 100, 100))
        # augmented = shear(img, 0, horizontal=True)
        # augmented = shear(img, 0.3, horizontal=True)
        augmented = distortion(img)
        augmented.show()

    except Exception as e:
        print("there is an issue :", e)


if __name__ == "__main__":
    main()
