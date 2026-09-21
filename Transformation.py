#! /usr/bin/env python3
import argparse
import cv2

from transformation.segmentation import (
    create_leaf_mask,
    largest_contour,
    extract_roi
)
from transformation.analysis import (
    measure_leaf,
    compute_landmarks,
    colour_histogram
)
from transformation.visualisation import display_pipeline


def parse_args():
    """
    Parse command-line arguments

    Returns:
        argparse.Namespace: parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description='Analyse leaf shape and colour and display transformations'
    )
    parser.add_argument('image_path', help='image path')
    args = parser.parse_args()
    return args


def transformation_pipeline(image):
    """
    Orchestrates transformations

    Args:
        image (np.Array): image as an array

    Returns:
        dict: {transformation name: computation results}
    """
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    mask = create_leaf_mask(blurred)
    contour = largest_contour(mask)
    roi = extract_roi(contour)
    measurements = measure_leaf(contour)
    landmarks = compute_landmarks(contour, 50)
    histogram = colour_histogram(image)

    pipeline = {
        "blur": blurred,
        "mask": mask,
        "roi": roi,
        "contour": contour,
        "measurements": measurements,
        "landmarks": landmarks,
        "histogram": histogram
    }
    return pipeline


def main():
    """Load specified image and display its analysis"""
    try:
        args = parse_args()

        image = cv2.imread(args.image_path)  # OpenCV manages file internally
        if image is None:
            raise FileNotFoundError(f"Could not read image: {args.image_path}")
        pipeline = transformation_pipeline(image)
        display_pipeline(image, pipeline)

    except Exception as e:
        print("Exception:", e)


if __name__ == "__main__":
    main()
