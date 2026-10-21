#! /usr/bin/env python3
import argparse
import cv2
from pathlib import Path

from analysis import IMAGE_EXTENSIONS
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
from transformation.visualisation import (
    render_pipeline, display_pipeline, save_pipeline, render_specified_pipeline
)


PROTECTED_DIRECTORIES = [
    Path('data'),
    Path('augmented_directory')
]


def parse_args():
    """
    Parse command-line arguments

    Returns:
        argparse.Namespace: parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description='Analyse leaf shape and colour and display transformations'
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--mask", action="store_const",
                       dest="transformation_type",const="mask")
    group.add_argument("--landmarks", action="store_const" ,
                       dest="transformation_type",const="landmarks")
    group.add_argument("--blur", action="store_const" ,
                       dest="transformation_type", const="blur")
    group.add_argument("--roi", action="store_const" ,
                       dest="transformation_type",const="roi")
    group.add_argument("--contour", action="store_const" ,
                       dest="transformation_type",const="contour")
    group.add_argument("--histogram", action="store_const" ,
                       dest="transformation_type",const="histogram")

    parser.add_argument('path', help='image or directory path')
    parser.add_argument(
        "--dst", "-dst",
        help="Destination directory for transformed images"
    )

    args = parser.parse_args()
    return args


def paths_overlap(path_a, path_b):
    """
    Check whether two paths are identical or one contains the other
    """
    a = path_a.resolve()
    b = path_b.resolve()

    return a == b or a in b.parents or b in a.parents


def validate_src_dest(src, dst):
    """
    Validate source and destination directories

    Args:
        src (Path): Source directory containing images
        dst (Path): Destination directory for transformed images

    Returns:
        Path: Resolved destination directory path.

    Raises:
        NotADirectoryError: If the source is not a directory or the
            destination exists but is not a directory.
        ValueError: If the destination overlaps with the source or
            protected directories, or is not empty.
    """
    src = src.resolve()
    dst = dst.resolve()

    if not src.is_dir():
        raise NotADirectoryError(
            f"Source directory does not exist: {src}"
        )

    if paths_overlap(src, dst):
        raise ValueError(
            "Destination must not overlap with source directory"
        )

    for protected in PROTECTED_DIRECTORIES:
        if paths_overlap(dst, protected):
            raise ValueError(
                f"Destination overlaps protected directory: "
                f"{protected.resolve()}"
            )

    if dst.exists():
        if not dst.is_dir():
            raise NotADirectoryError(
                f"Destination is not a directory: {dst}"
            )

    return dst


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


def process_directory(src, dst, transform_type=None):
    """
    Process images in a source directory and save transformations.
    """
    dst = validate_src_dest(src, dst)
    image_paths = sorted(
        path for path in src.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise ValueError(f"No images found in {src}")

    dst.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        pipeline = transformation_pipeline(image)
        if transform_type == None:
            images = render_pipeline(image, pipeline)
        else:
            images = render_specified_pipeline(image, pipeline, transform_type)

        save_pipeline(images, image_path, dst)

        print(f"Processed: {image_path.name}")


def main():
    """Display or save leaf transformations"""
    try:
        args = parse_args()

        path = Path(args.path)
        transform_type = args.transformation_type  
            
        if path.is_file():
            image = cv2.imread(args.path)  # OpenCV manages file internally
            if image is None:
                raise FileNotFoundError(f"Could not read image: {args.path}")
            pipeline = transformation_pipeline(image)
            images = render_pipeline(image, pipeline)
            display_pipeline(images)
        elif path.is_dir():
            process_directory(Path(path), Path(args.dst), transform_type)
        else:
            raise FileNotFoundError(f'Invalid path: {path}')
    except Exception as e:
        print("Exception:", e)


if __name__ == "__main__":
    main()
