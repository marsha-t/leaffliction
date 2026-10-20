#! /usr/bin/env python3
import argparse
from pathlib import Path
import shutil

from augmentation.dataset import scan_dataset
from augmentation.io import (
    build_augmentation_output_paths,
)


def parse_args():
    """
    Parse command-line arguments

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description=(
            "Remove dataset augmentations, classification manifest and "
            "generated symbolic links"
        )
    )
    parser.add_argument(
        '--evaluation',
        action='store_true',
        help='Prepare for evaluation: keeps manifests and augmented_directory'
    )
    parser.add_argument('directory_path', help='Dataset directory path')
    return parser.parse_args()


def remove_empty_directories(directory: Path, stop: Path):
    """
    Remove empty directories from directory upwards, including stop.

    Args:
        directory (Path): First directory to attempt to remove
        stop (Path): Highest directory that may be removed
    """
    directory = Path(directory)
    stop = Path(stop)

    while directory == stop or stop in directory.parents:
        try:
            directory.rmdir()
        except OSError:
            break

        if directory == stop:
            break

        directory = directory.parent


def remove_augmented_files(dataset_root, remove_augmented_directory=True):
    """
    Remove detected augmented images from source directory.
    Optionally remove balanced dataset in augmented_directory

    Args:
        dataset_root (Path): Root directory of source dataset
        remove_augmented_directory (bool): Whether to remove augmented_directory
    """
    dataset = scan_dataset(dataset_root)
    data_root = dataset_root.parent
    augmented_root = Path("augmented_directory").resolve()

    for class_name, images in dataset.items():
        removed = 0

        for image, augmentations in images.items():
            for augmentation_name in augmentations:
                original_output, augmented_output = (
                    build_augmentation_output_paths(
                        image,
                        augmentation_name,
                        data_root,
                        augmented_root,
                    )
                )
                if original_output.exists():
                    original_output.unlink()
                    removed += 1

        print(f"Removed {removed} augmentations for {class_name}")

    if remove_augmented_directory:
        augmented_dataset = augmented_root / dataset_root.name

        if augmented_dataset.is_dir():
            shutil.rmtree(augmented_dataset)
            print(f"Removed augmented dataset: {augmented_dataset}")
        if augmented_root.exists():
            remove_empty_directories(augmented_root, augmented_root)


def remove_manifest(dataset_root):
    """
    Remove classification manifest and empty manifest directories

    Args:
        dataset_root (Path): Root directory of source dataset
    """
    manifests_root = Path('manifests').resolve()
    manifest_path = (manifests_root / dataset_root.name / 'dataset.csv')

    if manifest_path.exists():
        manifest_path.unlink()
        remove_empty_directories(manifest_path.parent, manifests_root)
        print(f"Removed manifest: {manifest_path}")


def remove_symlinks(dataset_root):
    """
    Remove generated training and validation symlinks

    Args:
        dataset_root (Path): Root directory of source dataset
    """
    views_root = Path('dataset_views').resolve()
    dataset_views = views_root / dataset_root.name
    if not dataset_views.is_dir() or dataset_views.is_symlink():
        return

    removed = 0

    for class_dir in dataset_views.iterdir():
        if not class_dir.is_dir() or class_dir.is_symlink():
            continue

        for split_name in ('train', 'validation'):
            split_dir = class_dir / split_name

            if not split_dir.is_dir() or split_dir.is_symlink():
                continue

            for image_path in split_dir.iterdir():
                if image_path.is_symlink():
                    image_path.unlink()
                    removed += 1

            remove_empty_directories(split_dir, views_root)
    print(f"Removed {removed} dataset-view symbolic links")


def remove_transformation_outputs():
    """
    Remove generated transformation outputs
    """
    output_root = Path("transformation_outputs").resolve()

    if output_root.is_dir() and not output_root.is_symlink():
        shutil.rmtree(output_root)
        print(f"Removed transformation outputs: {output_root}")


def main():
    """
    Remove detected dataset augmentations & copies, symlinks and manifest
    """
    try:
        args = parse_args()

        dataset_root = Path(args.directory_path).resolve()

        remove_symlinks(dataset_root)
        remove_augmented_files(
            dataset_root, remove_augmented_directory=not args.evaluation
        )
        remove_transformation_outputs()
        if not args.evaluation:
            remove_manifest(dataset_root)
    except Exception as e:
        print("Exception:", e)


if __name__ == "__main__":
    main()
