#! /usr/bin/env python3
import argparse
from pathlib import Path

from augmentation.dataset import scan_dataset
from augmentation.io import (
    build_augmentation_output_paths,
    remove_empty_parents
)


def parse_args():
    """
    Parse command-line arguments

    Returns:
        argparse.Namespace: parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Remove dataset augmentations and classification manifest"
    )
    parser.add_argument('directory_path', help='directory path')
    args = parser.parse_args()
    return args


def main():
    """Remove detected dataset augmentations & copies, and manifest"""
    try:
        args = parse_args()

        dataset_root = Path(
            args.directory_path
        ).resolve()

        dataset = scan_dataset(dataset_root)
        data_root = dataset_root.parent
        augmented_root = Path(
            "augmented_directory"
        ).resolve()

        for class_name in dataset:
            removed = 0
            images = dataset[class_name]
            for image, augmentations in images.items():
                if not augmentations:
                    continue
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

                    if augmented_output.exists():
                        augmented_output.unlink()
                        remove_empty_parents(
                            augmented_output,
                            augmented_root
                        )
            print(f"Removed {removed} augmentations for {class_name}")

        manifests_root = Path('manifests').resolve()
        manifest_path = (manifests_root / dataset_root.name / 'dataset.csv')
        if manifest_path.exists():
            manifest_path.unlink()
            remove_empty_parents(manifest_path, manifests_root)
            print(f"Removed manifest: {manifest_path}")

    except Exception as e:
        print("Exception:", e)


if __name__ == "__main__":
    main()
