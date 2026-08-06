#! /usr/bin/env python3
import argparse
from pathlib import Path

from augmentation.dataset import scan_dataset
from augmentation.io import augmented_paths, remove_empty_parents

def parse_args():
    """
    Parse command-line arguments

    Returns:
        argparse.Namespace: parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description='Augmentation')
    parser.add_argument('directory_path', help='directory path')
    args = parser.parse_args()
    return args


def main():
    try:
        args = parse_args()

        path = Path(args.directory_path)
        dataset = scan_dataset(path)
        for class_name in dataset:
            removed = 0
            images = dataset[class_name]
            for image, augmentations in images.items():
                if not augmentations:
                    continue
                for augmentation_name in augmentations:
                    original_output, augmented_output = augmented_paths(
                        image,
                        augmentation_name
                    )
                    if original_output.exists():
                        original_output.unlink()
                        removed += 1

                    if augmented_output.exists():
                        augmented_output.unlink()
                        remove_empty_parents(
                            augmented_output,
                            Path("augmented_directory")
                        )
            print(f"Removed {removed} augmentations for {class_name}")
    except Exception as e:
        print("there is an issue :", e)


if __name__ == "__main__":
    main()
