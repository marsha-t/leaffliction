import random
import csv
from pathlib import Path

from augmentation.io import (
    build_augmentation_output_paths, parse_augmented_path
)


def split_originals(dataset, validation_ratio=0.2, seed=42):
    """
    Split original images into training and validation sets by class

    Args:
        dataset(`dict`): Mapping of class names to dictionaries of original
            image paths and existing augmentations (if any)
        validation_ratio (float): Proportion of original images assigned to
            validation set
        seed (int): Seed used for reproducible shuffling

    Returns:
        tuple[`dict`, `dict`]: Training and validation mappings from class
            names to lists of original image paths e.g.,
            {
                "Apple_rust": [
                    Path("/data/Apple/Apple_rust/image20.JPG"),
                    Path("/data/Apple/Apple_rust/image21.JPG"),
                ],
            }

    Raises:
        ValueError: If validation ratio is not strictly between 0 and 1, or
            if a class contains fewer than 2 images
    """
    if not (0 < validation_ratio < 1):
        raise ValueError('Validation ratio should be strictly between 0 and 1')

    rng = random.Random(seed)
    train = {}
    validation = {}

    for class_name in sorted(dataset):
        originals = sorted(dataset[class_name])

        if len(originals) < 2:
            raise ValueError(
                f'Class {class_name} must contain at least 2 original images'
            )

        rng.shuffle(originals)

        validation_count = round(len(originals) * validation_ratio)
        validation_count = max(1, min(validation_count, len(originals) - 1))

        validation[class_name] = originals[:validation_count]
        train[class_name] = originals[validation_count:]
    return train, validation


def select_originals(dataset, selected):
    """
    Select original-image families from a scanned dataset based on split

    Args:
        dataset (`dict`): Dataset produced by scan_dataset()
        selected (`dict`): Class names mapped to selected original paths

    Returns:
        `dict`: A dataset containing only the selected originals and their
            existing augmentation histories
            {
                "Apple_rust": {
                    Path("/data/Apple/Apple_rust/image1.JPG"): {
                        "Flip",
                        "Rotate",
                    },
                    Path("/data/Apple/Apple_rust/image2.JPG"): set(),
                },
            }
    """
    results = {}
    for class_name, images in selected.items():
        results[class_name] = {}
        for image in images:
            results[class_name][image] = set(dataset[class_name][image])

    return results


def save_dataset_manifest(
    train_data,
    validation,
    augmentation_plan,
    dataset_root,
    manifest_path
):
    """
    Save training and validation dataset to CSV. This manifest file contains
    - training originals,
    - existing training augmentations
    - planned training augmentations, and
    - validation originals.

    Args:
        training_dataset (`dict`): Nested dictionary:
            {class_name: {image path: set(augmentations)}}
            (as produced by select_originals()
        validation (`dict`): Validation class names mapped to lists of original
            image paths (as produced by split_originals())
        augmentation_plan (`dict`): Class names mapped to lists of tuples
            (image, augmentation name) (via create_augmentation_plan())
        dataset_root (str or Path): Root directory containing class directories
        manifest_path (str or Path): CSV file to create

    Raises:
        FileExistsError: If the manifest already exists
        ValueError: If an image path is outside dataset_root
    """
    dataset_root = Path(dataset_root).resolve()
    data_root = dataset_root.parent
    manifest_path = Path(manifest_path)

    rows_by_path = {}

    def add_row(image_path, class_name, split_name):
        """
        """
        relative_path = Path(image_path).resolve().relative_to(dataset_root)
        rows_by_path[relative_path.as_posix()] = {
            'path': relative_path.as_posix(),
            'class_name': class_name,
            'split': split_name
        }

    # Add training data and existing augmentations
    for class_name in sorted(train_data):
        for image_path in sorted(train_data[class_name]):
            add_row(image_path, class_name, 'train')

            existing_augmentations = train_data[class_name][image_path]
            for augmentation_name in sorted(existing_augmentations):
                augmented_path, _ = build_augmentation_output_paths(
                    image_path,
                    augmentation_name,
                    data_root
                )
                add_row(augmented_path, class_name, 'train')

    # Add planned augmentations
    for class_name in sorted(augmentation_plan):
        for image_path, augmentation_name in augmentation_plan[class_name]:
            augmented_path, _ = build_augmentation_output_paths(
                image_path,
                augmentation_name,
                data_root
            )
            add_row(augmented_path, class_name, 'train')

    # Add validation data
    for class_name in sorted(validation):
        for image_path in sorted(validation[class_name]):
            add_row(image_path, class_name, 'validation')

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [rows_by_path[path] for path in rows_by_path]
    with manifest_path.open(
        "x",
        newline="",
        encoding="utf-8",
    ) as manifest_file:
        fieldnames = ['path', 'class_name', 'split']
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_dataset_manifest(manifest_path, dataset_root):
    """
    Load training and validation samples from a dataset manifest

    Args:
        manifest_path (str or Path): Dataset manifest CSV to load
        dataset_root (str or Path): Root containing class directories

    Returns:
        tuple[`dict`, `dict`]: Class names mapped to training and validation
            paths e.g.,
            {
                "Apple_rust": [
                    Path("/data/Apple/Apple_rust/image1.JPG"),
                    Path("/data/Apple/""Apple_rust/image1_Flip.JPG"),
                ],
            },
            {
                "Apple_rust": [
                    Path("/data/Apple/Apple_rust/image20.JPG"),
                ],
            },

    Raises:
        ValueError: If the manifest columns or split names are invalid
    """
    dataset_root = Path(dataset_root).resolve()
    manifest_path = Path(manifest_path)

    with manifest_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as manifest_file:
        reader = csv.DictReader(manifest_file)
        expected_fields = ['path', 'class_name', 'split']
        if reader.fieldnames != expected_fields:
            raise ValueError(
                f'Manifest must contain these columns: {expected_fields}'
            )
        train = {}
        validation = {}
        splits = {'train': train, 'validation': validation}

        for row in reader:
            split_name = row['split']
            class_name = row['class_name']
            relative_path = row['path']

            if split_name not in splits:
                raise ValueError(
                    f"Manifest contains unknown split "
                    f"{split_name!r}"
                )

            if not class_name:
                raise ValueError(
                    "Manifest contains an empty class name"
                )

            if not relative_path:
                raise ValueError(
                    "Manifest contains an empty path"
                )

            image_path = (dataset_root / Path(relative_path)).resolve()

            splits[split_name].setdefault(class_name, []).append(image_path)

    return train, validation


def create_plan_from_manifest(train):
    """
    Reconstruct augmentation execution plan from training samples

    Args:
        train (dict): Class names mapped to all training paths
            (from load_dataset_manifest())

    Returns:
        dict: Class names mapped to planned augmentations e.g.,
            {
                "Apple_rust": [
                    (
                        Path("/data/Apple/Apple_rust/image1.JPG"),
                        "Rotate",
                    ),
                    (
                        Path("/data/Apple/Apple_rust/image2.JPG"),
                        "ElasticDistortion",
                    ),
                ],
            }
    """
    plan = {}

    for class_name in sorted(train):
        plan[class_name] = []

        for image_path in train[class_name]:
            parsed = parse_augmented_path(image_path)

            if parsed is None:
                continue

            source_path, augmentation_name = parsed
            plan[class_name].append((source_path, augmentation_name))

    return plan
