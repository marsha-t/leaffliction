import random
import csv
from pathlib import Path


def split_originals(dataset, validation_ratio=0.2, seed=42):
    """
    Split original images into training and validation sets by class

    Args:
        dataset(dict): Mapping of class names to dictionaries of original
            image paths and existing augmentations (if any)
        validation_ratio (float): Proportion of original images assigned to
            validation set
        seed (int): Seed used for reproducible shuffling

    Returns:
        tuple[dict, dict]: Training and validation mappings from class names
            to lists of original image paths

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
        originals = sorted(dataset[class_name].keys())

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


def save_split(train, validation, dataset_root, manifest_path):
    """
    Save training and validation assignments to a CSV file
    Paths are stored relative to dataset_root using POSIX separators

    Args:
        train (dict): Class names mapped to paths for images for training
        validation (dict): Class names mapped to paths for images for
            validation
        dataset_root (str or Path): Root containing class directories
        manifest_path (str or Path): CSV file to create

    Raises:
        FileExistsError: if manifest already exists
        ValueError: if image path is outside dataset root
    """
    dataset_root = Path(dataset_root)
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    splits = (
        ('train', train),
        ('validation', validation),
    )
    rows = []
    for split_name, split_data in splits:
        for class_name in sorted(split_data):
            for image_path in sorted(split_data[class_name]):
                relative_path = Path(image_path).resolve().relative_to(
                    dataset_root.resolve()
                )
                rows.append({
                    'path': relative_path.as_posix(),
                    'class_name': class_name,
                    'split': split_name
                })
    with manifest_path.open(
        "x",
        newline="",
        encoding="utf-8",
    ) as manifest_file:
        fieldnames = ['path', 'class_name', 'split']
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_split(manifest_path, dataset_root):
    """
    Load training and validation assignments from a CSV manifest file

    Args:
        manifest_path (str or Path): CSV manifest to load
        dataset_root (str or Path): Root containing class directories

    Returns:
        tuple[dict, dict]: Training and validation mappings from class names
            to lists of image paths

    Raises:
        ValueError: If manifest has unexpected columns or split names
    """
    manifest_path = Path(manifest_path)
    dataset_root = Path(dataset_root).resolve()

    with manifest_path.open(
        "r",
        newline="",
        encoding='utf-8'
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
            if split_name not in splits:
                raise ValueError(
                    f'Manifest contains unknown split {split_name}'
                )

            class_name = row['class_name']
            if not class_name:
                raise ValueError('Manifest contains an empty class name')

            relative_path = row['path']
            if not relative_path:
                raise ValueError('Manifest contains an empty path')

            image_path = (dataset_root / Path(relative_path)).resolve()
            splits[split_name].setdefault(class_name, []).append(image_path)
        return train, validation


def ensure_split(
    dataset,
    dataset_root,
    manifest_path,
    validation_ratio=0.2,
    seed=42,
):
    """
    Load existing dataset split or create and save a new one

    If the split manifest already exists, the training and validation
    splits are loaded from it. Otherwise, the dataset is split according
    to validation_ratio and seed, saved to the manifest, and returned.

    Args:
        dataset: Dataset containing the original samples to split
        dataset_root: Root directory of the dataset. Used to resolve paths
            stored in the split manifest
        manifest_path: Path to the split manifest file
        validation_ratio: Fraction of the dataset assigned to the validation
            split. Defaults to 0.2.
        seed: Random seed used to produce a reproducible split. Defaults to 42.

    Returns:
        tuple: Containing the training and validation splits
    """
    manifest_path = Path(manifest_path)
    dataset_root = Path(dataset_root).resolve()

    if manifest_path.exists():
        return load_split(manifest_path, dataset_root)

    train, validation = split_originals(
        dataset,
        validation_ratio,
        seed,
    )
    save_split(train, validation, dataset_root, manifest_path)
    return train, validation
