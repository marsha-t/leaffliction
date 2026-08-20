import random
from PIL import Image
from pathlib import Path

from analysis import validate_directory
from augmentation.constants import AUGMENTATIONS
from augmentation.io import save_augmented_image, augmented_paths


def scan_dataset(root):
    """
    Validate directory structure, lists original images and
    records augmentations already applied to each original image
    Original images are identified using absolute paths

    Args:
        root (Path): Root directory of the dataset

    Returns:
        dict: Nested dictionary: e.g.,
            {
                "Apple_rust": {
                    Path("/data/Apple/Apple_rust/image1.JPG"): {
                        "Flip",
                        "Rotate",
                    },
                    Path("/data/Apple/Apple_rust/image2.JPG"): set(),
                },
                "Apple_scab": {
                    Path("/data/Apple/Apple_scab/image3.JPG"): {
                        "Crop",
                    },
            },
        }
    """
    root = Path(root).resolve()
    validate_directory(root)
    dataset = {}
    augmentation_names = {name for name, _ in AUGMENTATIONS}

    for sub_dir in root.iterdir():
        if not sub_dir.is_dir():
            continue

        images = {}
        for file in sub_dir.iterdir():
            if not file.is_file():
                continue
            parts = file.stem.rsplit('_', maxsplit=1)
            if len(parts) == 2 and parts[1] in augmentation_names:
                original_stem, augmentation_name = parts
                original_path = file.parent / f"{original_stem}{file.suffix}"
                images.setdefault(original_path, set()).add(augmentation_name)
            else:
                images.setdefault(file, set())
        dataset[sub_dir.name] = images

    return dataset


def calculate_target(dataset):
    """
    Calculate number of augmented images required for each class

    Args:
        dataset (dict): Dataset information produced by scan_dataset()

    Returns:
        dict: Mapping of class names to the number of additional images
            required for balancing
            {
                "Apple_rust": 200,
                "Apple_healthy": 0,
            }

    """
    totals = {}
    for class_name, images in dataset.items():
        original_count = len(images)
        augmented_count = sum(len(augmented) for augmented in images.values())
        totals[class_name] = original_count + augmented_count
    max_value = max(totals.values())

    target = {}
    for class_name, count in totals.items():
        target[class_name] = max_value - count
    return target


def create_augmentation_plan(dataset, targets, seed=42):
    """
    Select augmentation-images needed to meet balancing targets
    This function does not create image files.
    - Original images are processed in shuffled rounds so each image is
        considered before any image is selected again
    - For each image, one previously unused augmentation is selected at random

    Args:
        dataset (dict): Dataset information produced by scan_dataset()
        targets (dict): Number of new augmentations required
            for each class produced by calculate_target()
        seed (int): Seed controlling random shuffle

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

    Raises:
        ValueError: If a class cannot produce enough unique augmentations
            to meet its target.
    """
    rng = random.Random(seed)
    plan = {}

    for class_name in sorted(targets):
        plan[class_name] = []
        target = targets[class_name]
        images = dataset[class_name]
        used_augmentations = {
            image_path: set(used) for image_path, used in images.items()
        }

        remaining_capacity = sum(
            len(AUGMENTATIONS) - len(used) for used in images.values()
        )
        if target > remaining_capacity:
            raise ValueError(
                f"{class_name} requires {target} additional images, "
                f"but only {remaining_capacity} unique augmentations remain"
            )
        planned = 0
        while planned < target:
            image_paths = sorted(used_augmentations)
            rng.shuffle(image_paths)

            for image_path in image_paths:
                if planned >= target:
                    break

                used = used_augmentations[image_path]

                available_names = [
                    augmentation_name
                    for augmentation_name, _ in AUGMENTATIONS
                    if augmentation_name not in used
                ]

                if not available_names:
                    continue

                augmentation_name = rng.choice(
                    available_names
                )
                plan[class_name].append(
                    (image_path, augmentation_name)
                )
                used.add(augmentation_name)
                planned += 1
    return plan


def execute_augmentation_plan(
    plan,
    data_root=Path("data"),
    augmented_root=Path("augmented_directory"),
):
    """
    Generate augmented images specified by plan
    Existing augmented images are skipped

    Args:
        plan (dict): Class names mapped to planned augmentations as
            produced by create_augmentation_plan()
    """
    augmentation_functions = dict(AUGMENTATIONS)

    for class_name in sorted(plan):
        generated = 0
        skipped = 0

        for original_image, augmentation_name in plan[class_name]:
            original_output, augmented_output = augmented_paths(
                original_image,
                augmentation_name,
                data_root=data_root,
                augmented_root=augmented_root,
            )

            if original_output.exists() and augmented_output.exists():
                skipped += 1
                continue

            augmentation_function = augmentation_functions[augmentation_name]

            with Image.open(original_image) as image:
                augmented = augmentation_function(image)
                save_augmented_image(
                    augmented,
                    original_image,
                    augmentation_name,
                    data_root=data_root,
                    augmented_root=augmented_root,
                )
                generated += 1
        print(
            f"{class_name}: generated {generated}, "
            f"skipped {skipped}"
        )
