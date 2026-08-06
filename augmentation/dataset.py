import random
from PIL import Image

from analysis import validate_directory
from augmentation.constants import AUGMENTATIONS
from augmentation.io import save_augmented_image


def scan_dataset(root):
    """
    Validate directory structure, lists original images and
    records augmentations already applied to each original image

    Args:
        root (Path): Root directory of the dataset

    Returns:
        dict: Mapping of class names to dictionaries: 
            {original image paths: sets of augmentation names}
    """
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
            required for balancing.
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


def execute_augmentation_plan(dataset, plan):
    """
    Generate augmented images required to balance each class
    - Original images are processed in shuffled rounds so each image is
        considered before any image is selected again
    - For each image, one previously unused augmentation is selected at random

    Args:
        dataset (dict): Mapping of class names to dictionaries whose keys
            are original image paths and whose values are sets of already
            applied augmentation names
        plan (dict): Mapping of class names to the number of additional
            images to generate

    Raises:
        ValueError: If a class cannot produce enough unique augmentations
            to reach its target
    """
    for class_name, target in plan.items():
        images = dataset[class_name]
        generated = 0

        remaining_capacity = sum(
            len(AUGMENTATIONS) - len(used) for used in images.values()
        )
        if target > remaining_capacity:
            raise ValueError(
                f"{class_name} requires {target} additional images, "
                f"but only {remaining_capacity} unique augmentations remain"
            )

        while generated < target:
            image_paths = list(images)
            random.shuffle(image_paths)

            for image_path in image_paths:
                if generated >= target:
                    break
                used = images[image_path]
                available = [
                    augmentation
                    for augmentation in AUGMENTATIONS
                    if augmentation[0] not in used
                ]
                if not available:
                    continue
                augmentation_name, augmentation_function = random.choice(
                    available
                )
                with Image.open(image_path) as image:
                    augmented = augmentation_function(image)
                    save_augmented_image(
                        augmented,
                        image_path,
                        augmentation_name
                    )
                used.add(augmentation_name)
                generated += 1
        print(f'Augmented {target} images for {class_name}')
