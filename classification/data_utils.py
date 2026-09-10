from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

from augmentation.dataset import (
    scan_dataset,
    calculate_target,
    create_augmentation_plan,
    execute_augmentation_plan
)
from classification.splits import (
    select_originals,
    save_dataset_manifest,
    split_originals,
    load_dataset_manifest,
    create_plan_from_manifest
)
from classification.dataset import LeafDataset


def preview_nested_dict(dataset, header="", rows=1):
    if header:
        print(f"---{header}---")
    for class_name, images in dataset.items():
        if isinstance(images, dict):
            print(class_name, dict(list(images.items())[:rows]))
        elif isinstance(images, list):
            print(class_name, images[:rows])
    print()


def prepare_data(dataset_root):
    """
    """
    manifest_path = (Path('manifests') / dataset_root.name / 'dataset.csv')
    if manifest_path.exists():
        train, _ = load_dataset_manifest(manifest_path, dataset_root)
        augmentation_plan = create_plan_from_manifest(train)
    else:
        dataset = scan_dataset(dataset_root)
        # preview_nested_dict(dataset, 'Dataset')

        train_split, validation_split = split_originals(
            dataset,
            validation_ratio=0.2,
            seed=42
        )
        # preview_nested_dict(train_split, 'Train')
        # preview_nested_dict(validation_split, 'Validation')

        train_base = select_originals(dataset, train_split)
        # preview_nested_dict(train_base, 'Train + Existing Augmented')

        balance_target = calculate_target(train_base)
        print(balance_target)

        augmentation_plan = create_augmentation_plan(
            train_base,
            balance_target,
            seed=42,
        )
        # preview_nested_dict(augmentation_plan, 'Augmentation Plan')

        save_dataset_manifest(
            train_base,
            validation_split,
            augmentation_plan,
            dataset_root,
            manifest_path,
        )

    execute_augmentation_plan(
        augmentation_plan,
        data_root=dataset_root.parent,
        augmented_root=Path('augmented_directory')
    )

    return load_dataset_manifest(manifest_path, dataset_root)


def create_class_mapping(train, validation):
    """
    Create one numeric class mapping shared by both dataset splits

    Args:
        train (dict): Class names mapped to training image paths
        validation (dict): Class names mapped to validation image paths

    Returns:
        dict: Class names mapped to deterministic numeric labels

    Raises:
        ValueError: If training and validation contain different classes
    """
    train_classes = set(train)
    validation_classes = set(validation)

    if train_classes != validation_classes:
        raise ValueError(
            "Training and validation class names do not match"
        )

    return {
        class_name: index
        for index, class_name in enumerate(
            sorted(train_classes)
        )
    }


def compute_training_stats(train, class_to_index):
    """
    Compute mean and standard deviation channels in training images

    Args:
        train (dict): Class names mapped to training image paths
        class_to_index (dict): Class names mapped to numeric labels

    Returns:
        tuple[torch.Tensor, torch.Tensor]: mean and standard deviation tensors,
            each with shape (3,)
    """
    stats_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    stats_dataset = LeafDataset(train, class_to_index, stats_transform)
    stats_loader = DataLoader(stats_dataset, batch_size=64, shuffle=False)

    channel_sum = torch.zeros(3)
    channel_squared_sum = torch.zeros(3)
    pixel_count = 0

    for images, _ in stats_loader:  # images = [Batch, Channel, Height, Width]
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_squared_sum += (images ** 2).sum(dim=(0, 2, 3))
        pixel_count += images.shape[0] * images.shape[2] * images.shape[3]

    mean = channel_sum / pixel_count
    mean_squared = channel_squared_sum / pixel_count

    std = torch.sqrt(mean_squared - mean ** 2)

    return mean, std
