import argparse
from pathlib import Path

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


def parse_arguments():
    """
    """
    parser = argparse.ArgumentParser(
        description='Train leaf-disease classifier'
    )
    parser.add_argument(
        'dataset',
        help='Dataset directory containing one subdirectory per class'

    )
    return parser.parse_args()


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
        train, validation = load_dataset_manifest(manifest_path, dataset_root)
        augmentation_plan = create_plan_from_manifest(train)
    else:
        dataset = scan_dataset(dataset_root)
        # preview_nested_dict(dataset, 'Dataset')

        train_split, validation_split = split_originals(
            dataset,
            validation_ratio=0.2,
            seed=42
        )
        preview_nested_dict(train_split, 'Train')
        preview_nested_dict(validation_split, 'Validation')

        train_base = select_originals(dataset, train_split)
        preview_nested_dict(train_base, 'Train + Existing Augmented')

        balance_target = calculate_target(train_base)
        print(balance_target)

        augmentation_plan = create_augmentation_plan(
            train_base,
            balance_target,
            seed=42,
        )
        preview_nested_dict(augmentation_plan, 'Augmentation Plan')

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


def main():
    args = parse_arguments()
    dataset_root = Path(args.dataset).resolve()

    prepared = prepare_data(dataset_root)


if __name__ == "__main__":
    main()
