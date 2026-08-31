import argparse
from pathlib import Path
import torch 
from torch.utils.data import DataLoader 
from classification.data_utils import prepare_data, create_class_mapping, compute_training_stats
from classification.transforms import create_transform
from classification.dataset import LeafDataset
from classification.models import CustomCNN


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


def main():
    args = parse_arguments()
    dataset_root = Path(args.dataset).resolve()

    train, validation = prepare_data(dataset_root)

    class_to_index = create_class_mapping(train, validation)

    mean, std = compute_training_stats(train, class_to_index)
    print(mean)
    print(std)
    train_transform = create_transform(
        model_type='custom_cnn',
        mean=mean,
        std=std
    )

    val_transform = create_transform(
        model_type='custom_cnn',
        mean=mean,
        std=std
    )
    train_dataset = LeafDataset(
        train, class_to_index, train_transform
    )
    validation_dataset = LeafDataset(
        validation, class_to_index, val_transform
    )

    train_image, train_label = train_dataset[0]
    print(train_image.mode)
    print(train_label)
    validation_image, validation_label = validation_dataset[0]
    print(validation_image.mode)
    print(validation_label)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    model = CustomCNN(num_classes=4)
    print(model)
    x = torch.randn(8, 3, 224, 224)
    print(x.shape)
    output = model(x)
    print(output.shape)
    num_parameters = sum(
        p.numel()
        for p in model.parameters()
    )
    num_trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f'No. of parameters: {num_parameters}')
    print(f'No. of trainable parameters: {num_trainable}')

if __name__ == "__main__":
    main()
