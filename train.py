import argparse
from pathlib import Path
import torch 
from torch.utils.data import DataLoader

from classification.data_utils import prepare_data, create_class_mapping, compute_training_stats
from classification.transforms import create_transform
from classification.dataset import LeafDataset
from classification.models import create_model
from classification.training import train_model


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
    parser.add_argument(
        '--model',
        choices=['custom_cnn', 'pretrained_cnn', 'pretrained_vit'],
        default='custom_cnn',
        help='Model architecture to train'
    )
    parser.add_argument(
            '--batch-size',
            type=int,
            default=32,
            help='Training batch size'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=20,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        help='Learning rate'
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    dataset_root = Path(args.dataset).resolve()

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f'Using device: {device}')

    train, validation = prepare_data(dataset_root)

    class_to_index = create_class_mapping(train, validation)

    mean, std = compute_training_stats(train, class_to_index)
    print(f'Mean: {mean}')
    print(f'Std: {std}')

    train_transform = create_transform(
        model_type=args.model,
        mean=mean,
        std=std
    )

    val_transform = create_transform(
        model_type=args.model,
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
    print(f'Train image 1 mode: {train_image.mode}')
    print(f'Train image 1 label: {train_label}')
    validation_image, validation_label = validation_dataset[0]
    print(f'Validation image 1 mode: {validation_image.mode}')
    print(f'Validation image 1 label: {validation_label}')

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=32,
        shuffle=True
    )

    model = create_model(args.model, len(class_to_index))
    model = model.to(device)
    print('----Model----')
    print(model)
    num_parameters = sum(
        p.numel()
        for p in model.parameters()
    )
    num_trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f'No. of parameters: {num_parameters}')
    print(f'No. of trainable parameters: {num_trainable}')

    checkpoint_dir = Path('checkpoints') / args.model
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = train_model(
        model,
        train_loader,
        validation_loader,
        device,
        args.learning_rate,
        args.epoch,
        checkpoint_dir,
    )


if __name__ == "__main__":
    main()
