import argparse
from pathlib import Path
import torch
import math
from torch.utils.data import DataLoader
import json
from datetime import datetime

from classification.data_utils import (
    prepare_data, create_class_mapping, compute_training_stats
)
from classification.transforms import create_transform
from classification.dataset import LeafDataset
from classification.models import create_model
from classification.training import train_model


def parse_arguments():
    """
    Parse training arguments and validate numeric options

    Returns:
        argparse.Namespace: Validated training arguments
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
        choices=['custom_cnn'],  # + 'pretrained_cnn', 'pretrained_vit'
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
        '--epochs',
        type=int,
        default=20,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning-rate', '--learning_rate',
        dest='learning_rate',
        type=float,
        default=1e-3,
        help='Learning rate',
    )
    parser.add_argument(
        '--run-name',
        help='Name for this training run; defaults to a timestamp',
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error('--batch-size must be at least 1')

    if args.epochs < 1:
        parser.error('--epochs must be at least 1')

    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('--learning-rate must be a finite number greater than 0')
    if args.run_name is not None:
        if not args.run_name or not all(
            char.isascii() and (char.isalnum() or char in '-_')
            for char in args.run_name
        ):
            parser.error(
                '--run-name must contain only letters, numbers, '
                'hyphens, and underscores'
            )
    return args


def main():
    """Prepare data, train classifier, and save checkpoints and history"""
    args = parse_arguments()

    run_name = args.run_name or datetime.now().strftime(
        '%Y-%m-%d_%H-%M-%S-%f'
    )
    checkpoint_dir = Path('checkpoints') / args.model / run_name

    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise SystemExit(
            f'Run directory already exists: {checkpoint_dir}. '
            'Choose another --run-name.'
        )

    print(f'Run outputs: {checkpoint_dir}')

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False
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

    checkpoint_metadata = {
        'model_type': args.model,
        'class_to_index': class_to_index,
        'preprocessing': {
            'mean': mean.tolist(),
            'std': std.tolist(),
        }
    }

    history = train_model(
        model,
        train_loader,
        validation_loader,
        device,
        args.learning_rate,
        args.epochs,
        checkpoint_dir,
        checkpoint_metadata
    )
    history_path = checkpoint_dir / 'history.json'
    with history_path.open('w', encoding='utf-8') as history_file:
        json.dump(history, history_file, indent=2)


if __name__ == "__main__":
    main()
