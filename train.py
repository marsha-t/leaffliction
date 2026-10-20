import argparse
from pathlib import Path
import getpass
import torch
import math
from torch.utils.data import DataLoader
import json
from datetime import datetime

from classification.data_utils import (
    prepare_data, create_class_mapping
)
from classification.transforms import (
    get_preprocessing_config,
    create_transform
)
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
        choices=[
            'custom_cnn',
            'pretrained_cnn',
            'custom_vit',
            'pretrained_vit'
        ],
        default='pretrained_cnn',
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


def configure_torch_cache():
    """
    Configure PyTorch cache directory based on execution environment:
    - At 42: use user's /goinfre directory (and create cache directory if it
        does not exist)
    - Otherwise, use default PyTorch Hub cache

    Raises:
        OSError: If the cache directory cannot be created.
    """
    goinfre = Path("/goinfre")

    if goinfre.is_dir():
        username = getpass.getuser()
        cache_dir = goinfre / username / "torch_cache"
    else:
        cache_dir = Path.home() / ".cache" / "torch" / "hub"

    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(cache_dir))

    print(f"PyTorch cache: {cache_dir}")


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

    configure_torch_cache()

    train, validation = prepare_data(dataset_root)

    # class_to_index = create_class_mapping(train, validation)

    # preprocessing = get_preprocessing_config(
    #     args.model,
    #     train,
    #     class_to_index
    # )

    # transform = create_transform(
    #     args.model,
    #     preprocessing
    # )

    # train_dataset = LeafDataset(
    #     train, class_to_index, transform
    # )
    # validation_dataset = LeafDataset(
    #     validation, class_to_index, transform
    # )

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True
    # )
    # validation_loader = DataLoader(
    #     validation_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=False
    # )

    # model = create_model(args.model, len(class_to_index))
    # model = model.to(device)
    # print('----Model----')
    # print(model)
    # num_parameters = sum(
    #     p.numel()
    #     for p in model.parameters()
    # )
    # num_trainable = sum(
    #     p.numel() for p in model.parameters() if p.requires_grad
    # )
    # print(f'No. of parameters: {num_parameters}')
    # print(f'No. of trainable parameters: {num_trainable}')

    # checkpoint_metadata = {
    #     'model_type': args.model,
    #     'class_to_index': class_to_index,
    #     'preprocessing': preprocessing
    # }

    # history = train_model(
    #     model,
    #     train_loader,
    #     validation_loader,
    #     device,
    #     args.learning_rate,
    #     args.epochs,
    #     checkpoint_dir,
    #     checkpoint_metadata
    # )
    # history_path = checkpoint_dir / 'history.json'
    # with history_path.open('w', encoding='utf-8') as history_file:
    #     json.dump(history, history_file, indent=2)


if __name__ == "__main__":
    main()
