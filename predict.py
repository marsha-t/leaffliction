import argparse
from pathlib import Path
import torch
from PIL import Image

from classification.models import create_model
from classification.transforms import create_transform


def parse_arguments():
    """
    Parse image path and required checkpoint path from command line

    Returns:
        argparse.Namespace: Parsed prediction arguments
    """
    parser = argparse.ArgumentParser(
        description='Predict the leaf-disease class of an image'
    )
    parser.add_argument(
        'image',
        help='Path to image file'
    )
    parser.add_argument(
        '--checkpoint',
        required=True,
        help='Path to checkpoint file'
    )
    return parser.parse_args()


def load_checkpoint(checkpoint_path, device):
    """
    Reconstruct the model from checkpoint metadata, load its weights,
        move it to the requested device, and enable evaluation mode
    Also load preprocessing configuration

    Args:
        checkpoint_path (str or Path): Path to a training checkpoint
        device (torch.device): Device on which to load and run the model

    Returns:
        tuple: Restored model, preprocessing transform, and a dictionary
            mapping integer class indices to class names
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = create_model(
        checkpoint['model_type'], len(checkpoint['class_to_index'])
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    transform = create_transform(
        model_type=checkpoint['model_type'],
        mean=checkpoint['preprocessing']['mean'],
        std=checkpoint['preprocessing']['std'],
    )

    index_to_class = {
        index: class_name
        for class_name, index in checkpoint['class_to_index'].items()
    }

    return model, transform, index_to_class


def main():
    """Load a checkpoint and print the predicted class of image"""
    args = parse_arguments()
    image_path = Path(args.image).resolve()
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    model, transform, index_to_class = load_checkpoint(args.checkpoint, device)

    with Image.open(image_path) as image:
        image = image.convert('RGB')
        transformed_image = transform(image)

    input_tensor = transformed_image.unsqueeze(0)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        predicted_index = logits.argmax(dim=1).item()
        print(index_to_class[predicted_index])


if __name__ == "__main__":
    main()
