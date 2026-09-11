import argparse
from pathlib import Path
import torch
from PIL import Image

from classification.models import create_model
from classification.transforms import create_transform


def parse_arguments():
    """
    """
    parser = argparse.ArgumentParser(
        description='Train leaf-disease classifier'
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