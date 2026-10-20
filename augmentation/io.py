from pathlib import Path
import shutil

from analysis import IMAGE_EXTENSIONS
from augmentation.constants import AUGMENTATIONS


def build_augmentation_output_paths(
    original_path,
    augmentation_name,
    data_root=Path('data'),
    augmented_root=None
) -> tuple[Path, Path]:
    """
    Build the two output paths for an augmented image

    Args:
        original_path (str or Path): Path to the original image
        augmentation_name (str): Name of the augmentation
        data_root (str or Path): Path to root of dataset
        augmented_root (str or Path or None): Path to augmented directory

    Returns:
        tuple[Path, Path | None]:
            (path in original dataset, path in augmented_directory)
    """
    original_path = Path(original_path).resolve()

    filename = (
        f"{original_path.stem}_"
        f"{augmentation_name}"
        f"{original_path.suffix}"
    )

    original_output = original_path.parent / filename
    augmented_output = None
    if augmented_root is not None:
        data_root = Path(data_root).resolve()
        augmented_root = Path(augmented_root).resolve()
        relative_dir = original_path.parent.relative_to(data_root)
        augmented_output = augmented_root / relative_dir / filename

    return original_output, augmented_output


def save_augmented_image(
    image,
    path,
    name,
    data_root=Path('data'),
    augmented_root=None
):
    """
    Save augmented image in original data folder and
        optionally in augmented_directory folder

    Args:
        image (Image.Image): augmented image
        path (str): filepath of original image
        name (str): augmentation type
        data_root (str or Path): Path to root of dataset
        augmented_root (str or Path or None): Path to augmented directory

    Raises:
        ValueError if input file does not come from /data
    """
    original_output, augmented_output = build_augmentation_output_paths(
        path,
        name,
        data_root,
        augmented_root,
    )

    # Save in /data
    image.save(original_output)

    # Save in /augmented_directory
    if augmented_output is not None:
        augmented_output.parent.mkdir(
            parents=True,
            exist_ok=True
        )
        image.save(augmented_output)


def is_augmented_image(path):
    """
    Determine whether image is an augmented image
    - Augmented images are identified by suffix matching one
    of the supported augmentation names (e.g. '_Flip', '_Rotate')

    Args:
        path (Path): Path to an image

    Returns:
        bool: True if the image appears to be an augmented image,
            False otherwise
    """
    return parse_augmented_path(path) is not None


def parse_augmented_path(path):
    """
    Extract source path and augmentation name from augmented path

    Args:
        path (str or Path): Original or augmented image path

    Returns:
        tuple[Path, str] | None: Source path and augmentation name,
            or None when the path is not augmented
    """
    path = Path(path)

    augmentation_names = [name for name, _ in AUGMENTATIONS]

    for augmentation_name in augmentation_names:
        marker = f"_{augmentation_name}"

        if path.stem.endswith(marker):
            original_stem = path.stem[:-len(marker)]

            source_path = path.with_name(f"{original_stem}{path.suffix}")

            return source_path, augmentation_name

    return None


def copy_dataset(source_root, destination_root):
    """
    Copy existing dataset images into destination directory,
    preserving class subdirectories

    Args:
        source_root (Path): Original dataset path
        destination_root (Path): Output dataset path
    """
    source_root = Path(source_root).resolve()
    destination_root = Path(destination_root).resolve()

    if source_root == destination_root:
        raise ValueError("Source and destination must differ")
    if destination_root.is_relative_to(source_root):
        raise ValueError("Destination cannot be inside source")

    for class_dir in source_root.iterdir():
        if not class_dir.is_dir():
            continue
        destination_class = (
            destination_root / source_root.name / class_dir.name
        )
        destination_class.mkdir(parents=True, exist_ok=True)
        for image_path in class_dir.iterdir():
            if (
                image_path.is_file()
                and image_path.suffix.lower() in IMAGE_EXTENSIONS
            ):
                shutil.copy2(image_path, destination_class/image_path.name)
