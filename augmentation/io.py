from pathlib import Path

from augmentation.constants import AUGMENTATIONS


def augmented_paths(
    original_path,
    augmentation_name,
    data_root=Path('data'),
    augmented_root=Path('augmented_directory')
) -> tuple[Path, Path]:
    """
    Return the paths where an augmented image should be stored

    Args:
        original_path (str or Path): Path to the original image
        augmentation_name (str): Name of the augmentation
        data_root (str or Path): Path to root of dataset
        augmented_root (str or Path): Path to augmented directory

    Returns:
        tuple[Path, Path]:
            (path in original dataset, path in augmented_directory)
    """
    original_path = Path(original_path).resolve()
    data_root = Path(data_root).resolve()
    augmented_root = Path(augmented_root).resolve()
    
    filename = (
        f"{original_path.stem}_"
        f"{augmentation_name}"
        f"{original_path.suffix}"
    )

    original_output = original_path.parent / filename
    relative_dir = original_path.parent.relative_to(data_root)
    augmented_output = augmented_root / relative_dir / filename

    return original_output, augmented_output


def save_augmented_image(
    image,
    path,
    name,
    data_root=Path('data'),
    augmented_root=Path('augmented_directory')
):
    """
    Save augmented image in original data folder and in
        augmented_directory folder

    Args:
        image (Image.Image): augmented image
        path (str): filepath of original image
        name (str): augmentation type
        data_root (str or Path): Path to root of dataset
        augmented_root (str or Path): Path to augmented directory

    Raises:
        ValueError if input file does not come from /data
    """
    original_output, augmented_output = augmented_paths(
        path,
        name,
        data_root,
        augmented_root,
    )
    
    # Save in /data
    image.save(original_output)

    # Save in /augmented_directory
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
    stem = path.stem
    return any(
        stem.endswith(f"_{name[0]}") for name in AUGMENTATIONS
    )


def remove_empty_parents(path: Path, stop: Path):
    """
    Remove empty parent directories up to, but not including, stop

    Args:
        path (Path): Deleted file path
        stop (Path): Highest directory that should never be removed
    """
    directory = path.parent

    while directory != stop:
        try:
            directory.rmdir()
        except OSError:
            break  # Directory isn't empty

        directory = directory.parent
