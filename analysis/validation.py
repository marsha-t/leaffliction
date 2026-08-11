from pathlib import Path

IMAGE_EXTENSIONS = ('.jpg', '.png')


def validate_directory(path: str) -> list[str]:
    """
    Validate that filepath is to a directory containing sub directories of images

    Args:
    path (str): directory path

    Return:
    List(str): list of warnings
        - there are files in directory (instead of sub-directories)
        - there are non-image files in sub-directory

    Raises:
    ValueError if path does not exist or path is not to a directory
    """
    dataset_dir = Path(path)
    warnings = []

    if not dataset_dir.exists():
        raise ValueError(f'{path} not found')
    if not dataset_dir.is_dir():
        raise ValueError(f'{path} is not a directory')

    sub_dir_count = 0
    for sub_dir in dataset_dir.iterdir():
        if not sub_dir.is_dir():
            warnings.append(f"{sub_dir} is not a directory")
            continue
        image_count = 0
        for file in sub_dir.iterdir():
            if not file.is_file():
                warnings.append(f"{file} is not a file")
            elif file.suffix.lower() not in IMAGE_EXTENSIONS:
                warnings.append(f"{file} is not an image file")
            else:
                image_count += 1
        if image_count == 0:
            raise ValueError(f'Sub directory {sub_dir} has no image files')
        sub_dir_count += 1
    if sub_dir_count == 0:
        raise ValueError(f'{path}: No sub-directories with image files found')
    return warnings
