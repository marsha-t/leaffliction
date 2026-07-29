from PIL import Image


CropBox = tuple[int, int, int, int]


def crop(image: Image.Image, box: CropBox) -> Image.Image:
    """
    Crop image using specified crop box
    - Coordinates are (left, upper, right, lower)
    - (0, 0) is top left corner of image

    Args:
        image (Image.Image): Pillow image
        box (CropBox): (left, upper, right, lower)

    Returns:
        Image.Image: new Image object

    Raises:
        ValueError:
            If the crop box is invalid, has non-positive width
            or height, or extends outside the image
    """
    # Implicitly checks that there are 4 elements and returns ValueError
    left, upper, right, lower = box

    if left >= right:
        raise ValueError(
            'Right coordinate must be greater than left coordinate'
        )
    if upper >= lower:
        raise ValueError(
            'Upper coordinate must be less than lower coordinate'
        )
    if not (
        0 <= left
        and 0 <= upper
        and right <= image.width
        and lower <= image.height
    ):
        raise ValueError(
            f'Crop box {box} must lie within image bounds. '
            f'Image size: ({image.width}, {image.height})'
        )
    return image.crop(box)
