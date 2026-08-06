from PIL import Image


def shear(
    image: Image.Image,
    shear_factor: float,
    horizontal: bool
) -> Image.Image:
    """
    Apply horizontal or vertical shear transformation to an image

    Args:
        image (Image.Image): Pillow image
        shear_factor (float): Amount of shear to apply, between -1 and 1
        horizontal (bool): True for horizontal shear, False for vertical shear

    Returns:
        Image.Image: new sheared object

    Raises:
        ValueError:
            if shear_factor is outside -1 and 1 (too aggressive)
    """
    if not -1 <= shear_factor <= 1:
        raise ValueError('Shear factor should be between -1 and 1')
    if horizontal:
        return image.transform(
            image.size,
            Image.Transform.AFFINE,
            data=(1, shear_factor, 0, 0, 1, 0)
        )
    else:
        return image.transform(
            image.size,
            Image.Transform.AFFINE,
            data=(1, 0, 0, shear_factor, 1, 0)
        )
