from PIL import Image


def flip(img, flip_type="h"):
    """
    Flip an image horizontally or vertically:
        reverse position of pixels along horizontal/vertical axis while
        preserving image content and label
    Defaults to horizontal

    Args:
        img (Image.Image): Input image.
        flip_type (str): 'h' for horizontal or 'v' for vertical.
            Defaults to horizontal.

    Returns:
        Image.Image: Flipped image.
    """
    if flip_type == "v":
        flipped_img = img.transpose(Image.FLIP_TOP_BOTTOM)
    else:
        flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return (flipped_img)
