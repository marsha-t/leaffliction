from PIL import Image


def distortion(
    image: Image.Image,
    alpha: float = 20,
    sigma: float = 5,
    seed: int | None = None
) -> Image.Image:
    """
    Implement elastic distortion

    Args:
        image (Image.Image): Pillow image
        alpha (float): Maximum displacement strength in pixels
        sigma (float): Standard deviation of Gaussian smoothing 
            applied to displacement field
        seed (int | None): Random seed for reproducible distortions.
            If None, a different distortion is generated each time

    Returns:
        Image.Image: Distorted image.
    """
