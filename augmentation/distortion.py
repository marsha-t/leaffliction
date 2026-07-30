from PIL import Image
import numpy as np
import cv2

def distortion(
    image: Image.Image,
    alpha: float = 20,
    sigma: float = 5,
    seed: int | None = None
) -> Image.Image:
    """
    Implement elastic distortion
    - Initial displacement maps created using Uniform(-1, 1)

    Args:
        image (Image.Image): Pillow image
        alpha (float): Scale factor controlling distortion strength
        sigma (float): Standard deviation of Gaussian smoothing 
            applied to displacement field
        seed (int | None): Random seed for reproducible distortions.
            If None, a different distortion is generated each time

    Returns:
        Image.Image: Distorted image.
    """

    image_array = np.array(image)

    rng  = np.random.default_rng(seed)
    dx = rng.uniform(-1, 1, size=image_array.shape[:2])
    dy = rng.uniform(-1, 1, size=image_array.shape[:2])

    dx = cv2.GaussianBlur(dx, (0,0), sigmaX=sigma)
    dy = cv2.GaussianBlur(dy, (0,0), sigmaX=sigma)

    dx *= alpha
    dy *= alpha
    x, y = np.meshgrid(np.arange(image_array.shape[1]), np.arange(image_array.shape[0]))

    new_x = x + dx
    new_y = y + dy

    new_x = new_x.astype(np.float32)
    new_y = new_y.astype(np.float32)

    distorted_image_array = cv2.remap(image_array, new_x, new_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    distorted = Image.fromarray(distorted_image_array)
    return distorted