from PIL import Image
import matplotlib.pyplot as plt


def rotate(img, angle=90):
    """
    rotate an image around certain angle around its center
    Args:
            img : pillow image
            angle : the angle you want to rotate the image by
    Return:
        rotated pillow image
    """
    rotated = img.rotate(angle)
    fig, plots = plt.subplots(nrows=1, ncols=2, figsize=(15, 2))
    plots[0].imshow(img)
    plots[0].set_title("original")
    plots[1].imshow(rotated)
    plots[1].set_title("rotate")
    plt.show()
