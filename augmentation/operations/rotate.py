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
    return rotated
