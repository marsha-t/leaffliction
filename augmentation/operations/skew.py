from PIL import Image


def skew(img):
    """
    Skewing an image is a geometric transformation where you shift pixels
    along one direction while keeping the other direction unchanged
    (1,0.5, 0,0,1,0))
    these numbers means that the x axis will move to the right and theis will appear more in the bottom pixels as y increas downside
    x2 = ax+yb+c
    y2= dx+ey+d
    (a,b,c,d,e,f) are the parameters passed to AFFINE
        Args:
            img : pillow image
        Return:
            skewed pillow image
    """
    skewed_img = img.transform(img.size, Image.AFFINE, (1, 0, 0, 0.5, 1, 0))
    return skewed_img
