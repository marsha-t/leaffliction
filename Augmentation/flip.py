import matplotlib.pyplot as plt


def flip(img_arr):
    """
    mirrored version of an image by reversing the positions of its pixels
    along the horizontal or vertical axis while preserving the image content
    and label.
    as a default one im using the vertical flipping
     Args:
        img_arr : numpy array of the image
    Return:
        flipped array of the image horizontally
    """
    flipped_img = img_arr[:, ::-1, :]
    return (flipped_img)


def vertical_flip(img_arr):
    """
    flip the image horizontaly
     Args:
            img_arr : numpy array of the image
        Return:
            flipped array of the image vertically
    """
    flipped_img = img_arr[::-1, :, :]
    return (flipped_img)


def visualize(img_arr):
    horizontal_flipped_image = flip(img_arr)
    verticaly_flipped_image = vertical_flip(img_arr)

    fig, plots = plt.subplots(nrows=1, ncols=3, figsize=(15, 2))
    plots[0].imshow(img_arr)
    plots[0].set_title("original")
    plots[1].imshow(horizontal_flipped_image)
    plots[1].set_title("horizontal_flipped")
    plots[2].imshow(verticaly_flipped_image)
    plots[2].set_title("vertical_flipped")
    plt.show()
