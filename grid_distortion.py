import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from PIL import Image


def distortion(img, grid_size, strength):

    img_arr = np.array(img)

    height, width = img_arr.shape[:2]

    # Divide image into grids
    cell_height = height / grid_size
    cell_width = width / grid_size

    # Grid points
    x_points = np.linspace(0, width - 1, grid_size + 1)
    y_points = np.linspace(0, height - 1, grid_size + 1)

    distorted_points = []

    # Move grid points
    for y_index, y in enumerate(y_points):
        row = []

        for x_index, x in enumerate(x_points):

            # keep borders fixed
            if (x_index == 0 or 
                x_index == len(x_points)-1 or
                y_index == 0 or
                y_index == len(y_points)-1):

                new_x = x
                new_y = y

            else:
                dx = random.uniform(-strength, strength)
                dy = random.uniform(-strength, strength)

                new_x = x + dx
                new_y = y + dy

            row.append((new_x, new_y))

        distorted_points.append(row)


    # Create pixel coordinate maps
    map_x, map_y = np.meshgrid(
        np.arange(width),
        np.arange(height)
    )

    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)


    # Calculate displacement for every grid cell
    for row in range(grid_size):
        for col in range(grid_size):

            x0 = int(x_points[col])
            x1 = int(x_points[col+1])

            y0 = int(y_points[row])
            y1 = int(y_points[row+1])


            # original corners
            p00 = (x_points[col], y_points[row])
            p10 = (x_points[col+1], y_points[row])
            p01 = (x_points[col], y_points[row+1])
            p11 = (x_points[col+1], y_points[row+1])


            # distorted corners
            d00 = distorted_points[row][col]
            d10 = distorted_points[row][col+1]
            d01 = distorted_points[row+1][col]
            d11 = distorted_points[row+1][col+1]


            # displacement
            dx00 = d00[0] - p00[0]
            dy00 = d00[1] - p00[1]

            dx10 = d10[0] - p10[0]
            dy10 = d10[1] - p10[1]

            dx01 = d01[0] - p01[0]
            dy01 = d01[1] - p01[1]

            dx11 = d11[0] - p11[0]
            dy11 = d11[1] - p11[1]


            # pixels inside this cell
            for y in range(y0, y1):
                for x in range(x0, x1):

                    # normalized position inside cell
                    if x1 != x0:
                        tx = (x - x0) / (x1 - x0)
                    else:
                        tx = 0

                    if y1 != y0:
                        ty = (y - y0) / (y1 - y0)
                    else:
                        ty = 0


                    # bilinear interpolation of displacement
                    dx = (
                        (1-tx)*(1-ty)*dx00 +
                        tx*(1-ty)*dx10 +
                        (1-tx)*ty*dx01 +
                        tx*ty*dx11
                    )

                    dy = (
                        (1-tx)*(1-ty)*dy00 +
                        tx*(1-ty)*dy10 +
                        (1-tx)*ty*dy01 +
                        tx*ty*dy11
                    )


                    map_x[y,x] = x + dx
                    map_y[y,x] = y + dy


    # Apply distortion
    distorted = cv2.remap(
        img_arr,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis("off")


    plt.subplot(1,2,2)
    plt.imshow(distorted)
    plt.title("Grid Distortion")
    plt.axis("off")


    plt.savefig("grid_distortion_result.png", bbox_inches="tight")

    plt.show()
    return distorted

# ---------------------------
# Testing
# ---------------------------



# Show both images
