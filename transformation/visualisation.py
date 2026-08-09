import cv2
import numpy as np


def draw_mask(image, mask):
    """
    Display the segmented region on a white background

    Args:
        image (np.ndarray): original image
        mask (np.ndarray): binary mask of detected leaf

    Returns:
        np.ndarray: Image where masked region retains its original
            appearance and background is white
    """
    result = np.full_like(image, 255)
    foreground = cv2.bitwise_and(image, image, mask=mask)
    result[mask > 0] = foreground[mask > 0]
    return result


def draw_roi(image, mask, roi):
    """
    Draw region of interest and binary mask on image

    Args:
        image (np.ndarray): original image
        mask (np.ndarray): binary mask of detected leaf
        roi (tuple[int, int, int, int]): bounding rectangle of ROI
            (x, y, width, height)

    Returns:
        np.ndarray: Annotated image showing ROI and binary mask
    """
    annotated = image.copy()

    overlay = image.copy()
    overlay[mask > 0] = (0, 255, 0)
    alpha = 0.4
    annotated = cv2.addWeighted(overlay, alpha, image, 1-alpha, 0)

    x, y, w, h = roi
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return annotated


def draw_analysis(image, contour, measurements):
    """
    Draw analysis results on image

    Args:
        image (np.ndarray): original image
        contour (np.ndarray): contours of detected leaf
        measurements (dict): measurements of leaf

    Returns:
        np.ndarray: Annotated image showing analysis results
    """
    analyzed = image.copy()
    x, y, w, h = cv2.boundingRect(contour)
    cv2.drawContours(
        analyzed,
        [contour],
        -1,
        (255, 0, 255),
        2
    )
    cv2.line(
        analyzed,
        (x, y + h // 2),
        (x + w, y + h // 2),
        (255, 0, 255),
        2
    )
    cv2.line(
        analyzed,
        (x + w // 2, y),
        (x + w // 2, y + h),
        (255, 0, 255),
        2
    )
    cx = int(measurements["center_x"])
    cy = int(measurements["center_y"])

    cv2.circle(
        analyzed,
        (cx, cy),
        5,
        (255, 0, 255),
        3
    )

    return analyzed



def draw_landmarks(image, landmarks):
    """
    Draw pseudolandmarks on image

    Args:
        image (np.ndarray): original image
        landmarks: computed pseudolandmarks describing leaf shape  # TODO landmarks type to be determined
    
    Returns:
        np.ndarray: Annotate image containing landmarks
    """
    img = image.copy()

    color_top = (255, 0, 0)
    color_bottom = (255, 0, 255)
    color_center = (0, 165, 255) 

    radius = 4

    for pt in landmarks["top"]:
        cv2.circle(img, (int(pt[0]), int(pt[1])), radius, color_top, -1)

    for pt in landmarks["bottom"]:
        cv2.circle(img, (int(pt[0]), int(pt[1])), radius, color_bottom, -1)

    for pt in landmarks["center"]:
        cv2.circle(img, (int(pt[0]), int(pt[1])), radius, color_center, -1)
    return img


def plot_histogram(img):
    """
    Visualise colour histogram

    Args:
        histogram (dict): computed histogram data

    Returns:
        None

    """
   
    width = 700
    height = 500
    hist_img = np.ones(
        (height, width, 3),
        dtype=np.uint8
    ) * 255
    histograms = []

    for channel in range(3):
        hist = cv2.calcHist(
            [img],
            [channel],
            None,
            [256],
            [0, 256]
        )

        hist = cv2.normalize(
            hist,
            None,
            0,
            height - 80,
            cv2.NORM_MINMAX
        )

        histograms.append(hist.flatten())
    left = 60
    bottom = height - 50
    top = 30
    right = width - 20
    for y in range(top, bottom + 1, 50):
        cv2.line(
            hist_img,
            (left, y),
            (right, y),
            (220, 220, 220),
            1
        )
    for x in range(left, right + 1, 50):
        cv2.line(
            hist_img,
            (x, top),
            (x, bottom),
            (220, 220, 220),
            1
        )
    cv2.line(
        hist_img,
        (left, top),
        (left, bottom),
        (0, 0, 0),
        2
    )

    cv2.line(
        hist_img,
        (left, bottom),
        (right, bottom),
        (0, 0, 0),
        2
    )
    colors = [
        (255, 0, 0),   
        (0, 255, 0),   
        (0, 0, 255)    
    ]

    for hist, color in zip(histograms, colors):

        for i in range(1, 256):

            x1 = int(
                left + (i - 1) * (right - left) / 256
            )

            x2 = int(
                left + i * (right - left) / 256
            )

            y1 = bottom - int(hist[i - 1])
            y2 = bottom - int(hist[i])

            cv2.line(
                hist_img,
                (x1, y1),
                (x2, y2),
                color,
                2
            )
    for value in range(0, 256, 50):

        x = int(
            left + value * (right - left) / 256
        )

        cv2.putText(
            hist_img,
            str(value),
            (x - 10, bottom + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )
    cv2.putText(
        hist_img,
        "Pixel Value",
        (300, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1
    )

    cv2.putText(
        hist_img,
        "Frequency",
        (5, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1
    )
    cv2.putText(
        hist_img,
        "Color Histogram",
        (250, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2
    )
    return hist_img


def display_pipeline(original_image, pipeline):
    """
    Display output of transformation pipeline

    Args:
        pipeline (dict): Intermediate results and computed features
            generated by transformation pipeline

    Returns:
        None
    """
    images = {}
    images['original'] = original_image
    images['blur'] = pipeline['blur']
    images['mask'] = draw_mask(original_image, pipeline['mask'])
    images['roi'] = draw_roi(
      original_image, pipeline['mask'], pipeline['roi']
    )
    images['analysis'] = draw_analysis(
      original_image, pipeline['contour'], pipeline['measurements']
    )
    images['landmarks'] = draw_landmarks(
      original_image, pipeline['landmarks']
    )
    images['color_histogram'] = plot_histogram(original_image)
    # TODO this currently opens each image in a separate window. Fix to show everything in one window
    for name, image in images.items():
        cv2.imshow(name, image)
        cv2.waitKey(0)  # wait indefinitely until key press
        cv2.destroyAllWindows()  # close all windows

    # plot_histogram(pipeline['histogram'])
