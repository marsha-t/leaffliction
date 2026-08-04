import cv2
# import numpy as np


def create_leaf_mask(image):
    """
    Create binary mask of leaf

    Args:
        image (np.ndarray): original image (in BGR)

    Returns:
        np.ndarray: binary mask
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # TODO consider adding morphological operations to tidy up mask
    # kernel = np.ones((5, 5), np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def extract_roi(mask):
    """
    Extract leaf ROI from mask

    Args:
        mask (np.ndarray): binary mask of leaf

    Returns:
        tuple[int, int, int, int]: bounding rectangle of ROI
            (x, y, width, height)
    """
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    largest = max(contours, key=cv2.contourArea)

    return cv2.boundingRect(largest)
