# def analyse_leaf():
import cv2
import numpy as np


def measure_leaf(contour):
    """
     Calculate some of the image measurements such as
     area perimeter and centroid of the object 
     Args:
        contour (np.ndarray): Contour outlining the leaf
    
     Returns :
      measurement : dictionary that contains different measurement of the leaf  

    """
    measurement = {}
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    M = cv2.moments(contour)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    measurement["area"] = area
    measurement["perimeter"] = perimeter
    measurement["center_x"] = cx
    measurement["center_y"] = cy
    return measurement


def resample_path(points, n_points):
    """Resample an ordered sequence of 2D points to n_points equally
    spaced (by arc length) points along the path."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return np.repeat(points, n_points, axis=0)

    deltas = np.diff(points, axis=0)
    seg_lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    cum_length = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_length = cum_length[-1]

    if total_length == 0:
        return np.repeat(points[:1], n_points, axis=0)

    target_lengths = np.linspace(0, total_length, n_points)
    x_interp = np.interp(target_lengths, cum_length, points[:, 0])
    y_interp = np.interp(target_lengths, cum_length, points[:, 1])
    return np.column_stack([x_interp, y_interp])


def compute_landmarks(contour, n_landmarks):
    """
    Compute the image landmarks
    Args:
    contour (np.ndarray): Contour outlining the leaf

    Returns :
    landmarks points: top center and bottom
    """
    pts = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    left_idx = np.argmin(pts[:, 0])
    right_idx = np.argmax(pts[:, 0])
    i1, i2 = sorted((left_idx, right_idx))
    path_a = pts[i1:i2 + 1]
    path_b = np.vstack([pts[i2:], pts[:i1 + 1]])
    if path_a[:, 1].mean() <= path_b[:, 1].mean():
        top_path, bottom_path = path_a, path_b
    else:
        top_path, bottom_path = path_b, path_a
    top = resample_path(top_path, n_landmarks)
    bottom = resample_path(bottom_path, n_landmarks)
    center = (top + bottom) / 2.0
    landmarks = {}
    landmarks["top"] = top
    landmarks["bottom"] = bottom
    landmarks["center"] = center
    return landmarks


def colour_histogram(img):
    """
        Compute the image colors frequencies
        Args:
        image (np.ndarray): original image (in BGR)
    
        Returns :
        histogram data: dict that contain the frequencies and the colors we want to plot
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)
    hue = pixels[:, 0]
    ranges = {
        "red": (0, 10),
        "orange": (11, 25),
        "yellow": (26, 35),
        "green": (36, 85),
        "cyan": (86, 100),
        "blue": (101, 130),
        "purple": (131, 160),
        "pink": (161, 179)
            }
    frequencies = {}

    for color, (low, high) in ranges.items():
        mask = (hue >= low) & (hue <= high)
        frequencies[color] = np.sum(mask)
    bar_colors = [
        "red",
        "orange",
        "yellow",
        "green",
        "cyan",
        "blue",
        "purple",
        "pink"
    ]  
    histogram_data = {}
    histogram_data["frequencies"] = frequencies
    histogram_data["bar_colors"] = bar_colors
    return histogram_data
