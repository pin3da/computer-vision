"""Set of utilities used across the application"""

import cv2


def plot_text(img, text, pos):
    """Plots some text inside the image with a fixed font, size and color."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, pos, font,
                1, (0, 255, 255), 2,
                cv2.LINE_AA)


def plot_points(img, points, col=(15, 213, 15)):
    """Draws a set of points in the image. Useful to draw the path of moving
    objects in the scene."""
    for p in points:
        cv2.circle(img, (p[0], p[1]), 3, col, -1)
