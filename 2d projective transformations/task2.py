import cv2
import numpy as np
import os

from src.homography import (
    apply_homography,
    compute_homography,
    biliniear_interpolation,
)

# Base directory to compute relative paths
DIR = os.path.dirname(__file__)


if __name__ == "__main__":
    # Read original matrix
    img = cv2.imread(os.path.join(DIR, './img/scan.jpg'))
    dy, dx, _ = img.shape
    # Use the size of the image as target
    Xp = np.array([[0, 0], [dx, 0], [dx, dy], [0, dy]])

    X = np.array([[100, 37], [869, 178], [870, 1105], [105, 1257]])
    H = compute_homography(X, Xp)
    print("homography matrix:\n", H)

    img_out = apply_homography(img, H, biliniear_interpolation)

    # Write result images.
    cv2.imwrite(os.path.join(DIR, './img/scan_transformed.jpg'), img_out)
