import cv2
import numpy as np
import os

from src.homography import (
    embed,
    compute_homography,
    biliniear_interpolation,
)

# Base directory to compute relative paths
DIR = os.path.dirname(__file__)


if __name__ == "__main__":
    X = np.array([[383, 215], [909, 124], [906, 665], [389, 604]])
    # Load the original matrix
    img = cv2.imread(os.path.join(DIR, './img/cat.jpg'))
    dy, dx, _ = img.shape
    Xp = np.array([[0, 0], [dx, 0], [dx, dy], [0, dy]])
    H = compute_homography(Xp, X)
    print("homography matrix:\n", H)

    background = cv2.imread(os.path.join(DIR, './img/original.jpg'))
    img_out = embed(img, X, background, H, biliniear_interpolation)

    # Write result images.
    cv2.imwrite(os.path.join(DIR, './img/cat_transformed.jpg'), img_out)
