import cv2
import numpy as np
import os

from src.homography import (
    apply_homography,
    compute_homography,
    biliniear_interpolation,
    round_interpolation,
)

# Base directory to compute relative paths
DIR = os.path.dirname(__file__)


if __name__ == "__main__":
    X = np.array([[383, 215], [909, 124], [906, 665], [389, 604]])
    Xp = np.array([[344, 198], [944, 198], [944, 689], [344, 689]])
    H = compute_homography(X, Xp)
    print("homography matrix:\n", H)

    # Load the original matrix
    img = cv2.imread(os.path.join(DIR, './img/original.jpg'))

    # Apply the same homography matrix with different estimation methods
    img_out = apply_homography(img, H, biliniear_interpolation)
    img_round = apply_homography(img, H, round_interpolation)

    # Write result images.
    cv2.imwrite(os.path.join(DIR, './img/transformed.jpg'), img_out)
    cv2.imwrite(os.path.join(DIR, './img/transformed_round.jpg'), img_round)
