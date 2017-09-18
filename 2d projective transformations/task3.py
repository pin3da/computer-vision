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
    # These points describe the embedding area.
    X = np.array([[383, 215], [909, 124], [906, 665], [389, 604]])
    # Load the original matrix
    img = cv2.imread(os.path.join(DIR, './img/cat.jpg'))
    dy, dx, _ = img.shape

    # Use the points of the original image to determine the homography
    Xp = np.array([[0, 0], [dx, 0], [dx, dy], [0, dy]])
    H = compute_homography(Xp, X)
    print("homography matrix:\n", H)

    # load the image that we want to use as background
    background = cv2.imread(os.path.join(DIR, './img/original.jpg'))

    # Apply the projective transformation to the pixes inside the embedding area
    img_out = embed(img, X, background, H, biliniear_interpolation)

    # Write result images.
    cv2.imwrite(os.path.join(DIR, './img/cat_transformed.jpg'), img_out)
