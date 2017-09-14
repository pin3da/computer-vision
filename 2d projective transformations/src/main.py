import cv2
import numpy as np
import os

from math import floor

DIR = os.path.dirname(__file__)


def gen_equations(p, pp):
    """Given one point :p: in the original matrix, and other point :pp: in the
    the destination matrix, generates a pair of equations to estimate the
    corresponding homography."""
    x, y = p[0], p[1]
    xp, yp = pp[0], pp[1]
    return [
        [x, y, 1, 0, 0, 0, - xp * x, - xp * y],
        [0, 0, 0, x, y, 1, - yp * x, - yp * y]
    ]


def compute_homography(X: np.ndarray, Xp: np.ndarray) -> np.ndarray:
    """Computes the homography matrix "H" which holds: X = HXp

    :param X: At least 4 points in the destination matrix.
    :param Xp: At least 4 points in the original matrix.

    The dimentions of X and Xp must be the same.
    """
    assert len(X) == len(Xp)
    n = len(X)
    A = []
    for i in range(n):
        A.extend(gen_equations(X[i], Xp[i]))

    b = []
    for i in range(n):
        b.extend([[Xp[i][0]], [Xp[i][1]]])

    A = np.array(A)
    b = np.array(b)
    ans, resid, rank, sigma = np.linalg.lstsq(A, b)
    assert rank == 8
    w = np.append(ans, [1]).reshape((3, 3))
    return w


def transform_point(point, H):
    x = np.array([[point[0]], [point[1]], [1]])
    xp = np.dot(H, x)
    return np.array([xp[0, 0] / xp[2, 0], xp[1, 0] / xp[2, 0]])


def apply_homography(X, H, interpolate_point):
    """Returns a new image which is equal to X*H"""
    # uint8 is to guarantee integers in the interval [0, 255]
    H = np.linalg.pinv(H)
    ans = np.zeros(X.shape, np.uint8)

    for y in range(X.shape[0]):
        for x in range(X.shape[1]):
            p = transform_point((x, y), H)
            ans[y][x] = interpolate_point(p, X)
    return ans


def valid(x, y, X):
    return 0 <= y < X.shape[0] and 0 <= x < X.shape[1]


def biliniear_interpolation(p, X):
    x, y = p
    x1 = int(floor(p[0]))
    x2 = x1 + 1
    y1 = int(floor(p[1]))
    y2 = y1 + 1
    if valid(x1, y1, X) and valid(x1, y2, X) and valid(x2, y1, X) and valid(x2, y2, X):
        new_pixel = np.array([0, 0, 0])
        for c in range(0, 3):
            left = np.array([[x2 - x, x - x1]])
            mid = np.array([
                [X[y1][x1][c], X[y1][x2][c]],
                [X[y2][x1][c], X[y2][x2][c]],
            ])
            right = np.array([[y2 - y], [y - y1]])
            ans = left.dot(mid).dot(right)[0, 0]
            new_pixel[c] = ans
        return new_pixel
    return np.array([0, 0, 0])


def round_interpolation(p, X):
    x, y = p
    x = int(round(x))
    y = int(round(y))
    if valid(x, y, X):
        return X[y][x]
    return np.array([0, 0, 0])


if __name__ == "__main__":
    X = np.array([[344, 797], [421, 480], [856, 416], [882, 772]])
    Xp = np.array([[475, 10], [10, 10], [10, 625], [475, 625]])
    H = compute_homography(X, Xp)
    print(H)
    img = cv2.imread(os.path.join(DIR, '../img/original.jpg'))
    print(img.shape)
    img_out_cv = cv2.warpPerspective(img, H, (img.shape[0], img.shape[1]))
    img_out = apply_homography(img, H, biliniear_interpolation)
    cv2.imwrite(os.path.join(DIR, '../img/transformed.jpg'), img_out)
    cv2.imwrite(os.path.join(DIR, '../img/transformed_cv.jpg'), img_out_cv)
    # cv2.imshow('image', img)
    # cv2.imshow('image2', img_out)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
