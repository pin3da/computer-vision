""" homography.py

Main module to compute and apply homographies to images, this module also
contains some interpolations that can be used in the transformation.
"""

import numpy as np

from math import floor


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
    # Construction of the linear system A h = b

    A = []
    # Each point generates a pair of equations. In this part, we are stacking
    # all the equations on the same matrix.
    for i in range(n):
        A.extend(gen_equations(X[i], Xp[i]))

    # Construction of the result
    b = []
    for i in range(n):
        b.extend([[Xp[i][0]], [Xp[i][1]]])

    # A and b are python arrays, they need to be converted to numpy arrays.
    A = np.array(A)
    b = np.array(b)

    # Solves the linear system A h = B (finds h) using least mean squares.
    ans, resid, rank, sigma = np.linalg.lstsq(A, b)

    # Asserts that we can find all the unknowns.
    assert rank == 8

    # Reshape the result vector as an homography matrix and add h[3][3] = 1
    w = np.append(ans, [1]).reshape((3, 3))
    return w


def transform_point(point, H):
    # Converts the point to homogeneous coordinates in order to apply the
    # homography
    x = np.array([[point[0]], [point[1]], [1]])
    xp = np.dot(H, x)
    # Converts the point back to euclidean coordinates.
    return np.array([xp[0, 0] / xp[2, 0], xp[1, 0] / xp[2, 0]])


def apply_homography(X, H, interpolate_point):
    """Returns a new image which is equal to X*H
    :param X: original image
    :param H: homography matrix
    :param interpolate_point: function to interpolate the points
    """
    # uint8 is to guarantee integers in the interval [0, 255]
    ans = np.zeros(X.shape, np.uint8)

    # Inverts the homography to "find" the points in the new image from the
    # interpolation of points in the original image.
    H = np.linalg.pinv(H)

    # Apply the same interpolation to all the pixels
    for y in range(X.shape[0]):
        for x in range(X.shape[1]):
            p = transform_point((x, y), H)
            ans[y][x] = interpolate_point(p, X)
    return ans


def valid(x, y, X):
    """Checks if the point x, y is inside the matrix X."""
    return 0 <= y < X.shape[0] and 0 <= x < X.shape[1]


def biliniear_interpolation(p, X):
    """Computes a bilinear interpolation of the point p in the matrix X.
    p can have non-integer coordinates.
    """
    x, y = p
    x1 = floor(p[0])
    x2 = x1 + 1
    y1 = floor(p[1])
    y2 = y1 + 1
    # Checks if all the points are inside the image, returns [0,0,0] otherwise.
    if valid(x1, y1, X) and valid(x1, y2, X) and valid(x2, y1, X) and valid(x2, y2, X):
        new_pixel = np.array([0, 0, 0])
        # Apply the same transformation to each color
        for c in range(0, 3):
            left = np.array([[x2 - x, x - x1]])
            mid = np.array([
                [X[y1][x1][c], X[y2][x1][c]],
                [X[y1][x2][c], X[y2][x2][c]],
            ])
            right = np.array([[y2 - y], [y - y1]])
            ans = left.dot(mid).dot(right)[0, 0]
            new_pixel[c] = ans
        return new_pixel
    return np.array([0, 0, 0])


def round_interpolation(p, X):
    """Returns the nearest integer point to p.
    """
    x, y = p
    x = int(round(x))
    y = int(round(y))
    # Checks if the pixel exists in the original matrix
    if valid(x, y, X):
        return X[y][x]
    return np.array([0, 0, 0])


def turn(a, b, c):
    """Returns positive if the points a-b-c makes a left turn.
    Returns negative if the points a-b-c makes a right turn.
    Returns 0.0 if the points a-b-c are colineal.
    """
    z = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if abs(z) < 1e-9:
        return 0.0
    return z


def point_in_polygon(p, poly):
    """ Determines if the point :p: is inside the polygon :poly:
    The polygon must be convex.
    Checks if all the edges of the polygon are to the same side of the point p
    """
    mask = 0
    n = len(poly)
    for i in range(n):
        j = (i + 1) % n
        z = turn(poly[i], poly[j], p)
        if z < 0.0:
            mask |= 1
        elif z > 0.0:
            mask |= 2
        elif z == 0.0:
            return False

        if mask == 3:  # Both sides seen.
            return False
    return mask != 0  # Exactly one side seen.


def embed(X, polygon_embed, background, H, interpolate_point):
    # uint8 is to guarantee integers in the interval [0, 255]
    ans = np.zeros(background.shape, np.uint8)

    # Inverts the homography to "find" the points in the new image from the
    # interpolation of points in the original image.
    H = np.linalg.pinv(H)

    # Apply the same interpolation to all the pixels
    for y in range(background.shape[0]):
        for x in range(background.shape[1]):
            # Only transform the pixels that are in the embedding area
            if point_in_polygon((x, y), polygon_embed):
                p = transform_point((x, y), H)
                ans[y][x] = interpolate_point(p, X)
            else:
                ans[y][x] = background[y][x]
    return ans
