"""Tracker-related implementations"""

import numpy as np
import cv2


class Tracker:
    """Counts the number of taxis in the scene and keeps track of the path of
    the last detected taxi"""

    def __init__(self, max_diff=300):
        self.cars = 0
        # real path
        self.path = []
        # filtered path using kalman filter
        self.path_kalman = []
        # max distance in pixels to determine if a object is the known or is new.
        self._max_diff = max_diff
        self._start = [40, 250]
        self._last_point = np.array([6666, 6666])
        self._setup_filter(self._start)

    def _setup_filter(self, start):
        """Initialize all the variables needed by the Kalman filter."""
        self.filter = cv2.KalmanFilter(4, 2)

        self.filter.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)

        self.filter.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], np.float32)

        self.filter.processNoiseCov = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32) * 0.03

        self.filter.correct(np.array([[start[0]], [start[1]]], np.float32))

    def update(self, contour):
        """Updates the state of the tracking using the last detected contour"""

        # We use the moments of the contour to determine the "position" of the car.
        # Based on: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html#moments
        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        raw_prediction = self.filter.predict()
        current_predict = np.array([
            int(raw_prediction[0]),
            int(raw_prediction[1])
        ])
        current = np.array([cx, cy])
        self.filter.correct(np.array([[cx], [cy]], np.float32))
        print("Prediction", current_predict)
        print("Medition", current)
        print("Dist", np.linalg.norm(self._last_point - current_predict))
        if np.linalg.norm(self._last_point - current) > self._max_diff:
            self.cars += 1
            self.path = [self._start]
            self.path_kalman = []
            self._setup_filter(self._start)
        else:
            self.path_kalman.append(current_predict)
            self.path.append(current)

        self._last_point = current
