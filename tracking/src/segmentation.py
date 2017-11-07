"""Implementation of the segmentation-related taks"""

import cv2
import numpy


class Segmentation:

    def __init__(self, ref_area=37945.0, ratio=0.9):
        # this magic number is based on the "normal" size of a taxi in the
        # scene. It is necessary because sometimes, very small contours are
        # detected (due to noise in the previous stages).
        self._ref_area = ref_area
        self._ratio = ratio

    def find_contour(self, img):
        """Find the largest contour in the image.
        Returns None if no contour is found.
        """
        im2, contours, hierarchy = cv2.findContours(
            img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            best_area = 0
            cnt = contours[0]
            for x in contours:
                area = cv2.contourArea(x)
                if area > best_area:
                    best_area = area
                    cnt = x
            # Filter outliers.
            if numpy.abs(self._ref_area - best_area) < self._ref_area * self._ratio:
                return cnt

        return None
