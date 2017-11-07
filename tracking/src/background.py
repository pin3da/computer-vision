"""Implementation of the background subtraction tasks"""

import cv2
import numpy


class IBgSubtractor:
    """Defines the interface for all the Background Substractors
    used with this module."""
    def subtract(img):
        """Substracts the background from the image :img:"""
        AssertionError("Not implemented")


class BgSubtractorOCV(IBgSubtractor):
    """OpenCV background subtractor"""
    def __init__(self, treshold=25):
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.fgbg.setVarThreshold(treshold)

    def subtract(self, img):
        return self.fgbg.apply(img)


class YellowSubtractor(IBgSubtractor):
    """Custom subtractor in charge of the identification of yellow objects
    in the scene."""
    def __init__(self):
        # define color space to filter the image
        self._lower = numpy.array([10, 90, 90])
        self._upper = numpy.array([90, 200, 200])

    def subtract(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._lower, self._upper)
        return cv2.bitwise_and(img, img, mask=mask)
