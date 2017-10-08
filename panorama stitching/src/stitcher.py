"""Stitcher: class to make a fully automatic panorama stitching.

Mainly based on the following OpenCV documentation:
    http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
"""

import cv2
import numpy
import os
from random import randint


class Stitcher:

    def __init__(self):

        # Creates only one descriptor to be used in all the calls to 'stitch'
        self.descriptor = cv2.xfeatures2d.SIFT_create()

    def stitch(self, images, destination_matches=None):
        """stitch an arbitrary number of images, all the intermediate matches
         will be saved in :destination_matches: if it is not None."""

        # Save the partial results to plot the intermediate matches and "carry"
        # the homography matrix.
        descriptors = []
        matches = []
        partial_results = []
        H_carry = None
        img_carry = None

        # Generate partial results for each pair of consecutive images.
        for i, image in enumerate(images):

            descriptors.append(
                # compute visual descriptors for the current image. "i" will be used
                # as identifier to save the descriptors.
                self.generate_descriptors(image, destination_matches, i)
            )
            if i > 0:

                # computes the matching points and the Homography matrix between
                # the current image and the previous one.
                matches.append(
                    self.compute_matches(
                        images[i],
                        images[i - 1],
                        descriptors[i],
                        descriptors[i - 1],
                        i - 1,
                    )
                )

                # H is the homography matrix.
                H, _ = matches[-1]

                #
                partial_results.append(
                    self._stitch(images[i - 1], images[i], H)
                )
                # If is the first homography just copy it.
                # If not, the new carry is the product between the previous carry
                # and the current homography.
                if H_carry is not None:
                    H_carry = H_carry.dot(H)
                    img_carry = self._stitch(img_carry, images[i], H_carry)
                else:
                    H_carry = H.copy()
                    img_carry = partial_results[-1].copy()

        # returns all the intermediate steps and the final image.
        return partial_results, img_carry

    def _stitch(self, image_1, image_2, H):
        """Helper function to stitch two images using the homography :H:

        Also crops the contours of the result image before return it.
        """
        result = cv2.warpPerspective(
            image_2,
            H,
            (image_1.shape[1] + image_2.shape[1], image_2.shape[0]),
        )
        tmp = image_1.shape[1]
        result[0:image_1.shape[0], 0:tmp] = image_1[:, 0:tmp]
        return self._crop_contours(result)

    @staticmethod
    def _crop_contours(image_1):
        gray = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        im2, contours, hierarchy = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)
        return image_1[y:y + h, x:x + w - 40]

    def generate_descriptors(self, image, destination_matches, id_desc):

        # computes descriptors using SIFT.
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        key_points, descriptors = self.descriptor.detectAndCompute(
            gray_image,
            None,
        )
        img = cv2.drawKeypoints(
            image,
            key_points,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        if destination_matches:
            cv2.imwrite(
                os.path.join(
                    destination_matches,
                    'descriptors{id}.jpg'.format(id=id_desc),
                ),
                img,
            )
        return key_points, descriptors

    @staticmethod
    def compute_matches(image_1, image_2, desc_1, desc_2, id_matching):
        """This method computes the matches between the keypoints of
        two images as suggested in the OpenCV tutorial. It uses a
        Fast Approximate Nearest Neighbor algorithm (FAAN) in order
        to find the matches. After that computes the best homography
        using some of those matches (only the inliers) using Random
        Sample Consensus (RANSAC)."""
        kp1, des1 = desc_1
        kp2, des2 = desc_2

        index_params = dict(algorithm=0, trees=5)  # 0 is FLANN_INDEX_KDTREE
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        # Only stores the "good" matches. Cutting with a predefined
        # threshold.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        # Define the minimum number of good matches
        if len(good) < 10:
            raise AssertionError("Not enough matches were found - %d/%d" % (len(good), 10))

        # Prepare the points to be used in the estimation of the homography.
        src_pts = numpy.float32([
            kp1[m.queryIdx].pt for m in good
        ]).reshape(-1, 1, 2)
        dst_pts = numpy.float32([
            kp2[m.trainIdx].pt for m in good
        ]).reshape(-1, 1, 2)

        # computes the homography matrix using RANSAC.
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        draw_params = dict(
            matchColor=(randint(200, 255), randint(0, 255), randint(0, 255)),
            singlePointColor=None,
            matchesMask=matches_mask,  # draw only inliers
            flags=2,
        )

        image_3 = cv2.drawMatches(
            image_1,
            kp1,
            image_2,
            kp2,
            good,
            None,
            **draw_params,
        )
        cv2.imwrite('img/matches{id}.jpg'.format(id=id_matching), image_3)
        return M, mask
