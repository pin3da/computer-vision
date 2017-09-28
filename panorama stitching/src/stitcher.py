# DOC: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
import cv2
import numpy
import os
from random import randint


class Stitcher:

    def __init__(self):
        self.descriptor = cv2.xfeatures2d.SIFT_create()

    def stich(self, images, destination_matches=None):
        descriptors = []
        matches = []
        partial_results = []
        H_carry = None
        img_carry = None
        for i, image in enumerate(images):
            descriptors.append(
                self.generate_descriptors(image, destination_matches, i)
            )
            if i > 0:
                matches.append(
                    self.compute_matches(
                        images[i],
                        images[i - 1],
                        descriptors[i],
                        descriptors[i - 1],
                        i - 1,
                    )
                )
                H, mask = matches[-1]
                partial_results.append(
                    self._stitch(images[i - 1], images[i], H)
                )
                if H_carry is not None:
                    H_carry = H_carry.dot(H)
                    img_carry = self._stitch(img_carry, images[i], H_carry)
                else:
                    H_carry = H.copy()
                    img_carry = partial_results[-1].copy()

        return partial_results, img_carry

    def _stitch(self, image_1, image_2, H, border_mode=None):
        result = cv2.warpPerspective(
            image_2,
            H,
            (image_1.shape[1] + image_2.shape[1], image_2.shape[0]),
        )
        tmp = image_1.shape[1]
        result[0:image_1.shape[0], 0:tmp] = image_1[:, 0:tmp]
        return self._crop_contours(result)

    def _crop_contours(self, image_1):
        gray = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)
        return image_1[y:y + h, x:x + w - 40]

    def generate_descriptors(self, image, destination_matches, id):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        key_points, descriptors = self.descriptor.detectAndCompute(gray_image, None)
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
                    'descriptors{id}.jpg'.format(id=id),
                ),
                img,
            )
        return (key_points, descriptors)

    def compute_matches(self, image_1, image_2, desc_1, desc_2, id):
        img1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
        kp1, des1 = desc_1
        kp2, des2 = desc_2

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        MIN_MATCH_COUNT = 10
        if len(good) > MIN_MATCH_COUNT:
            src_pts = numpy.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = numpy.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            h, w = img1.shape
            pts = numpy.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            img2 = cv2.polylines(img2, [numpy.int32(dst)], True, 255, 3, cv2.LINE_AA)

        else:
            print("Not enough matches were found - %d/%d" % (len(good), MIN_MATCH_COUNT))
            matchesMask = None

        draw_params = dict(
            matchColor=(randint(200, 255), randint(0, 255), randint(0, 255)),  # draw matches in green color
            singlePointColor=None,
            matchesMask=matchesMask,  # draw only inliers
            flags=2,
        )

        img3 = cv2.drawMatches(image_1, kp1, image_2, kp2, good, None, **draw_params)
        cv2.imwrite('img/matches{id}.jpg'.format(id=id), img3)
        return (M, mask)
