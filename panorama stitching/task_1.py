import cv2
import os

from src.stitcher import Stitcher

DIR = os.path.dirname(__file__)

if __name__ == '__main__':
    stitcher = Stitcher()
    # targets = ['./img/left.jpg', './img/right.jpg']
    targets = ['./img/t1.jpg', './img/t2.jpg', './img/t3.jpg']
    images = [
        cv2.imread(os.path.join(DIR, target)) for target in targets
    ]
    pairs, final = stitcher.stitch(
        images=images,
        destination_matches=os.path.join(DIR, './img/')
    )

    pairs.append(final)

    for i, img in enumerate(pairs):
        cv2.imwrite(os.path.join(DIR, './img/stitch{id}.jpg'.format(id=i)), img)
