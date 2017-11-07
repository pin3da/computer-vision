import cv2

from src.background import BgSubtractorOCV, YellowSubtractor
from src.segmentation import Segmentation
from src.tracker import Tracker
from src.utils import plot_text, plot_points

cap = cv2.VideoCapture('data/taxi.mp4')
bg_subtractor = BgSubtractorOCV()
yellow_subtractor = YellowSubtractor()
segmentation = Segmentation()
tracker = Tracker()
video_writer = cv2.VideoWriter(
    'out.avi',
    cv2.VideoWriter_fourcc(*'XVID'),
    20.0,
    (1280, 720)
)

while (1):
    ret, frame = cap.read()
    if frame is None:
        break

    yellow = yellow_subtractor.subtract(frame)
    background = bg_subtractor.subtract(yellow)
    contour = segmentation.find_contour(background)
    if contour is not None:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        tracker.update(contour)

    plot_text(frame, 'Total:%02d' % tracker.cars, (frame.shape[1] - 200, 50))
    plot_points(frame, tracker.path)
    plot_points(frame, tracker.path_kalman, col=(100, 100, 222))

    cv2.imshow('video', frame)
    video_writer.write(frame)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
