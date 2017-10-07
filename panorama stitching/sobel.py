import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('img/cat.jpg', 0)
kernel = np.ones((5,5),np.float32)/25
img = cv2.filter2D(img,-1,kernel)

edges = cv2.Canny(img, int(sys.argv[1]), int(sys.argv[2]))

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
