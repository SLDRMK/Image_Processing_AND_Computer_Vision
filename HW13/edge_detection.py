import cv2
from matplotlib import pyplot as plt

src = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)
canny = cv2.Canny(src, 100, 40)
sift_object = cv2.SIFT.create()
points = sift_object.detect(src, None)
sift = src.copy()
sift = cv2.drawKeypoints(src, points, sift, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.subplot(131, title="Original Image")
plt.imshow(src, cmap="gray")
plt.xticks([]), plt.yticks([])
plt.subplot(132, title="Canny Detected")
plt.imshow(canny, cmap="gray")
plt.xticks([]), plt.yticks([])
plt.subplot(133, title="SIFT Detected")
plt.imshow(sift, cmap="gray")
plt.xticks([]), plt.yticks([])
plt.show()