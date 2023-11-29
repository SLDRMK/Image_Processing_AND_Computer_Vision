import cv2
import matplotlib.pyplot as plt

# read the image
src = cv2.imread("5.jpg", cv2.IMREAD_GRAYSCALE)
plt.subplot(141, title="original")
plt.xticks([]), plt.yticks([])
plt.imshow(src, cmap="gray", vmin=0, vmax=255)

# calculate x_grad and y_grad
x_grad = cv2.convertScaleAbs(cv2.Sobel(src, ddepth=-1, dx=1, dy=0))
plt.subplot(142, title="x_grad")
plt.xticks([]), plt.yticks([])
plt.imshow(x_grad, cmap="gray", vmin=0, vmax=255)
y_grad = cv2.convertScaleAbs(cv2.Sobel(src, ddepth=-1, dx=0, dy=1))
plt.subplot(143, title="y_grad")
plt.xticks([]), plt.yticks([])
plt.imshow(y_grad, cmap="gray", vmin=0, vmax=255)

# add x_grad and y_grad
dst = cv2.convertScaleAbs(cv2.add(x_grad, y_grad, dtype=cv2.CV_16S))
plt.subplot(144, title="result")
plt.xticks([]), plt.yticks([])
plt.imshow(dst, cmap="gray", vmin=0, vmax=255)
plt.show()