import cv2
import matplotlib.pyplot as plt
import numpy as np

# image binaryzation
def binarazate(src, thresh):
    row, col = src.shape
    dst = src.copy()
    for i in range(row):
        for j in range(col):
            if src[i][j] > thresh:
                dst[i][j] = 255
            else:
                dst[i][j] = 0
    return dst

thresh = 100

# define different morphology kernals
kernal1 = np.ones((3,3), np.uint8)
kernal2 = np.ones((7,7), np.uint8)
kernal3 = np.ones((2,2), np.uint8)
for i in range(5):
    for j in range(5):
        kernal2[i+1][j+1] = 0

original = cv2.imread("./1.png", cv2.IMREAD_GRAYSCALE)

# binaryzate the image
binary = binarazate(original, thresh=thresh)

changed = cv2.dilate(binary, kernal2)
changed = cv2.erode(changed, kernal1)
changed = cv2.erode(changed, kernal1)
changed = cv2.erode(changed, kernal1)
changed = cv2.dilate(changed, kernal1)

row, col = original.shape
for i in range(row):
        for j in range(col):
            if changed[i][j] and binary[i][j]:
                changed[i][j] = 255
            else:
                changed[i][j] = 0
changed = cv2.dilate(changed, kernal3)

# count connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(changed, connectivity=8)
print("There are", num_labels-1, "bacteria")

plt.subplot(131, title="original")
plt.xticks([]), plt.yticks([])
plt.imshow(original, cmap="gray", vmin=0, vmax=255)
plt.subplot(132, title="binary")
plt.xticks([]), plt.yticks([])
plt.imshow(binary, cmap="gray", vmin=0, vmax=255)
plt.subplot(133, title="modified")
plt.xticks([]), plt.yticks([])
plt.imshow(changed, cmap="gray", vmin=0, vmax=255)
plt.show()