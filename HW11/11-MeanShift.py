import cv2
import numpy as np
from scipy.spatial import distance
from PIL import Image
import matplotlib.pyplot as plt

img = cv2.imread("./wall.JPG", cv2.IMREAD_ANYCOLOR)
filter = cv2.pyrMeanShiftFiltering(img, 100, 40, None, 2)   # do mean shift filtering
cv2.imwrite("result.png", np.hstack((img, filter)))

print("Finished!")
row, col, _ = img.shape
mask = np.zeros((row, col))
type = 1
for i in range(row):
    for j in range(col):
        if mask[i][j] != 0:
            continue
        mask[i][j] = type
        for m in range(row):
            for n in range(col):
                if mask[m][n] != 0:
                    continue
                if np.linalg.norm(filter[i][j] - filter[m][n]) < 300:
                    mask[m][n] = type
                #print(m, n, end=' ')
        print(i, j, type)
        type += 1                                           # combine similar pixels

cv2.imwrite("mask.png", mask)
mask = Image.open("mask.png")
mask = plt.cm.jet(mask)                                     # fake color

plt.subplot(131, title="original")
plt.xticks([]), plt.yticks([])
plt.imshow(img)
plt.subplot(132, title="filted")
plt.xticks([]), plt.yticks([])
plt.imshow(filter)
plt.subplot(133, title="mask")
plt.xticks([]), plt.yticks([])
plt.imshow(mask)
plt.show()