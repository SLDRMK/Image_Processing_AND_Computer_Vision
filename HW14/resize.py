import cv2
import numpy as np

src = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)
rows, cols = src.shape

dst0 = np.ones((512, 512)) * 255
dst = cv2.resize(src, dsize=(256, 256))
dst0[128:384, 128:384] = dst
cv2.imwrite("lena_small.png", dst0)