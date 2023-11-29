import cv2
import numpy as np
import matplotlib.pyplot as plt

M1 = np.float32([[0.7, 0.3, 100],
              [0.2, 0.8, 50]])                                  # define the transform matrix

M2 = np.float32([[0.7, 0.3, 100],
                [0.2, 0.8, 50],
                [0.01, 0.01, 1]])

img = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)              # read the image
rows, cols = img.shape                                          # record the original size

img1 = cv2.warpAffine(src=img, M=M1, dsize=(cols, rows))        # operate the Affine Transformation

img2 = cv2.warpPerspective(src=img, M=M2, dsize=(cols, rows))   # operate the Projective Transformation

plt.subplot(311, title="original")
plt.imshow(img, cmap="gray", vmin=0, vmax=255)
plt.subplot(312, title="Affine")
plt.imshow(img1, cmap="gray", vmin=0, vmax=255)
plt.subplot(313, title="Perspective")
plt.imshow(img2, cmap="gray", vmin=0, vmax=255)

plt.show()                                                      # compare the images