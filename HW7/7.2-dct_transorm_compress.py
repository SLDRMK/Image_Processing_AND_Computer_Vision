import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(filename='./lena.png', flags=cv2.IMREAD_GRAYSCALE)
plt.subplot(151, title="original")
plt.xticks([]), plt.yticks([])
plt.imshow(img, cmap="gray", vmin=0, vmax=255)

dct = cv2.dct(np.float32(img))
plt.subplot(152, title="dct transform")
plt.xticks([]), plt.yticks([])
plt.imshow(dct, cmap="gray", vmin=0, vmax=255)

idct = cv2.idct(dct)
plt.subplot(153, title="idct transformed")
plt.xticks([]), plt.yticks([])
plt.imshow(idct, cmap="gray", vmin=0, vmax=255)

dct_compressed = np.zeros(img.shape)
dct_compressed[0:256][0:256] = dct[0:256][0:256]
plt.subplot(154, title="dct compressed")
plt.xticks([]), plt.yticks([])
plt.imshow(dct_compressed, cmap="gray", vmin=0, vmax=255)

idct_compressed = cv2.idct(dct_compressed)
plt.subplot(155, title="idct compressed")
plt.xticks([]), plt.yticks([])
plt.imshow(idct_compressed, cmap="gray", vmin=0, vmax=255)

plt.show()