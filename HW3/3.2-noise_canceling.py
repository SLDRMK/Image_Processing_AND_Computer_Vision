import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

img = np.float32(cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE))              # read the image

rows, cols = img.shape

# use band-pass filter to shut down specific frequencies between R1 and R2 to eliminate the noise
# let R2 be 500 > 470, and use dichotomy to tune R1 to make the noise disappear,
# and then tune R2 in the same way to eliminate the noise and save as much information as possible
R1 = 173
R2 = 176

dft = cv2.dft(img, flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
# to do dft and fft

mask = np.ones((rows, cols, 2), np.uint8)
mask_show = np.ones((rows, cols))
for i in range(rows):
    for j in range(cols):
        distance = (i - rows / 2) * (i - rows / 2) + (j - cols / 2) * (j - cols / 2)
        if distance > R1 * R1 and distance < R2 * R2:
            mask[i][j] = mask_show[i][j] = 0
        else:
            mask_show[i][j] = 255
# to create a mask

fshift = dft_shift*mask
magnitude_spectrum_changed = 20*np.log(cv2.magnitude(fshift[:,:,0],fshift[:,:,1]))
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift, flags=cv2.DFT_SCALE)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
# to do ifft and idft

plt.subplot(151, title="original photo")
plt.xticks([]), plt.yticks([])
plt.imshow(img, cmap="gray", vmin=0, vmax=255)
plt.subplot(152, title="original frequency")
plt.xticks([]), plt.yticks([])
plt.imshow(magnitude_spectrum, cmap="gray", vmin=0, vmax=255)
plt.subplot(153, title="mask")
plt.xticks([]), plt.yticks([])
plt.imshow(mask_show, cmap="gray", vmin=0, vmax=255)
plt.subplot(154, title="edited frequency")
plt.xticks([]), plt.yticks([])
plt.imshow(magnitude_spectrum_changed, cmap="gray", vmin=0, vmax=255)
plt.subplot(155, title="edited photo")
plt.xticks([]), plt.yticks([])
plt.imshow(img_back, cmap="gray", vmin=0, vmax=255)
plt.show()