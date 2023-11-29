import cv2
import matplotlib.pyplot as plt
import cmath
import numpy as np


# calculate H
def degradation(w, h, a, b, T):
    x_middle = w / 2 + 1
    y_middle = h / 2 + 1
    Mo = np.zeros((w, h), dtype=complex)
    for u in range(w):
        for v in range(h):
            temp = cmath.pi * ((u - x_middle) * a + (v - y_middle) * b)
            if temp == 0:
                Mo[u, v] = T
            else:
                Mo[u, v] = T * cmath.sin(temp) / temp * cmath.exp(- 1j * temp)
    return Mo

if __name__ == "__main__":
    # read the image
    src = cv2.imread("blurred.jpg", cv2.IMREAD_GRAYSCALE)
    plt.subplot(121, title="original")
    plt.xticks([]), plt.yticks([])
    plt.imshow(src, cmap="gray", vmin=0, vmax=255)

    a = 0.035
    b = -0.035
    T = 1
    r = 0.000005

    # do fft
    w, h = src.shape
    G = np.fft.fft2(src)
    G_shift = np.fft.fftshift(G)

    # calculate H
    H = degradation(w, h, a, b, T)
    p = np.array([[0,-1,0],
                [-1,4,-1],
                [0,-1,0]])
    P = np.fft.fft2(p,[w, h])

    # restore the image
    F = G_shift *(np.conj(H) / (np.abs(H)**2+r*np.abs(P)**2))

    # do ifft and show the result
    f = np.abs(np.fft.ifft2(F))
    result = (f / np.max(f) * 255)
    result = result.astype("uint8")
    plt.subplot(122, title="restored")
    plt.xticks([]), plt.yticks([])
    plt.imshow(result, cmap="gray", vmin=0, vmax=255)
    plt.show()