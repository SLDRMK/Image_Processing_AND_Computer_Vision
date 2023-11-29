import cv2
import numpy as np
import math
import scipy.linalg as sl

r_squared = 100

def distance_squared(i, j, width):
    x1 = i / width
    y1 = i % width
    x2 = j / width
    y2 = j % width
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)

def pixel_distance_squared(pixeli, pixelj):
    return float((pixeli[0] - pixelj[0]) * (pixeli[0] - pixelj[0]) \
                 + (pixeli[1] - pixelj[1]) * (pixeli[1] - pixelj[1]) \
                    + (pixeli[2] - pixelj[2]) * (pixeli[2] - pixelj[2]))

src = cv2.imread("source.JPG", cv2.IMREAD_ANYCOLOR)

width, height, _ = src.shape
num_of_pixels = width * height
pixels = np.reshape(src, (num_of_pixels, 3))
W = np.zeros((num_of_pixels, num_of_pixels))
D = np.zeros((num_of_pixels, num_of_pixels))

distance_squared_ = 0
base_coefficient_ = 0

sigma_I_squared_ = np.var(pixels)
index = np.zeros(num_of_pixels)
for i in range(num_of_pixels):
    index[i] = i
sigma_index_squared = np.var(index)

for i in range(num_of_pixels):
    for j in range(i, num_of_pixels):
        distance_ = distance_squared(i, j, width)
        print(i, j)
        if distance_ > r_squared:
            continue
        elif i == j:
            W[i][j] = 1
        else:
            base_coefficient_ = math.exp(-1 * pixel_distance_squared(pixels[i], pixels[j]) / sigma_index_squared)
            W[i][j] = W[j][i] = math.exp(-1 * float(distance_squared_) / sigma_I_squared_) * base_coefficient_
    D[i][i] = np.sum(W[i][:])
np.save("W.npy", W)
np.save("D.npy", D)

eigen_value, eigen_vector = sl.eig(D - W, D)
print("eigen value:", eigen_value)
np.save("eigen_value.npy", eigen_value)
np.save("eigen_vector.npy", eigen_vector)

# eigen_value = np.load("eigen_value.npy")
# eigen_vector = np.load("eigen_vector.npy")
min1 = min2 = math.inf
index1 = index2 = -1
for i in range(eigen_value.size):
    if np.abs(eigen_value[i]) >= min2:
        continue
    elif np.abs(eigen_value[i]) >= min1 and np.abs(eigen_value[i]) < min2:
        min2 = eigen_value[i]
        index2 = i
    else:
        min2 = min1
        index2 = index1
        min1 = eigen_value[i]
        index1 = i
second_smallest_eigenvector = eigen_vector[:, index2]

print("second smallest eigenvector: ", second_smallest_eigenvector)

threshold = np.mean(second_smallest_eigenvector)
print("threshold: ", threshold)
mask = np.zeros(num_of_pixels)
for i in range(num_of_pixels):
    if second_smallest_eigenvector[i] > threshold:
        mask[i] = 255
mask = mask.reshape(width, height)
cv2.imwrite("mask.png", mask)

