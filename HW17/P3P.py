import cv2
import numpy as np

K = np.array([[1.23311611e+03, 0.00000000e+00, 8.35993852e+02],
              [0.00000000e+00, 1.23723649e+03, 6.91971597e+02],
              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

CHECKERBOARD = (8,11)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objpoints = []
imgpoints = [] 

objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

image = "calibration.jpg"
img = cv2.imread(image)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
points_2d = np.reshape(corners, (88, 2))
points_3d = []
for i in range(11):
    for j in range(8):
        points_3d.append([0.15*i+1, 0.15*j+1, 0])

rvecs = np.zeros(3)
tvecs = np.zeros(3)

_, rvecs, tvecs = cv2.solvePnP(np.array(points_3d), points_2d, K, None, rvecs, tvecs, useExtrinsicGuess=False, flags=cv2.SOLVEPNP_SQPNP)
print("rotation vector: ", rvecs)
print("translation vector: ", tvecs)

# output: 
# rotation vector:  [ 2.76899139 -0.08432218  0.53041338]
# translation vector:  [-1.18027247  1.24608765  2.73564148]