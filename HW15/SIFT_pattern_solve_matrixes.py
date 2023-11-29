import cv2
import numpy as np

query_src = cv2.imread("Pic_1.jpg", cv2.IMREAD_UNCHANGED)
train_src = cv2.imread("Pic_2.jpg", cv2.IMREAD_UNCHANGED)

sift_object = cv2.SIFT.create(1000)
query_points, query_descriptors = sift_object.detectAndCompute(query_src, None)
train_points, train_descriptors = sift_object.detectAndCompute(train_src, None)

matcher = cv2.DescriptorMatcher.create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE)
matches = list(matcher.match(query_descriptors, train_descriptors))
matches.sort(key=lambda x: x.distance)
size = len(matches)
matches = (matches[:int(size*0.15)])

imMatches = cv2.drawMatches(query_src, query_points, train_src, train_points, matches, None)
cv2.imwrite("matches.jpg", imMatches)

query_points_ = []
train_points_ = []

for match in matches:
    query_points_.append(query_points[match.queryIdx].pt)
    train_points_.append(train_points[match.trainIdx].pt)

H, _ = cv2.findHomography(np.array(query_points_), np.array(train_points_), method=cv2.RANSAC, ransacReprojThreshold=3)
print(H)

# output:
# [[ 1.13002386e+00  1.06410012e-01 -1.14584606e+02]
# [ 7.18882964e-02  1.16936731e+00  1.95297528e+00]
# [ 5.44428643e-05  6.62981426e-05  1.00000000e+00]]

F, _ = cv2.findFundamentalMat(np.array(query_points_), np.array(train_points_), method=cv2.RANSAC, ransacReprojThreshold=3)
print(F)

# output:
# [[ 1.87697275e-06  7.88721034e-06  7.35435247e-03]
# [-6.49732926e-06 -4.47547542e-07  3.07536075e-03]
# [-9.21958689e-03 -3.80107133e-03  1.00000000e+00]]