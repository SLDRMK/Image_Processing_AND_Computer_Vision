import cv2
import numpy as np
import open3d as o3d
from pandas import DataFrame
from pyntcloud import PyntCloud

# SIFT matching

K = np.array([[1.23311611e+03, 0.00000000e+00, 8.35993852e+02],
              [0.00000000e+00, 1.23723649e+03, 6.91971597e+02],
              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

query_src = cv2.imread("Pic/Pic_1.jpg", cv2.IMREAD_UNCHANGED)
train_src = cv2.imread("Pic/Pic_2.jpg", cv2.IMREAD_UNCHANGED)

sift_object = cv2.SIFT.create(1000)
query_points, query_descriptors = sift_object.detectAndCompute(query_src, None)
train_points, train_descriptors = sift_object.detectAndCompute(train_src, None)

matcher = cv2.DescriptorMatcher.create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE)
matches = list(matcher.match(query_descriptors, train_descriptors))
size = len(matches)

query_points_ = []
train_points_ = []

for match in matches:
    query_points_.append(query_points[match.queryIdx].pt)
    train_points_.append(train_points[match.trainIdx].pt)

# calculate F, E, R, t

F, _ = cv2.findFundamentalMat(np.array(query_points_), np.array(train_points_), method=cv2.RANSAC, ransacReprojThreshold=3)
# print(F)

E = K.T @ F @ K
U, _, V_T = np.linalg.svd(E, full_matrices=True)
U = np.array(U)
V_T = np.array(V_T)

W = np.array([[0, -1, 0],
              [1, 0, 0],
              [0, 0, 1]])

R_1 = U @ W @ V_T
R_2 = U @ W.T @ V_T
t_x = E @ np.linalg.inv(R_1)
u = 0.5 * np.array([[t_x[2][1] - t_x[1][2]],
                    [t_x[0][2] - t_x[2][0]],
                    [t_x[1][0] - t_x[0][1]]])
R = [R_1, R_1, R_2, R_2]
t = [u, -u, u, -u]
P_2 = [np.hstack([R_1, u]), np.hstack([R_1, -u]),
       np.hstack([R_2, u]), np.hstack([R_2, -u])]
P_1 = np.hstack([np.identity(3), np.zeros((3, 1))])

# choose the best P_2

calculate = np.zeros(4)

for match_index in range(size):
    print("processing: ", match_index, " / ", size)
    for i in range(4):
        print(i)
        A = K @ P_2[i]
        print(train_points_[match_index])
        x = list(train_points_[match_index])
        x.append(1)
        x = np.reshape(x, (3, 1))
        X = np.linalg.inv(A.T @ A) @ A.T @ x
        X = np.reshape(X, (4, ))
        if X[3] == 0:
            print("solving error")
            continue
        X = X[0:3] / X[3]
        if X[2] <= 0:
            print("not in front of camera 1")
            continue
        dot_value = (np.reshape(X.T, (3,1)) + R[i].T @ t[i]).T @ np.reshape(R[i][2,:], (3, 1))
        if dot_value > 0:
            calculate[i] += 1
        else:
            print("not in front of camera 2")

P_2_ = P_2[0]
for i in range(1, 4):
    if calculate[i] > calculate[0]:
        P_2_ = P_2[i]

points_3d = []

# build the 3d point cloud

for match_index in range(size):
    print("processing: ", match_index, " / ", size)
    A = K @ P_2_
    x = list(train_points_[match_index])
    x.append(1)
    x = np.reshape(x, (3, 1))
    X = np.linalg.inv(A.T @ A) @ A.T @ x
    X = np.reshape(X, (4, ))
    if X[3] == 0:
        print("solving error")
        continue
    points_3d.append(X[0:3] / X[3])
    print(X[0:3] / X[3])

points_3d = np.array(points_3d)

print(points_3d.shape)

point_cloud_raw = DataFrame(points_3d[:,0:3])
point_cloud_raw.columns = ['x', 'y', 'z']
point_cloud_pynt = PyntCloud(point_cloud_raw)

point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)

o3d.visualization.draw_geometries([point_cloud_o3d])