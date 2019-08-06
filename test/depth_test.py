# -*- coding: utf-8 -*-
import numpy as np

P2 = np.asarray([[
    7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01, 0.000000e+00,
    7.215377e+02, 1.728540e+02, 2.163791e-01, 0.000000e+00, 0.000000e+00,
    1.000000e+00, 2.745884e-03
]]).reshape((3, 4))

K = P2[:3, :3]
KT = P2[:, -1]
T = np.dot(np.linalg.inv(K), KT)

# x,y,z
C_3d = np.asarray([-16.53, 2.39, 58.49])
# h,w,l
dim = np.asarray([1.67, 1.87, 3.69])

bottom_3d = C_3d + np.asarray([0, 0.5 * dim[0], 0])
top_3d = C_3d - np.asarray([0, 0.5 * dim[0], 0])

bottom_3d_homo = np.append(bottom_3d, 1)
top_3d_homo = np.append(top_3d, 1)

bottom_2d_homo = np.dot(P2, bottom_3d_homo)
top_2d_homo = np.dot(P2, top_3d_homo)

lambda_bottom = bottom_2d_homo[-1]
bottom_2d_homo = bottom_2d_homo / lambda_bottom
bottom_2d = bottom_2d_homo[:-1]

lambda_top = top_2d_homo[-1]
top_2d_homo = top_2d_homo / lambda_top
top_2d = top_2d_homo[:-1]

delta_2d = top_2d - bottom_2d
delta_3d = dim[0]

f = K[0, 0]
# validate algorithm
lambda_ = f * dim[0] / delta_2d[1]
lambda_ = f * dim[0] / 20.2

C_2d = np.asarray([407.9734, 201.73137])
C_2d_homo = np.append(C_2d, 1)

C_3d_ = lambda_ * np.dot(np.linalg.inv(K), C_2d_homo) - T

# lambda to depth
depth = lambda_ - T[-1]
# print(depth)
# print(C_3d[-1])
print(C_3d)
print(C_3d_)
