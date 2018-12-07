"""
可分离的滤波器
实例与时间复杂度比较
"""
import numpy as np
import cv2

image = cv2.imread("1.png")


def rectangle(K):
    return np.ones((K, K)) / K ** 2


rec3_v = np.array([1, 1, 1]) / 3
rec3 = np.outer(rec3_v, rec3_v)

double_linear_v = np.array([1, 2, 1]) / 4
double_linear = np.outer(double_linear_v, double_linear_v)

gauss_v = np.array([1, 4, 6, 4, 1]) / 16
gauss = np.outer(gauss_v, gauss_v)

sobel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) / 8

corner_v = np.array([1, -2, 1]) / 2
corner = np.outer(corner_v, corner_v)

out = cv2.filter2D(image, -1, corner)

out_self = cv2.GaussianBlur(image,5,)

print(sobel)
print(corner)

cv2.namedWindow("1")
cv2.imshow("1", image)

cv2.namedWindow("2")
cv2.imshow("2", out)

cv2.waitKey(0)
