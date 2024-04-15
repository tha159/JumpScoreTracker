import cv2
import myutils
import numpy as np

# 读取图像
gray = cv2.imread(r'D:\user\work\sport\sport_demo1\img\onlyRuler.jpg', cv2.IMREAD_GRAYSCALE)  # 替换为您的图片路径
img0 = cv2.imread(r'D:\user\work\sport\sport_demo1\img\onlyRuler.jpg')  # 替换为您的图片路径

# 对图像进行阈值处理
ret, thresh_img = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

smoothed_image = cv2.GaussianBlur(thresh_img, (5, 5), 200)
kernel = np.ones((2, 2), np.uint8)  # 可根据实际情况调整核的大小
opening = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel)
erosion = cv2.erode(thresh_img, kernel, iterations=2)
edges = cv2.Canny(thresh_img, 50, 150, apertureSize=3)
erosion_wb = cv2.bitwise_not(erosion)
# 使用高斯滤波器进行平滑处理

# # 查找图像中的轮廓
# contours, hierarchy = cv2.findContours(p4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# # 在原始图像上绘制轮廓
# contour_img = cv2.cvtColor(p5, cv2.COLOR_BGR2GRAY)
# cv2.drawContours(p5, contours, -1, (0, 255, 0), 1)

# 显示结果
cv2.imshow('img0', img0)
cv2.imshow('edges', edges)
cv2.imshow('smoothed_image', smoothed_image)
cv2.imshow('thresh_img', thresh_img)
cv2.imshow('opening', opening)
cv2.imshow('erosion', erosion)
cv2.imshow('erosion_wb', erosion_wb)
# cv2.imwrite('../img/erosion.jpg',erosion_wb)


cv2.waitKey(0)
cv2.destroyAllWindows()
