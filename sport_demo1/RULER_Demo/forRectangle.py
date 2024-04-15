import cv2
import myutils
import numpy as np

# 读取图像
gray = cv2.imread(r'../img/onlyRedArea.jpg', cv2.IMREAD_GRAYSCALE)  # 替换为您的图片路径
img0 = cv2.imread(r'../img/onlyRedArea.jpg')  # 替换为您的图片路径

# 对图像进行阈值处理
ret, thresh_img = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

edges = cv2.Canny(thresh_img, 50, 150, apertureSize=3)
# # 查找图像中的轮廓
# contours, hierarchy = cv2.findContours(p4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# # 在原始图像上绘制轮廓
# contour_img = cv2.cvtColor(p5, cv2.COLOR_BGR2GRAY)
# cv2.drawContours(p5, contours, -1, (0, 255, 0), 1)
lines = cv2.HoughLines(edges, 1, np.pi/180, 2)

show_lines_img = img0.copy()
# 绘制检测到的直线
for rho, theta in lines[:, 0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(show_lines_img, (x1, y1), (x2, y2), (0, 0, 255), 2,minLineLength=20, maxLineGap=7)

# 显示结果
cv2.imshow('img0', img0)
cv2.imshow('edges', edges)
cv2.imshow('show_lines_img', show_lines_img)
cv2.imshow('thresh_img', thresh_img)


cv2.waitKey(0)
cv2.destroyAllWindows()
