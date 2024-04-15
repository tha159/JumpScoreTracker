import cv2
import numpy as np

# 读取图像
img = cv2.imread(r'D:\user\work\sport\sport_demo1\img\onlyRuler.jpg', cv2.IMREAD_GRAYSCALE)  # 替换为您的图片路径

# # 对图像进行阈值处理
# ret, thresh_img = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)

# 查找图像中的轮廓
contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 在原始图像上绘制轮廓
contour_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.drawContours(contour_img, contours, -1, (0, 0, 0), 1)

# 显示结果
cv2.imshow('Original', img)
# cv2.imshow('Threshold', thresh_img)
cv2.imshow('Contours', contour_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
