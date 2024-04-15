import cv2
import myutils
import numpy as np

# 读取图像
img = cv2.imread(r'D:\user\work\sport\sport_demo1\img\onlyRuler.jpg', cv2.IMREAD_GRAYSCALE)  # 替换为您的图片路径

# 对图像进行阈值处理
# ret, thresh_img = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)

# 定义结构元素（这里使用3x3的矩形结构元素）
kernel = np.ones((1,2), np.uint8)

# 对图像进行侵蚀操作
# erosion = cv2.erode(img, kernel, iterations=1)

# 对侵蚀后的图像进行膨胀操作
dilation = cv2.dilate(img, kernel, iterations=1)

# 显示结果
cv2.imshow('Original', img)
# cv2.imshow('threshold', thresh_img)
# cv2.imshow('Erosion', erosion)
cv2.imshow('Dilation', dilation)
cv2.imwrite('../img/dilation.jpg',dilation)


cv2.waitKey(0)
cv2.destroyAllWindows()
