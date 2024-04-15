import cv2
import numpy as np

# 读取图像
image = cv2.imread('../img/ruler.jpg')

# 将图像从 BGR 转换为 HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义红色的HSV范围
lower_red = np.array([0, 20, 40])
upper_red = np.array([255, 100, 150])

# 创建掩码，找到HSV图像中的红色区域
mask = cv2.inRange(hsv_image, lower_red, upper_red)

# 在原始图像中根据掩码找到红色区域
red_area = cv2.bitwise_and(image, image, mask=mask)

red_area_gray = cv2.cvtColor(red_area, cv2.COLOR_BGR2GRAY)
# 查找图像中的轮廓
contours, hierarchy = cv2.findContours(red_area_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 初始化最大轮廓和最大面积
max_contour = None
max_area = -1

# 遍历所有轮廓
for contour in contours:
    # 计算当前轮廓的面积
    area = cv2.contourArea(contour)

    # 如果当前轮廓的面积大于最大面积，则更新最大面积和对应的轮廓
    if area > max_area:
        max_area = area
        max_contour = contour

img_edge = image.copy()
# 在原始图像上绘制轮廓
contour_img = cv2.cvtColor(img_edge, cv2.COLOR_BGR2GRAY)
cv2.drawContours(img_edge, [max_contour], 0, (255, 255, 255), thickness=2)

mask_forContour = np.zeros_like(image)
cv2.drawContours(mask_forContour, [max_contour], 0, (255, 255, 255), thickness=cv2.FILLED)
# 将原始图像与掩码图像进行按位与操作
onlyRedArea = cv2.bitwise_and(image, mask_forContour)



# 显示结果
cv2.imshow('Original Image', image)
cv2.imshow('Red Area', red_area)
cv2.imshow('img_edge Area', img_edge)
cv2.imshow('onlyRedArea ', onlyRedArea)

# 将图像保存到指定路径
# cv2.imwrite('../img/onlyRedArea.jpg', onlyRedArea)

cv2.waitKey(0)
cv2.destroyAllWindows()
