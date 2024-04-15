import cv2
import numpy as np

# 读取图像
image = cv2.imread(r'D:\user\work\sport\sport_demo1\img\onlyRuler.jpg')

# 将图像转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 对灰度图像进行二值化处理
# ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# 查找图像中的轮廓
contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(len(contours))
# 定义面积阈值范围
min_area = 20  # 最小面积
max_area = 200  # 最大面积

# 新建一个全黑图像，用于绘制保留的轮廓
result = np.zeros_like(image)

# 在新图像上绘制符合面积范围的轮廓
for contour in contours:
    area = cv2.contourArea(contour)
    cv2.drawContours(result, [contour], -1, (0, 255, 255), cv2.FILLED)

# 将保留的轮廓作为掩码，从原始图像中提取对应区域
result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
masked_image = cv2.bitwise_and(image, image, mask=result)

# 显示结果
cv2.imshow('Result', masked_image)
# cv2.imwrite('../img/onlyRuler.jpg',masked_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
