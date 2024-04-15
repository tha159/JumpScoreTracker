#least squares method

import cv2
import numpy as np

# 读取图像
image = cv2.imread('your_image.jpg')

# 将图像转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def least_squares_fit(lines):
    # 1.取出所有坐标点
    x_coords = np.ravel([[line[0][0],line[0][2]] for line in lines])
    y_coords = np.ravel([[line[0][1], line[0][3]] for line in lines])
    # 2.进行直线拟合.得到多项式系数
    poly = np.polyfit(x_coords, y_coords, deg = 1)
    # 3.根据多项式系数,计算两个直线上的点,用于唯一确定这条直线
    point_min = (np.min(x_coords), np.polyval(poly, np.min(x_coords)))
    point_max = (np.max(x_coords), np.polyval(poly, np.max(x_coords)))
    return np.array([point_min, point_max], dtype=np.int)


# 使用LSD算法检测直线
lsd = cv2.createLineSegmentDetector(0)  # 参数0表示默认LSD算法
lines, _, _, _ = lsd.detect(gray)

# 合并直线的距离阈值
merge_distance_threshold = 20

# 合并检测到的直线
def merge_lines(lines, threshold):
    merged_lines = []

    for line1 in lines:
        x1, y1, x2, y2 = map(int, line1.flatten())
        p1 = np.array([x1, y1])
        p2 = np.array([x2, y2])

        merged = False
        for line2 in merged_lines:
            x3, y3, x4, y4 = map(int, line2.flatten())
            q1 = np.array([x3, y3])
            q2 = np.array([x4, y4])

            if np.linalg.norm(p1 - q1) < threshold or np.linalg.norm(p2 - q2) < threshold:
                merged = True
                break

        if not merged:
            merged_lines.append(line1)

    return merged_lines

# 合并直线
merged_lines = merge_lines(lines, merge_distance_threshold)

# 在原始图像上绘制合并后的直线
for line in merged_lines:
    x1, y1, x2, y2 = map(int, line.flatten())
    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 显示结果
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
