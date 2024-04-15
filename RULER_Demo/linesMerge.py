import cv2
import numpy as np

def compute_slope(line):
    x1, y1, x2, y2 = line
    if x2 - x1 == 0:
        return np.inf
    return (y2 - y1) / (x2 - x1)

def distance_between_points(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def are_segments_adjacent(seg1, seg2, threshold_distance):
    dist_start_start = distance_between_points(seg1[0][:2], seg2[0][:2])
    dist_start_end = distance_between_points(seg1[0][:2], seg2[0][2:])
    dist_end_start = distance_between_points(seg1[0][2:], seg2[0][:2])
    dist_end_end = distance_between_points(seg1[0][2:], seg2[0][2:])

    return (dist_start_start < threshold_distance or dist_start_end < threshold_distance or
            dist_end_start < threshold_distance or dist_end_end < threshold_distance)

def merge_adjacent_lines(lines, threshold_distance, threshold_slope_change):
    merged_lines = []
    merged_line = lines[0]
    for line in lines[1:]:
        prev_slope = compute_slope(merged_line[0])
        curr_slope = compute_slope(line[0])
        if abs(curr_slope - prev_slope) > threshold_slope_change or not are_segments_adjacent(merged_line, line, threshold_distance):
            merged_lines.append(merged_line)
            merged_line = line
        else:
            merged_line[0][2:] = line[0][2:]
    merged_lines.append(merged_line)
    return merged_lines



# 读取图像
image = cv2.imread(r'D:\user\work\sport\sport_demo1\img\onlyRuler.jpg')

# 将图像转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
showLSD = np.zeros_like(gray)

# 使用LSD算法检测直线
lsd = cv2.createLineSegmentDetector(0)  # 参数0表示默认LSD算法
lines, _, _, _ = lsd.detect(gray)

# 在原始图像上绘制LSD检测到的直线
for line in lines:
    x1, y1, x2, y2 = map(int, line.flatten())
    cv2.line(showLSD, (x1, y1), (x2, y2), (255, 255, 255), 1)
print(len(lines))


# 合并相邻的线段
threshold_distance = 4  # 定义相邻线段的距离阈值
threshold_slope_change =2 # 定义斜率变化阈值
merged_lines = merge_adjacent_lines(lines, threshold_distance, threshold_slope_change)

print(len(merged_lines))
showMerge = np.zeros_like(gray)
# 在原始图像上绘制合并后的线段
for line in merged_lines:
    x1, y1, x2, y2 = map(int, line[0])
    cv2.line(showMerge, (x1, y1), (x2, y2), (255, 255, 255), 1)

# 显示结果
cv2.imshow('showLSD', showLSD)
cv2.imshow('merged_lines',showMerge)
# cv2.imwrite('../img/lsd.jpg',showLSD)


cv2.waitKey(0)
cv2.destroyAllWindows()
