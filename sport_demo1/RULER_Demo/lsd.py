import random
import cv2
import numpy as np

# 定义函数来计算线段的长度
def line_length(line):
    x1, y1, x2, y2 = line[0]
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# 从原始的线段列表中移除长度低于阈值的线段
def filter_short_lines(lines, threshold):
    filtered_lines = []
    lengths = [line_length(line) for line in lines]
    mean_length = np.mean(lengths)
    std_length = np.std(lengths)

    for line, length in zip(lines, lengths):
        if length >= mean_length - threshold * std_length:
            filtered_lines.append(line)

    return filtered_lines
def group_lines(lsd_lines):
    # 根据中点的 x 位置对线段进行排序
    sorted_lines = sorted(lsd_lines, key=lambda line: (line[0][0] + line[0][2]) / 2)

    # 每两个划分为一组
    groups = [sorted_lines[i:i+2] for i in range(0, len(sorted_lines), 2)]

    # 把划分完的小组放进一个大组里
    big_group = []
    for group in groups:
        big_group.append(group)

    return big_group

def extract_middle_lines(big_group):
    middle_lines = []
    for group in big_group:
        # Extracting endpoints of the lines in the group
        x1_1, y1_1, x2_1, y2_1 = group[0][0]
        x1_2, y1_2, x2_2, y2_2 = group[1][0]
        # 计算第一条线段的最上面和最下面的顶点
        if y1_1 > y2_1:
            top_1 = (x1_1, y1_1)
            bottom_1 = (x2_1, y2_1)
        else:
            top_1 = (x2_1, y2_1)
            bottom_1 = (x1_1, y1_1)

        # 计算第二条线段的最上面和最下面的顶点
        if y1_2 > y2_2:
            top_2 = (x1_2, y1_2)
            bottom_2 = (x2_2, y2_2)
        else:
            top_2 = (x2_2, y2_2)
            bottom_2 = (x1_2, y1_2)

        # 计算顶点对的中点坐标
        midpoint_1 = ((top_1[0] + top_2[0]) / 2, (top_1[1] + top_2[1]) / 2)
        midpoint_2 = ((bottom_1[0] + bottom_2[0]) / 2, (bottom_1[1] + bottom_2[1]) / 2)

        # Creating a new line segment with the midpoint as both endpoints
        middle_line = np.array([[[midpoint_1[0], midpoint_1[1],midpoint_2[0], midpoint_2[1]]]], dtype=np.float32)
        middle_lines.append(middle_line)

    return middle_lines

# 读取图像
image = cv2.imread(r'D:\user\work\sport\sport_demo1\img\dilation.jpg')
image0 = cv2.imread(r'D:\user\work\sport\sport_demo1\img\onlyRedArea.jpg')


# 将图像转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
showLSD = image.copy()  # 创建一个与原始图像大小相同的副本，用于绘制检测到的直线
showMiddle = np.zeros_like(image)

# 使用LSD算法检测直线
lsd = cv2.createLineSegmentDetector(0)  # 参数0表示默认LSD算法
lsd_lines, _, _, _ = lsd.detect(gray)

threshold = 1.5 # 您可以调整阈值以达到所需的过滤效果
filtered_lines = filter_short_lines(lsd_lines, threshold)
print(f'filtered_lines:{len(filtered_lines)}')

if len(filtered_lines) == 92:
    big_group = group_lines(filtered_lines)
    print(f'group_lines:{len(big_group)}')
    middle_lines = extract_middle_lines(big_group)
    print(f'middle_lines:{len(middle_lines)}')



# 在图像上绘制检测到的直线
for idx, lines in enumerate(middle_lines):
    for line in lines:
        x1, y1, x2, y2 = map(int, line[0])
        cv2.line(image0, (x1, y1), (x2, y2),(0,255,255), 1)  # 绘制直线

# 显示结果
cv2.imshow('showMiddle', image0)
cv2.waitKey(0)
cv2.destroyAllWindows()
