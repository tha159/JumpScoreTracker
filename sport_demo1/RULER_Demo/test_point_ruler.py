
import cv2
import numpy as np

def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        # Check if point is on the edge of polygon
        if ((p1y < y <= p2y) or (p2y < y <= p1y)) and (x >= min(p1x, p2x)):
            if p1y == p2y and x != max(p1x, p2x):
                inside = not inside
        p1x, p1y = p2x, p2y
    return inside



long_real_ruler_lines = [[[74, 730], [455, 414]], [[370, 728], [517, 444]], [[397, 712], [516, 459]], [[406, 731], [530, 443]], [[433, 711], [527, 463]], [[444, 731], [540, 446]], [[467, 715], [541, 462]], [[480, 735], [553, 444]], [[502, 716], [555, 463]], [[517, 738], [564, 444]], [[539, 718], [568, 464]], [[555, 737], [576, 447]], [[573, 720], [582, 464]], [[593, 740], [587, 446]], [[610, 718], [594, 467]], [[630, 740], [599, 446]], [[646, 723], [607, 464]], [[667, 741], [611, 447]], [[681, 721], [621, 467]], [[705, 744], [622, 444]], [[716, 722], [635, 465]], [[742, 743], [634, 446]], [[751, 721], [648, 466]], [[777, 741], [646, 446]], [[786, 721], [661, 466]], [[814, 741], [658, 447]], [[820, 721], [674, 465]], [[849, 740], [670, 448]], [[854, 721], [687, 464]], [[883, 738], [682, 449]], [[886, 718], [701, 467]], [[917, 736], [693, 450]], [[918, 716], [714, 468]], [[951, 735], [705, 450]], [[950, 715], [727, 469]], [[981, 730], [719, 454]], [[978, 711], [743, 471]], [[1010, 725], [733, 456]], [[1009, 710], [754, 470]], [[1041, 724], [744, 455]], [[1033, 704], [772, 474]], [[1070, 722], [757, 456]], [[1063, 703], [784, 473]], [[1096, 717], [772, 459]], [[1090, 700], [797, 474]], [[1124, 715], [784, 458]]]
# x = 774
# y = 560
x = 769
y = 581
image = cv2.imread(r'D:\user\work\sport\sport_demo1\img\2findRuler.jpg')

imgHeight = image.shape[0]
imgWidth = image.shape[1]

for i in range(1, len(long_real_ruler_lines) - 1):
    # 如果点在两个刻度线构成的区域内
    p1 = np.array(long_real_ruler_lines[i][0])
    p2 = np.array(long_real_ruler_lines[i][1])
    p3 = np.array(long_real_ruler_lines[i + 1][0])
    p4 = np.array(long_real_ruler_lines[i + 1][1])
    is_land_point_in_this_gap = point_in_polygon([x, y], [p1, p2, p3, p4])
    if is_land_point_in_this_gap == True:
        print(666)
    # print(is_land_point_in_this_gap)0
        # 绘制四条线段
    color = (255,255,0)
    thickness = 1
    cv2.line(image, p1, p2, color, thickness)
    cv2.line(image, p2, p4, color, thickness)
    cv2.line(image, p3, p4, color, thickness)
    cv2.line(image, p3, p1, color, thickness)
    # cv2.polylines(image, [p1, p2, p3, p4], isClosed=True, color=(255, 0, 255), thickness=5)
    # cv2.addWeighted(image, 0.5, image, 0.5, 00, image)
    cv2.circle(image, (x, y), 2, (255, 255, 255), cv2.FILLED)
    # cv2.circle(image, (769, 581), 5, (255, 255, 255), cv2.FILLED)
    cv2.imshow('image', image)
    cv2.waitKey(0)

