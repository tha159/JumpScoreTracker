#输入的图像需要是边缘检测后的结果
#minLineLengh(线的最短长度，比这个短的都被忽略)和MaxLineCap (两条直线之间的最大间隔，小于此值，认为是- - 条直线)
#rho距离精度，theta角度精度, threshod超过设定阈值才被检测出线段

import cv2
import numpy as np

cv2.namedWindow('hough')
cv2.createTrackbar('minThreshold', 'hough', 300, 1000, lambda x: x)
cv2.createTrackbar('maxThreshold', 'hough', 900, 1000, lambda x: x)
cv2.createTrackbar('threshold', 'hough', 1, 100, lambda x: x)
cv2.createTrackbar('minLineLength', 'hough', 0, 100, lambda x: x)
cv2.createTrackbar('maxLineGap', 'hough', 1, 100, lambda x: x)

while True:
    img = cv2.imread('../img/erosion.jpg', cv2.IMREAD_COLOR)  # 使用彩色模式读取图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像

    minThreshold = cv2.getTrackbarPos('minThreshold', 'hough')
    maxThreshold = cv2.getTrackbarPos('maxThreshold', 'hough')
    threshold = cv2.getTrackbarPos('threshold', 'hough')
    minLineLength = cv2.getTrackbarPos('minLineLength', 'hough')
    maxLineGap = cv2.getTrackbarPos('maxLineGap', 'hough')


    # kernel = np.ones((2, 2), np.uint8)  # 可根据实际情况调整核的大小
    #
    # # 进行开运算
    # opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    # erosion = cv2.erode(opening, kernel, iterations=1)
    # erosion = opening

    edges = cv2.Canny(img, minThreshold, maxThreshold)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)
    cv2.imshow('erosion', img)
    cv2.imshow('hough', edges)

    if lines is not None:
        print(len(lines))

    else:
        print("No lines detected.")
    cleaned=[]
    real_line = np.zeros_like(gray)
    black = np.zeros_like(gray)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(black, (x1, y1), (x2, y2), (255, 255, 255), 1)
                if abs(x2-x1) <=5 and abs(y1-y2) <=110 and abs(y1-y2) >=40:
                    cleaned.append((x1,y1,x2,y2))
                    cv2.line(real_line, (x1,y1),(x2,y2),(255, 255, 255),1)

        cv2.imshow('lines', black)
        cv2.imshow('real_line', real_line)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
