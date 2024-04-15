#输入的图像需要是边缘检测后的结果
#minLineLengh(线的最短长度，比这个短的都被忽略)和MaxLineCap (两条直线之间的最大间隔，小于此值，认为是- - 条直线)
#rho距离精度，theta角度精度, threshod超过设定阈值才被检测出线段

import cv2
import numpy as np

cv2.namedWindow('hough')

cv2.createTrackbar('threshold', 'hough', 10, 100, lambda x: x)
cv2.createTrackbar('minLineLength', 'hough', 3, 100, lambda x: x)
cv2.createTrackbar('maxLineGap', 'hough', 1, 100, lambda x: x)

while True:
    img = cv2.imread('../img/lsd.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh_img = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(thresh_img, 50, 150, apertureSize=3)
    threshold = cv2.getTrackbarPos('threshold', 'hough')
    minLineLength = cv2.getTrackbarPos('minLineLength', 'hough')
    maxLineGap = cv2.getTrackbarPos('maxLineGap', 'hough')
    show = img.copy()
    smoothed = cv2.GaussianBlur(gray, (5, 5), 0)
    lines = cv2.HoughLinesP(smoothed, 1, np.pi / 180, threshold=threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)

        cv2.imshow('lines', img)
        cv2.imshow('thresh_img', thresh_img)
        cv2.imshow('edges', edges)
        cv2.imshow('hough', show)


    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
