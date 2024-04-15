import cv2
import numpy as np

# 定义全局变量
clicked_points = []

def get_hsv(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_pixel = hsv[y, x]
        clicked_points.append((x, y, hsv_pixel))
        print("HSV Value at ({}, {}): {}".format(x, y, hsv_pixel))

# 读取图像
image = cv2.imread('../img/ruler.jpg')

# 创建窗口并设置鼠标回调函数
cv2.namedWindow('image')
cv2.setMouseCallback('image', get_hsv)

print("Click on the image to get HSV values. Press 'q' to quit.")

while True:
    cv2.imshow('image', image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()

print("Clicked HSV Values:")
for point in clicked_points:
    print("Point ({}, {}): {}".format(point[0], point[1], point[2]))
