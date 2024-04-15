import cv2
import numpy as np



image = cv2.imread('../img/ruler.jpg')  # 替换为您的图片路径

lower = np.uint8([80, 60, 60])
upper = np.uint8([150, 100, 100])
# lower_ _red 和高Fupper_ red的 部分分别变成0，lower_ _red ~ upper_ red之 间的值变成255,相当于过滤背景
white_mask = cv2.inRange(image, lower, upper)
masked = cv2.bitwise_and(image, image, mask = white_mask)
img_gray = cv2.cvtColor(masked, cv2.COLOR_RGB2GRAY)
edge_img = cv2.Canny(img_gray,300,900)
# 显示结果
cv2.imshow('image', image)
cv2.imshow('white_mask', white_mask)
cv2.imshow('masked', masked)
cv2.imshow('img_gray', img_gray)
cv2.imshow('edge_img', edge_img)


cv2.waitKey(0)
cv2.destroyAllWindows()