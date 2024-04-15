import cv2
import numpy as np

# 读取图片
img = cv2.imread(r'D:\user\work\sport\sport_demo1\img\ruler.jpg', cv2.IMREAD_GRAYSCALE)  # 替换为您的图片路径
img0 = cv2.imread(r'D:\user\work\sport\sport_demo1\img\ruler.jpg')  # 替换为您的图片路径
img01 = img0


ret, thresh_img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('thresh_img', thresh_img)

# median_img = cv2.medianBlur(thresh_img,3)
median_img = cv2.GaussianBlur(thresh_img,(3,3),1)
# median_img = cv2.boxFilter(thresh_img,-1,(3,3),normalize=False)

cv2.imshow('median', median_img)


# cv2.namedWindow('median_win')
# cv2.createTrackbar('median_value', 'median_win', 3, 20, lambda x: x)
#
#
# while True:
#     median_value = cv2.getTrackbarPos('median_value', 'median_win')
#
#     median_img = cv2.medianBlur(thresh_img,median_value)
#     img_black = np.zeros_like(thresh_img)
#     cv2.imshow('median_win', median_img)
#
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break

cv2.waitKey(0)
cv2.destroyAllWindows()