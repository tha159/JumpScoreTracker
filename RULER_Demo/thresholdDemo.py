import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('../img/onlyRedArea.jpg', cv2.COLOR_BGR2GRAY)

ret, thresh1 = cv2.threshold(image, 227, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(image, 227, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(image, 227, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(image, 227, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(image, 227, 255, cv2.THRESH_TOZERO_INV)
cv2.namedWindow('thresh2_win')
cv2.createTrackbar('thresh2', 'thresh2_win', 200, 255, lambda x: x)


while True:
    thresh2 = cv2.getTrackbarPos('thresh2', 'thresh2_win')

    ret, thresh = cv2.threshold(image, thresh2, 255, cv2.THRESH_BINARY_INV)
    img_black = np.zeros_like(image)
    cv2.imshow('thresh2_win', thresh)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

