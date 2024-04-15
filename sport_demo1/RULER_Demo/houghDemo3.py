import cv2 as cv
import numpy as np

img = cv.imread('../img/lsd.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray,50,150,apertureSize=3)

show = img.copy()
lines = cv.HoughLinesP(edges,1,np.pi/180,1,minLineLength=60,maxLineGap=7)

for line in lines:
    x1,y1,x2,y2 = line[0]
    cv.line(img,(x1,y1),(x2,y2),(255,0,0),5)

cv.imshow("img",show)
cv.imshow("show",img)

cv.waitKey()
cv.destroyAllWindows()
