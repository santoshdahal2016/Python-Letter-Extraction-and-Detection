import cv2 as cv
import numpy as np

img1 = cv.imread('a.png',0)
img2 = cv.imread('test.png',0)
ret, thresh = cv.threshold(img1, 127, 255,0)
ret, thresh2 = cv.threshold(img2, 127, 255,0)


edged = cv.Canny(thresh, 30, 200)
edged2 = cv.Canny(thresh2, 30, 200)

cv.imshow("Letter", edged)
cv.waitKey(0)


cv.imshow("Test", edged2)
cv.waitKey(0)


contours,hierarchy = cv.findContours(edged,2,1)
cnt1 = contours[0]
contours,hierarchy = cv.findContours(edged2,2,1)
cnt2 = contours[0]
ret = cv.matchShapes(cnt1,cnt2,1,0.0)
print( ret )

# 0.382813795706 -> between a and b
# 1.79769313486e+308 -> between a and a


