import numpy as np
import cv2 as cv
import pytesseract


img = cv.imread("test.png")


img = cv.resize(img , (50,50))

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = cv.threshold(gray, 0, 255,cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

cv.imshow("Resized",gray)
cv.waitKey(0)

text = pytesseract.image_to_string(gray)
print(text)
