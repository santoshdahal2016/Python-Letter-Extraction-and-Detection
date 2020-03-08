import cv2
import numpy as np


image = cv2.imread('a-scene.jpg',1) #read image
image = np.average(image , weights=[0 , 1 , 1] ,axis=2)

cv2.imshow("Gray", image)
cv2.waitKey(0)