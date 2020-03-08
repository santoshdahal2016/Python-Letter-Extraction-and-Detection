import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

video_capture = cv.VideoCapture(0)

while(1):
	ret, img = video_capture.read()
	gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
	corners = cv.goodFeaturesToTrack(gray,25,0.01,10)
	corners = np.int0(corners)
	for i in corners:
	    x,y = i.ravel()
	    cv.circle(img,(x,y),3,255,-1)
	cv.imshow("Good Feature to Track",img)
	cv.waitKey(1)

	if cv.waitKey(5) & 0xFF == ord('q'):
		break

# object_img = cv.imread('/a.png', 0)
# scene_img = cv.imread('/a-scene.jpg', 0)

# gray = cv.cvtColor(object_img,cv.COLOR_BGR2GRAY)
# corners_object = cv.goodFeaturesToTrack(gray,25,0.01,10)

# gray = cv.cvtColor(scene_img,cv.COLOR_BGR2GRAY)
# scene_object = cv.goodFeaturesToTrack(gray,25,0.01,10)

# bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# matches = bf.match(corners_object, scene_object)
# matches = sorted(matches, key=lambda x: x.distance)# draw first 50 matches
# match_img = cv.drawMatches(scene_img, scene_object, object_img, corners_object, matches[:50], None)
# cv.imshow('Matches', match_img)
# cv.waitKey()