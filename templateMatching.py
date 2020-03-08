# import the necessary packages
import numpy as np
import imutils
import glob
import cv2


template = cv2.imread("a.png")
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = cv2.Canny(template, 50, 200)
(tH, tW) = template.shape[:2]
cv2.imshow("Template", template)
cv2.waitKey(0)

image = cv2.imread("a-scene.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
found = None
# loop over the scales of the image
for scale in np.linspace(0.2, 1.0, 20)[::-1]:
	# resize the image according to the scale, and keep track
	# of the ratio of the resizing
	resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
	r = gray.shape[1] / float(resized.shape[1])
	# if the resized image is smaller than the template, then break
	# from the loop
	if resized.shape[0] < tH or resized.shape[1] < tW:
		break
	edged = cv2.Canny(resized, 50, 200)
	result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
	(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
	# check to see if the iteration should be visualized
	# draw a bounding box around the detected region
	clone = np.dstack([edged, edged, edged])
	cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
		(maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
	cv2.imshow("Visualize", clone)
	cv2.waitKey(0)
	# if we have found a new maximum correlation value, then update
	# the bookkeeping variable
	if found is None or maxVal > found[0]:
		found = (maxVal, maxLoc, r)
		
(_, maxLoc, r) = found
(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
# draw a bounding box around the detected result and display the image
cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
cv2.imshow("Image", image)
cv2.waitKey(0)