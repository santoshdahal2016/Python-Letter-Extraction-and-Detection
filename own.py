import cv2
import numpy as np
import imutils
import pytesseract
import re
# image_src = cv2.imread("a-scene.jpg")
# gray = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
# ret, gray = cv2.threshold(gray, 250,255,0)

# contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# largest_area = sorted(contours, key=cv2.contourArea)[-1]
# mask = np.zeros(image_src.shape, np.uint8)
# cv2.drawContours(mask, [largest_area], 0, (255,255,255,255), -1)
# dst = cv2.bitwise_and(image_src, mask)
# mask = 255 - mask
# roi = cv2.add(dst, mask)

# roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
# ret, gray = cv2.threshold(roi_gray, 250,255,0)
# contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# max_x = 0
# max_y = 0
# min_x = image_src.shape[1]
# min_y = image_src.shape[0]

# for c in contours:
#     if 150 < cv2.contourArea(c) < 100000:
#         x, y, w, h = cv2.boundingRect(c)
#         min_x = min(x, min_x)
#         min_y = min(y, min_y)
#         max_x = max(x+w, max_x)
#         max_y = max(y+h, max_y)

# roi = roi[min_y:max_y, min_x:max_x]
# cv2.imshow("roi", mask)
# cv2.waitKey(0)

def getContours(thresh, orig):
	(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]

	# calculate the area of each contour
	area = np.zeros(len(cnts))
	for i, contour in enumerate(cnts):
		area[i] = cv2.contourArea(contour)
		# print 'area: ', area[i]

	# filter contours by area
	cnts_filt_indx = [i for i,v in enumerate(area) if v > CONTOUR_AREA_THRESH]

	
	# draw contours on original image
	for cont_i in enumerate(cnts_filt_indx):
		cnt = cnts[cont_i[1]]
		cv2.drawContours(orig, [cnt], 0, (0,255,0), 3) 


video_capture = cv2.VideoCapture(0)
cv2.namedWindow("Letter")
cv2.namedWindow("Feed")

while(1):
	ret, image = video_capture.read()
	# image = cv2.imread('a-scene.jpg',1) #read image

	# mser = cv2.MSER_create()
	# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 

	# lower_blue = np.array([0, 5, 50], np.uint8)
	# upper_blue = np.array([179, 50, 255], np.uint8)
	  
	# mask = cv2.inRange(hsv, lower_blue, upper_blue) 
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	ret, bw_img = cv2.threshold(gray,127,255,cv2. THRESH_BINARY)
	edged = cv2.Canny(np.invert(bw_img), 30, 200)


	# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
	# connected = cv2.morphologyEx(bw_img, cv2.MORPH_CLOSE, kernel)

	contours, hierarchy = cv2.findContours(edged,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

	contours = sorted(contours, key=lambda x: cv2.contourArea(x))

	contours = list(filter(lambda cnt: cv2.contourArea(cnt) <= 100000,contours)) 
	contours = list(filter(lambda cnt: cv2.contourArea(cnt) > 6000,contours)) 
	# for idx in range(len(contours)):
	# 	x, y, w, h = cv2.boundingRect(contours[idx])
	# 	cropped_image = image[y:y+h , x:x+w+10]
	# 	cv2.imshow("Game Boy Screen", cropped_image)
	# 	cv2.waitKey(0)
	# print(array_of_texts)
	# regions,_ = mser.detectRegions(bw_img)

	# hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]


	# cv2.polylines(image, hulls, 1, (0, 0, 255), 2)
	print(len(contours))
	# cv2.imshow("Game Boy Screen", image)
	# cv2.waitKey(1)
	img_contours = cv2.drawContours(image, contours, -1, (0,255,0), 3)
	cv2.imshow("Feed", image)
	cv2.waitKey(1)

	if(len(contours) > 0):
		x, y, w, h = cv2.boundingRect(contours[0])
		# cv2.rectangle(image, (x,y), (x+w , y+h), (255,0,0), 2)
		cropped_image = image[y:y+h , x:x+w+10]
		text = pytesseract.image_to_string(cropped_image)
		print(text)
		cv2.imshow("Letter", cropped_image)
		cv2.waitKey(1)

	if cv2.waitKey(5) & 0xFF == ord('q'):
		break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows() 