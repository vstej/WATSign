import math
import sys
import numpy as np
import cv2
import os

cap = cv2.VideoCapture(0)

lower_red = np.array([0,150,100])
upper_red = np.array([255,255,180])

lower_red1 = np.array([0, 70, 0])
upper_red1 = np.array([10, 255, 255])


while(True):

	ret, image = cap.read()

	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


	filter_mask = cv2.inRange(hsv, lower_red, upper_red)
	second_filter_mask = cv2.inRange(hsv, lower_red1, upper_red1)
	final_mask = filter_mask + second_filter_mask
	black_and_white_image = cv2.bitwise_and(image, hsv, final_mask)
	gaussian = cv2.GaussianBlur(black_and_white_image, (3, 3), 0)
	canny = cv2.Canny(gaussian,10,250)
	closed = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))


	_, contours, _ = cv2.findContours(closed,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	for countour in contours:

		approximation = cv2.arcLength(countour,True)
		closest_shape = 0.03 * approximation
		approx = cv2.approxPolyDP(countour, closest_shape , True)
		if len(approx) == 8 and cv2.contourArea(countour) > 30000:
			x,y,w,h = cv2.boundingRect(countour)
			cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),10,8)


	cv2.imshow('Result', image)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()