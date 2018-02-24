import numpy as np
import cv2
import argparse
import glob
import dlib
import math
from time import time as timer
import sys


video_capture = cv2.VideoCapture(0)
objects_svm = []
spotters = []
display_list = []
show_list = []
elapsed = int()
start = timer()


for files in glob.glob("./svmfiles/*.svm"):
    objects_svm.append(files)

for files in glob.glob("./indicator/*.jpg"):
	show_list.append(files)

for svm in objects_svm:
    detector = dlib.fhog_object_detector(svm)
    spotters.append(detector)


while True:
	ret, frame = video_capture.read()

	results = dlib.fhog_object_detector.run_multiple(spotters, frame, 0, 0.0)

	elapsed += 1;
	cv2.putText(frame,('{0:3.3f} FPS'.format(elapsed / (timer() - start))), (1000, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,69,0), 2)



	if len(results[2]) > 0:
		annotation = objects_svm[int(results[2][0])].lstrip('./svmfiles/').replace('.svm', '')
		locationdetector = objects_svm[int(results[2][0])]
		locdetector = dlib.simple_object_detector(locationdetector)
		location = locdetector(frame)
		for coordinates in location: 
			cv2.rectangle(frame,(coordinates.left(), coordinates.top()), (coordinates.right(), coordinates.bottom()),(0,0,0),2)
			cv2.putText(frame, annotation, (coordinates.left() + 5, coordinates.top()-5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)


		if annotation not in display_list:
			display_list.append(annotation)
			item_font = cv2.FONT_HERSHEY_SIMPLEX
			black_screen = frame.copy()

		variability = 50
		cv2.rectangle(black_screen,(0,0),(180,1000), (0,0,0),-1)


		for item in display_list:
			variability +=45
			cv2.putText(frame, item, (20, 0 + variability), item_font, 1.0, (255, 255, 255), 2)
			annotationstring = "./indicator/"+ item + ".jpg"
		
			if annotationstring in show_list:
			    curr_image = cv2.imread(annotationstring)
			    cv2.imshow('Video', curr_image)
			    cv2.imwrite("output.jpg", curr_image)

		cv2.addWeighted(black_screen,0.7,frame,0.7,0,frame)


	display_list = []
	show_list = []
	cv2.namedWindow('Video',cv2.WINDOW_NORMAL)
	imS = cv2.resize(frame,(1080,720))

	cv2.imshow('Video', imS)

	if elapsed % 5 == 0:
		sys.stdout.write('\r')
		sys.stdout.write('{0:3.3f} FPS'.format(elapsed / (timer() - start)))
		sys.stdout.flush()


	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

video_capture.release()
cv2.destroyAllWindows()