from detector import ObjectDetector
import numpy as np
import cv2
import argparse
import glob
import dlib


# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)
objects_svm = []
detectors = []
display_list = []
show_list = []

for files in glob.glob("./svmfiles/*.svm"):
    objects_svm.append(files)

for files in glob.glob("./indicator/*.jpg"):
	show_list.append(files)

for svm in objects_svm:
    detector = dlib.fhog_object_detector(svm)
    detectors.append(detector)

ap = argparse.ArgumentParser()
#ap.add_argument("-d","--detector",required=True,help="path to trained detector to load...")
#ap.add_argument("-a","--annotate",default=None,help="text to annotate...")
args = vars(ap.parse_args())


while True:
    # Grab a single frame of video
	ret, frame = video_capture.read()
	#image = video_capture.read()[1]
	results = dlib.fhog_object_detector.run_multiple(detectors, frame, upsample_num_times=0, adjust_threshold=0.0)

	if len(results[2]) > 0:
		annotation = objects_svm[int(results[2][0])].lstrip('./svmfiles/').replace('.svm', '')
		if annotation not in display_list:
			display_list.append(annotation)
		#print("item found %s" % annotation)	
		#detector.detect(frame,annotation)
		item_font = cv2.FONT_HERSHEY_SIMPLEX
		overlay = frame.copy()

		variability = 50
		cv2.rectangle(overlay,(0,0),(180,1000), (0,0,0),-1)


		for item in display_list:
			variability +=45
			cv2.putText(frame, item, (50, 0 + variability), item_font, 1.0, (255, 255, 255), 2)
			annotationstring = "./indicator/"+ item + ".jpg"
			if annotationstring in show_list:
				curr_image = cv2.imread(annotationstring)
				cv2.imshow('Video', curr_image)

		opacity = 0.7
		cv2.addWeighted(overlay,opacity,frame,opacity,0,frame)


	display_list = []
	show_list = []
	cv2.namedWindow('Video',cv2.WINDOW_NORMAL)
	imS = cv2.resize(frame,(1080,720))

   		# Display the resulting image
	cv2.imshow('Video', imS)
		
    # Hit 'q' on the keyboard to quit!
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Release handle to the webcam

video_capture.release()
cv2.destroyAllWindows()

