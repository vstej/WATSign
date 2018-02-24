import numpy as np
import cv2
import sys
import dlib 
from imutils.paths import list_images

annotations = []
image_annotations = []
detector_name = "./svmfiles/" + (sys.argv[1].lstrip('./training_data/').replace("/","")) + ".svm"


for training_image in list_images(sys.argv[1]):

    image = cv2.imread(training_image)
    bounding_box = cv2.selectROI(image, False)
    annotations.append([dlib.rectangle(left = int(bounding_box[0]), top = int(bounding_box[1]), right = int(bounding_box[0]+bounding_box[2]), bottom = int(bounding_box[1]+bounding_box[3]))])
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image_annotations.append(image)


detector = dlib.train_simple_object_detector(image_annotations,annotations,dlib.simple_object_detector_training_options())
detector.save(detector_name)

resulting_win = dlib.image_window()
resulting_win.set_image(detector)
dlib.hit_enter_to_continue()