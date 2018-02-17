import numpy as np
import cv2
import argparse
import glob
import dlib
import math


# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture('boston.mp4')
objects_svm = []
detectors = []
display_list = []
show_list = []
str_distance = ""


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

class ObjectDetector(object):
    def __init__(self,options=None,loadPath=None):
        #create detector options
        self.options = options
        if self.options is None:
            self.options = dlib.simple_object_detector_training_options()

        #load the trained detector (for testing)
        if loadPath is not None:
            self._detector = dlib.simple_object_detector(loadPath)

    def _prepare_annotations(self,annotations):
        annots = []
        for (x,y,xb,yb) in annotations:
            annots.append([dlib.rectangle(left=int(x),top=int(y),right=int(xb),bottom=int(yb))])
        return annots

    def _prepare_images(self,imagePaths):
        images = []
        for imPath in imagePaths:
            image = cv2.imread(imPath)
            #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            images.append(image)
        return images

    def fit(self, imagePaths, annotations, visualize=False, savePath=None):
        annotations = self._prepare_annotations(annotations)
        images = self._prepare_images(imagePaths)
        self._detector = dlib.train_simple_object_detector(images, annotations, self.options)

        #visualize HOG
        if visualize:
            win = dlib.image_window()
            win.set_image(self._detector)
            dlib.hit_enter_to_continue()

        #save detector to disk
        if savePath is not None:
            self._detector.save(savePath)

        return self

    def predict(self,image):
        boxes = self._detector(image)
        preds = []
        for box in boxes:
            (x,y,xb,yb) = [box.left(),box.top(),box.right(),box.bottom()]
            preds.append((x,y,xb,yb))
        return preds

    def detect(self,image,annotate=None):
        preds = self.predict(image)
        
        for (x,y,xb,yb) in preds:
            distancei = (2*3.14 * 180)/((xb-x)+(yb-y)*360)*1000 + 9
            distance = math.floor(distancei/2)

            #draw and annotate on image
            cv2.rectangle(image,(x,y),(xb,yb),(0,0,255),2)
            str_distance = str(distance) + ' Inch'
            cv2.putText(frame,'Distance = ' + str(distance) + ' Inch', (x+40,y-40),cv2.FONT_HERSHEY_SIMPLEX,1.0,(128,255,0),2)

            if annotate is not None and type(annotate)==str:
                cv2.putText(image,annotate,(x+5,y-5),cv2.FONT_HERSHEY_SIMPLEX,1.0,(128,255,0),2)

counter = 0;
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    counter+=1
    #image = video_capture.read()[1]
    if counter %60 == 0 : 
        results = dlib.fhog_object_detector.run_multiple(detectors, frame, upsample_num_times=0, adjust_threshold=0.0)
    #detector = ObjectDetector(loadPath = objects_svm[int(results[2][0])])



        if len(results[2]) > 0:
            annotation = objects_svm[int(results[2][0])].lstrip('./svmfiles/').replace('.svm', '')
            locationdetector = objects_svm[int(results[2][0])]
            locdetector = ObjectDetector(loadPath=locationdetector)
            locdetector.detect(frame,annotate=annotation)

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
                    cv2.waitKey(0) 

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



