
# -*- coding: utf-8 -*-
#
#  ObjectRecognitionTFVideo.py
#  Description:
#        Use ModelNetV2-SSD model to detect objects on image or video

# import packages
import sys
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import tensorflow as tf


# load model from path
model= tf.saved_model.load("./pretrained_models/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model")
print("model loaded")

#load class names
def read_label_map(label_map_path):

    item_id = None
    item_name = None
    items = {}
    
    with open(label_map_path, "r") as file:
        for line in file:
            line.replace(" ", "")
            if line == "item{":
                pass
            elif line == "}":
                pass
            elif "id" in line:
                item_id = int(line.split(":", 1)[1].strip())
            elif "display_name" in line: #elif "name" in line:
                item_name = line.split(":", 1)[1].replace("'", "").strip()

            if item_id is not None and item_name is not None:
                #items[item_name] = item_id
                items[item_id] = item_name
                item_id = None
                item_name = None

    return items

class_names=read_label_map("./pretrained_models/ssd_mobilenet_v2_320x320_coco17_tpu-8/mscoco_label_map.pbtxt")
class_colors = np.random.uniform(0, 255, size=(len(class_names), 3))


if __name__ == '__main__':

    # Camera initialisation
    print("Start Camera...")
    vs = VideoStream(src=0, resolution=(1600, 1200)).start() #from usb cam
    #vs = VideoStream(usePiCamera=True, resolution=(1600, 1200)).start() #from RPi cam
    #vc = cv2.VideoCapture('./data/Splash - 23011.mp4') #from video file

    time.sleep(2.0)
    fps = FPS().start()
    
    #Main loop
    while True:
        #get image
        img = vs.read() # Get video stream
        #img= cv2.imread('./data/two-boats.jpg') #from image file
        #ret, img=vc.read() #from video or ip cam

                #process image
        img = imutils.resize(img, width=800) #max width 800 pixels 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # get height and width of image
        h, w, _ = img.shape

        input_tensor = np.expand_dims(img, 0)

        # predict from model
        resp = model(input_tensor)

        # iterate over boxes, class_index and score list
        for boxes, classes, scores in zip(resp['detection_boxes'].numpy(), resp['detection_classes'], resp['detection_scores'].numpy()):
            for box, cls, score in zip(boxes, classes, scores): # iterate over sub values in list
                if score > 0.61: # we are using only detection with confidence of over 0.6
                    ymin = int(box[0] * h)
                    xmin = int(box[1] * w)
                    ymax = int(box[2] * h)
                    xmax = int(box[3] * w)
                                    
                    # write classname for bounding box
                    cls=int(cls) #convert tensor to index
                    label = "{}: {:.2f}%".format(class_names[cls],score * 100)
                    cv2.putText(img, label, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 1, class_colors[cls], 1)
                    
                    #display position
                    X= (xmax+xmin)/2
                    Y= (ymax+ymin)/2
                    poslbl= "X: ({},{})".format(X,Y)
                    cv2.circle(img, (int(X)-15, int(Y)), 1, class_colors[cls], 2)    
                    cv2.putText(img, poslbl, (int(X), int(Y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_colors[cls], 2)
                    
                    # draw on image
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), class_colors[cls], 4)
                
                
        # Show video frame
        cv2.imshow("Frame", img)
        key = cv2.waitKey(1) & 0xFF

        # Exit script with letter q
        if key == ord("q"):
            break

        # FPS update 
        fps.update()

    # Stops fps and display info
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    cv2.destroyAllWindows()
    vs.stop()
    #vc.release()