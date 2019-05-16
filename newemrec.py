# import the necessary packages
from __future__ import division
from centroid.centroidtracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

import pandas as pd
import threading
from collections import Counter

from PyQt5.QtWidgets import QApplication
from models import Camera
from views import StartWindow


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default = "deploy.prototxt",
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default="res10_300x300_ssd_iter_140000.caffemodel",
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W) = (None, None)

# parameters for loading data and images
emotion_model_path = './models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
emotion_classifier = load_model(emotion_model_path)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0)
vs.start()
time.sleep(2.0)

detected_objects = {}
face = {}

# starting video streaming

cv2.namedWindow('window_frame', cv2.WINDOW_AUTOSIZE)
#cv2.createTrackbar('just slider', 'window_frame', 0, 10, callback)

###################################################################################################
###################################################################################################
##########################################################
df = pd.DataFrame(columns=['faceID', 'Anger', 'Disgust', 'Fear',\
					'Happiness', 'Neutral', 'Sadness', 'Surprised'])	

# FUNCTION OF RECOGNITION

def recognition(arguments):
	global df
	frame, rects, detected_objects, iteration, objectID = arguments
	bgr_image = frame
	gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
	rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

	faces = rects
	for face_coordinates in faces:

		x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
		gray_face = gray_image[y1:y2, x1:x2]
		try:
			gray_face = cv2.resize(gray_face, emotion_target_size)
		except:
			continue

		gray_face = preprocess_input(gray_face, True)
		gray_face = np.expand_dims(gray_face, 0)
		gray_face = np.expand_dims(gray_face, -1)
		emotion_prediction = emotion_classifier.predict(gray_face)
		
		em_pre = emotion_prediction[0]
		#em_pre = (em_pre-min(em_pre))/(max(em_pre)-min(em_pre))
		#emp_pre = float(em_pre)
		df = df.append({'faceID': objectID, 'Anger': em_pre[0], 'Disgust': em_pre[1], 'Fear': em_pre[2],\
                      'Happiness': em_pre[3], 'Neutral': em_pre[4], 'Sadness': em_pre[5], 'Surprised': em_pre[6]},\
                     ignore_index=True)
		df.to_csv('rawData.csv')
		emotion_probability = np.max(emotion_prediction)
		emotion_label_arg = np.argmax(emotion_prediction)
		emotion_text = emotion_labels[emotion_label_arg]
		emotion_window.append(emotion_text)

		if len(emotion_window) > frame_window:
			emotion_window.pop(0)
		try:
			emotion_mode = mode(emotion_window)
		except:
			continue
		
		detected_objects.setdefault(objectID, []).append(emotion_text)
		
	bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
	cv2.imshow('window_frame', bgr_image)
	

# END OF THE RECOGNITION FUNCTION
iteration = 1
# loop over the frames from the video stream
while True:
	
	# read the next frame from the video stream and resize it
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# if the frame dimensions are None, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construct a blob from the frame, pass it through the network,
	# obtain our output predictions, and initialize the list of
	# bounding box rectangles
	blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),
		(104.0, 177.0, 123.0))
	net.setInput(blob)
	detections = net.forward()
	rects = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# filter out weak detections by ensuring the predicted
		# probability is greater than a minimum threshold
		if detections[0, 0, i, 2] > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object, then update the bounding box rectangles list
			box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
			rects.append(box.astype("int"))

			# draw a bounding box surrounding the object so we can
			# visualize it
			(startX, startY, endX, endY) = box.astype("int")
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 255, 0), 2)

	# update our centroid tracker using the computed set of bounding
	# box rectangles
	objects = ct.update(rects)
	#print('objects.items', objects.items())
	# loop over the tracked objects
	for (objectID, centroid) in objects.items():
		# draw both the ID of the object and the centroid of the
		# object on the output frame
		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
		
		arguments = frame, rects, detected_objects, iteration , objectID
				
		recognition(arguments)
		
##########	
	iteration +=1
###########
	if cv2.waitKey(1) & 0xFF == ord('q'):
			break
			
print ("HERE IS THE LAST LINE")


# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

