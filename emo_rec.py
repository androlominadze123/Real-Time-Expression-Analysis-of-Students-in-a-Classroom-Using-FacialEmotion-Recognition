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

from decimal import Decimal

import os
import glob


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default = "deploy.prototxt",
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default="res10_300x300_ssd_iter_140000.caffemodel",
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.7,
	help="minimum probability to filter weak detections")
ap.add_argument("-d", "--dataset", type = str,
	help="choose between FER213 and KDEF datasets")
args = vars(ap.parse_args())

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W) = (None, None)

# parameters for loading data and images
emotion_model_path = './models/emotion_model.hdf5'
#emotion_labels = get_labels('fer2013')
#emotion_labels = get_labels('KDEF')
emotion_labels = args["dataset"]
print("emotion labels", emotion_labels)
# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
emotion_classifier = load_model(emotion_model_path)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]
print('size',emotion_target_size)
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

# For averaging create a list
emotion_list = []

# starting video streaming

cv2.namedWindow('window_frame', cv2.WINDOW_AUTOSIZE)



df = pd.DataFrame(columns=['faceID','angry', 'disgust', 'fear', 'happy', 'sad',
                'surprise', 'neutral', 'max_emotion'])

meanDataFrame = pd.DataFrame(columns=['faceID', 'angry', 'disgust', 'fear', 'happy', 'sad',
                'surprise', 'neutral'])


def sorting():
	dataToSort = pd.read_csv("rawData.csv", index_col=0)
	sorted = dataToSort.sort_values(by=['faceID'])
	sorted.to_csv('sorted.csv')
def averaging(iteration):
	#global meanDataFrame

	data = pd.read_csv("rawData.csv", index_col=0)
	mean = np.mean(data.drop(['faceID', 'max_emotion'], axis = 1), axis = 0)
	max  = np.argmax(mean)
	

	mean.to_csv('averageData.csv')



# FUNCTION OF RECOGNITION
def recognition(arguments):
	
	global df
	frame, rects, detected_objects, iteration, objectID, centr = arguments
	
	bgr_image = frame
	gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
	rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
	
	face_coordinates = rects
		
	x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
	
	#crop faces from image
	gray_face = gray_image[y1:y2, x1:x2]
	
	#resize the face image to desired size
	gray_face = cv2.resize(gray_face, (emotion_target_size))
	
	resized = gray_face
	#cv2.imwrite("dataset/User." + str(objectID) + '.' + str(iteration) + ".jpg", resized)
	
	# emotion prediction
	gray_face = preprocess_input(gray_face, True)
	gray_face = np.expand_dims(gray_face, 0)
	gray_face = np.expand_dims(gray_face, -1)
	emotion_prediction = emotion_classifier.predict(gray_face)
	emotion_probability = np.max(emotion_prediction)
	emotion_label_arg = np.argmax(emotion_prediction)
	
	emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

	bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
	
	print('centroid:', centr)
	######################################## EMOTION PUT ON SCREEN
	emo = emotion_prediction
	emo_txt = (f"angry: {str(round(emo[0][0],4))}\n"
					f"disgust: {str(round(emo[0][1],4))}\n"
					f"fear: {str(round(emo[0][2],4))}\n"
					f"happy: {str(round(emo[0][3],4))}\n"
					f"sad: {str(round(emo[0][4],4))}\n"
					f"surprise: {str(round(emo[0][5],4))}\n"
					f"neutral: {str(round(emo[0][6],4))}\n")
			
	for i, line in enumerate(emo_txt.split('\n')):
		cv2.putText(bgr_image, line, (centr[0] + 35, centr[1] + i * 15),
			cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0, 0, 255), 1)

	
	###############################################
	cv2.imshow('window_frame', bgr_image)
	em_pre = emotion_prediction[0]
	
	
	# save the predicted probabilities for each face	
	df = df.append({'faceID': objectID, 'angry': em_pre[0], 'disgust': em_pre[1], 'fear': em_pre[2],\
                      'happy': em_pre[3], 'sad': em_pre[4], 'surprise': em_pre[5],\
					  'neutral': em_pre[6], 'max_emotion': emotions[emotion_label_arg]}, ignore_index=True)

	df.to_csv('rawData.csv')

	# sort the predictions by face_ID
	sorted = sorting()
	# average all the emotions to get overall mood
	average_emotion = averaging(iteration)	
	
# END OF THE RECOGNITION FUNCTION
	
	
iteration = 1
# loop over the frames from the video stream
while True:
	print("frame #\n:",iteration)

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
	
	print('objects:', objects)
	print('rects:', rects)
	#print('objects.items', objects.items())
	# loop over the tracked objects
	for (objectID, centroid) in objects.items():
		
		# draw both the ID of the object and the centroid of the
		# object on the output frame
		text = "ID {}".format(objectID)
	
		try:			
			arguments = frame, rects[objectID], detected_objects, iteration , objectID, centroid
		except:
			continue
			
		recognition(arguments)
		
		
				
	iteration +=1

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

print ("HERE IS THE LAST LINE")

cv2.destroyAllWindows()
vs.stop()


