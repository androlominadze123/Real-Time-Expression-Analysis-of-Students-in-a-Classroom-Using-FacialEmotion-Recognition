# -*- coding: utf-8 -*-

import os

import sys

import cv2

import numpy as np

def main():

	output_path = sys.argv[1]

	if os.path.exists(output_path):

		os.system('rm -rf {}'.format(output_path))

	os.system('mkdir {}'.format(output_path))

	label_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
	#data = pd.read_csv('fer2013.csv', delimiter=',')
	data = np.genfromtxt('fer2013.csv',delimiter=',',dtype=None)

	labels = data[1:,0].astype(np.int32)
	image_buffer = data[1:,1]
	images = np.array([np.fromstring(image, np.uint8, sep=' ') for image in image_buffer])
	usage = data[1:,2]
	dataset = zip(labels, images, usage)
	for i, d in enumerate(dataset):
		usage_path = os.path.join(output_path, d[-1])
		label_path = os.path.join(usage_path, label_names[d[0]])
		img = d[1].reshape((48,48))
		img_name = '%08d.jpg' % i
		img_path = os.path.join(label_path, img_name)
		if not os.path.exists(usage_path):
			os.system('mkdir {}'.format(usage_path))
		if not os.path.exists(label_path):
			os.system('mkdir {}'.format(label_path))
		cv2.imwrite(img_path, img)
		print ('Write {}'.format(img_path))
if __name__ == '__main__':
    main()