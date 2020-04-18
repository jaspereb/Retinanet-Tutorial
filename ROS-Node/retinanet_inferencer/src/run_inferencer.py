#!/usr/bin/env python
# Author: Jasper Brown
# jasperebrown@gmail.com
# 2020

# IMPORTANT: This is a node wrapper for the detector object, the model should be set there

# Retinanet stuff
import keras
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image
import tensorflow as tf

# Generic stuff
import sys
import cv2
import os
import numpy as np
import time

# ROS Stuff
import rospy
from std_msgs.msg import String, Header
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, BoundingBox2D, ObjectHypothesisWithPose
from geometry_msgs.msg import Pose2D
import rospkg
from cv_bridge import CvBridge
from copy import deepcopy


class ROS_Detector(object):
	def __init__(self):
		rospy.init_node('retinanet_inferencer', anonymous=True)
		args = rospy.myargv(argv=sys.argv)
		rospy.loginfo(args)
		sub_topic = '/object_detection_input'
		det_pub_topic = '/object_detection_output'
		img_pub_topic = '/object_detection_output_image'
		self.confidence_cutoff = 0.5
		self.model_path = '/mnt/0FEF1F423FF4C54B/TrainingOutput/KerasRetinanet/snapshots/PlumsInference.h5'

		#Not set by args
		self.color = (255, 255, 255)
		self.thickness = 2

		if len(args) == 3:
			rospy.loginfo("Using command line args for object detector")
			sub_topic = args[1]
			det_pub_topic = args[2]
			img_pub_topic = args[3]
			self.confidence_cutoff = args[4]
			self.model_path = args[5]

		self.detections_pub = rospy.Publisher(det_pub_topic, Detection2DArray, queue_size=10, latch=True)
		self.img_pub = rospy.Publisher(img_pub_topic, Image, queue_size=10, latch=True)
		self.bridge = CvBridge()

		# load label to names mapping for visualization purposes
		self.labels_to_names = {0: 'plum', 1: 'green_plum'}
		rospy.loginfo("The following label names are set in the run_inferencer.py script: {}".format(self.labels_to_names))

		#Setup model
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		self.session = tf.Session(config=config)
		keras.backend.tensorflow_backend.set_session(self.session)
		self.model = models.load_model(self.model_path, backbone_name='resnet50')

		self.img_sub = rospy.Subscriber(sub_topic, Image, self.callback)
		rospy.loginfo("Inferencer node initialised")
		rospy.spin()

	def callback(self, imageMsg):
		image = self.bridge.imgmsg_to_cv2(imageMsg, "bgr8")
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = image[:, :, ::-1].copy()

		# copy to draw on
		draw = image.copy()
		draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

		# Image formatting specific to Retinanet
		image = preprocess_image(image)
		image, scale = resize_image(image)

		# Run the inferencer
		try:
			with self.session.as_default():
				with self.session.graph.as_default():
					boxes, scores, labels = self.model.predict_on_batch(np.expand_dims(image, axis=0))
		except Exception as e:
			rospy.logerr(e)
			rospy.logwarn("WARNING: Has your model been converted to an inference model yet? "
				"see https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model")
			return

		# correct for image scale
		boxes /= scale

		# Construct the detection message
		header = Header(frame_id=imageMsg.header.frame_id)
		detections_message_out = Detection2DArray()
		detections_message_out.header = header
		detections_message_out.detections = []

		# visualize detections
		for box, score, label in zip(boxes[0], scores[0], labels[0]):
			# scores are sorted so we can break
			if score < self.confidence_cutoff:
				break

			# Add boxes and captions
			b = np.array(box).astype(int)
			cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), self.color, self.thickness, cv2.LINE_AA)

			if (label > len(self.labels_to_names)):
				print("WARNING: Got unknown label, using 'detection' instead")
				caption = "Detection {:.3f}".format(score)
			else:
				caption = "{} {:.3f}".format(self.labels_to_names[label], score)

			cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
			cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

			#Construct the output detection message
			bb = BoundingBox2D()
			det = Detection2D(header=header)
			hyp = ObjectHypothesisWithPose()
			center = Pose2D()
			hyp.id = label
			hyp.score = score
			bb.size_x = b[2] - b[0]
			bb.size_y = b[3] - b[1]
			center.x = float(b[2] + b[0])/2
			center.y = float(b[3] + b[1])/2
			bb.center = center
			det.results.append(hyp)
			det.bbox = bb
			detections_message_out.detections.append(det)

		self.detections_pub.publish(detections_message_out)

		# Write out image
		image_message_out = self.bridge.cv2_to_imgmsg(draw, encoding="rgb8")
		self.img_pub.publish(image_message_out)


if __name__ == '__main__':
	try:
		ROS_Detector()
	except rospy.ROSInterruptException:
		rospy.logerr('Could not start object detection node.')
