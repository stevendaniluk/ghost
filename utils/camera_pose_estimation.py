#!/usr/bin/env python
from __future__ import print_function

import rospy
from sensor_msgs.msg import Image, CameraInfo
import cv2
import cv_bridge
import numpy as np
import argparse
import os
import math

__usage__ = """
camera_pose_estimation.py estimates the pose of the camera relative to 
a calibration checkerboard. The intended use is for with a calibration
board laying on the ground in the cameras FOV. The estimation will return
the height and pitch of the camera.
	 
Note that the results are relative to the surface of the calibration board
(i.e. if the board has any thickness it will have to be added to the height).
	 
Arguments include the board size (rows x columns), square size, and camera
topic to subscribe to.
"""

###########################################
class estimator:
	
	###########################################

	def __init__(self, topic, size, square):
		# Arguments
		self.size = size
		self.square = square
		
		# Camera info properties
		self.have_info = False
		self.K = np.zeros((3,3))
		self.D = np.zeros((1,5))
		
		# Flags for when to track corners and perform estimate (set by mouse events)
		self.estimate_flag = False
		self.corners_flag = False
		
		# Image sub
		self.image_sub = rospy.Subscriber(topic, Image, self.imgCallback)
		self.bridge = cv_bridge.CvBridge()
		
		# Camera info sub
		# (image_transport isn't supported for Python, so the CameraInfo message needs
		# to be handled separately, which is ok since only one message is needed)
		info_topic = topic.replace(os.path.basename(topic), "camera_info")
		self.cam_info_sub = rospy.Subscriber(info_topic, CameraInfo, self.camInfoCallback)
				
		cv2.namedWindow("Camera Pose Estimation")
		cv2.setMouseCallback("Camera Pose Estimation", self.captureCallback)
		
		print("Waiting for camera topic...")
			
	###########################################
	
	def captureCallback(self, event, x, y, flags, param):
		if event == cv2.EVENT_LBUTTONDOWN:
			if not self.corners_flag:
				self.corners_flag = True
				print("Click on image again to estimate pose.")
			else:
				self.estimate_flag = True
	
	###########################################
	
	def camInfoCallback(self, info):
		self.K = np.reshape(np.array(info.K), (3, 3))
		self.D = np.array(info.D)
		if np.sum(self.K) != 0 and np.sum(self.D) != 0:
			self.have_info = True
			self.cam_info_sub.unregister()
			print("Click on image to capture corners.")
				
	###########################################

	def imgCallback(self, img):
		if self.have_info:
			try:
				# Get one image with valid checkerboard points
				cv_image = self.bridge.imgmsg_to_cv2(img)
				
				if self.corners_flag:
					valid, corners = cv2.findChessboardCorners(cv_image, self.size)
					if valid:
						# Draw corners on the image and display
						corners = np.squeeze(corners)
						for i in range(len(corners)):
							cv2.circle(cv_image, tuple([corners[i][0], corners[i][1]]), 5, 255, thickness=2)
						
						# Wait for signal to form estimate
						if self.estimate_flag:
							self.estimatePose(cv_image, corners)					
								
				cv2.imshow("Camera Pose Estimation", cv_image)
				cv2.waitKey(3)
				
			except cv_bridge.CvBridgeError as e:
				print(e)
	
	###########################################
			
	def estimatePose(self, img, corners):
		# Corners start from the top right corner, move downwards to the bottom
		# row, then starts at the top of the column to the left.
		# Points are in the form [column, row]
		
		# Form the world points with X pointing forward, and Y pointing left
		pts_3d = np.array([]).reshape(0, 3)
		for n in range(self.size[1]):
			y = n*self.square
			for m in range(self.size[0]):
				x = -m*self.square
				pts_3d = np.vstack([pts_3d, [x, y, 0]])
		
		# Get checkerboard pose (unsubscribe once there is a valid estimate)
		valid, r_vec, t_vec = cv2.solvePnP(pts_3d, corners, self.K, self.D)
		
		if valid:
			self.image_sub.unregister()
		else:
			return
		
		# Form the inverse of the extrinsic matric (i.e. transform of camera wrt world)
		R_c = cv2.Rodrigues(r_vec)[0].T
		t_c = -np.matmul(R_c, t_vec)
		T_c = np.hstack([R_c, t_c])
		alpha, beta, gamma = cv2.decomposeProjectionMatrix(T_c)[-1]
		
		# Convert euler angles to roll-pitch-yaw of a camera with X forward and Y left
		roll = beta
		pitch = -alpha - 90
		yaw = gamma + 90
		
		height = t_c[2]
				
		# Output the results
		print("\n--- Pose Estimation Results ---")
		print("     Height: %9.5f m" % height)
		print("     Roll:   %+9.5f deg" % roll)
		print("     Pitch:  %+9.5f deg" % pitch)
		print("     Yaw:    %+9.5f deg\n" % yaw)
		
		# Exit when done
		cv2.destroyAllWindows()
		rospy.signal_shutdown("Done.")		

###########################################

if __name__ == '__main__':
	# Get aruguments
	parser = argparse.ArgumentParser(usage = __usage__)
	parser.add_argument("--size", help = "Checkerboard size: MxN (default 7x5)")
	parser.add_argument("--square", help = "Checkerboard square size (default 0.030m)", type = float)
	parser.add_argument("--topic", help = "Camera topic", default = "/camera/image_rect_mono")
	args = parser.parse_args()
	
	if args.size is None:
		args.size = "7x5"
		print("WARNING: Checkerboard size not set, using default of %s." % args.size)
	
	if args.square is None:
		args.square = 0.030
		print("WARNING: Checkerboard square size not set, using default of %.3f." % args.square)
	
	# Parse the size argument
	size = tuple([int(c) for c in args.size.split("x")])
	
	est = estimator(args.topic, size, args.square)
	rospy.init_node('camera_height_estimation')
		
	try:
		rospy.spin()
	except rospy.ROSInterruptException:
		print("Interrupt Exception: Shutting down")
		