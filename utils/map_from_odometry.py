#!/usr/bin/env python
from __future__ import print_function

import argparse
import cv2
import math
import numpy as np
import os
import rosbag
import rospkg
import rospy
import subprocess
import tf
import yaml

__usage__ = """
map_from_odometry.py generates a map from odometry data by creating lane
boundaries at a fixed distance on each side of the car's path. Everything
is relative to the initial pose of the car.

The following will be generated:
  -One image with the track outline
  -One image with the outline, start/finish line, and centreline
  -yaml file for the track

The path to the bagfile with odometry data, and th desired track width are
required arguments.
"""

###########################################
# getPoses
#
# Checks the bag file for the required topics, then extracts all the poses.
# Each position and orientation is relative to the initial first pose.

def getPoses(bag_name, topic):
	if not os.path.exists(bag_name):
		print("Bagfile does not exist")
		exit()
	
	# Check that the bag has the proper topics
	info_dict = yaml.load(subprocess.Popen(['rosbag', 'info', '--yaml', bag_name], stdout=subprocess.PIPE).communicate()[0])
	for topic_dict in info_dict["topics"]:
		if topic not in topic_dict["topic"]:
			print("Bagfile does not have the proper topic: %s" % topic)
			exit()
			
	# Load the bag
	bag = rosbag.Bag(bag_name)
	
	# Loop through bag data
	initialized = False
	poses = np.zeros((1,3))    # [x, y, psi]
	for topic, msg, time in bag.read_messages(topics = topic):			
		# Get filtered odometry poses
		# Need the first pose and orientation to adjust future messages 
		# (in case the initial pose is not at the origin)
		if not initialized:
			# Save the initial pose
			(r_dud, p_dud, initial_yaw) = tf.transformations.euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
			initial_pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, initial_yaw])
			
			# Precompute the rotation matrix to adjust future messages
			rot = np.array([[math.cos(initial_yaw), math.sin(initial_yaw), 0.0], [-math.sin(initial_yaw), math.cos(initial_yaw), 0.0], [0.0, 0.0, 1.0]])
			initialized = True
		
		# Get new pose and orientation
		(r_dud, p_dud, new_yaw) = tf.transformations.euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
		new_pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, new_yaw])
		
		# Correct the pose
		corrected_pose = np.matmul(rot, new_pose - initial_pose)
		
		# Only add if there is sufficient movement (avoids clusters of points)
		delta_pose = poses[-1] - corrected_pose
		d = np.sqrt(delta_pose[0]**2 + delta_pose[1]**2)
		if d > 0.001:
			poses = np.vstack([poses, corrected_pose])
					
	bag.close()
	return poses

###########################################
# loopClosure
#
# Finds the closest point near the end of the loop, then applies an
# exponentially decreasing offset to each point to create a smooth closure.

def loopClosure(poses):
	num_pts = poses.shape[0]
	
	# Crawling backwards, find the index that minimizes the distance from 
	# the starting point
	d_min = 0
	index = num_pts - 1
	for pt in reversed(poses):
		d = np.sqrt((pt[0] - poses[0, 0])**2 + (pt[1] - poses[0, 1])**2)
		if d > d_min:
			break
		else:
			index -= 1
	
	print("Closest Point Index %d/%d" % (index, (num_pts - 1)))
	
	# Trim points after closest point
	poses = poses[0:index]
	
	x_error = poses[0, 0] - poses[-1, 0]
	y_error = poses[0, 1] - poses[-1, 1]
		
	# Apply exponentially decreasing offset to all points
	alpha = 0.995  # Pretty much reaches zero after ~1000 points
	offset = np.array([x_error, y_error, 0])
	for pt in reversed(poses):
		pt += offset
		offset *= alpha	
	
	return poses

###########################################
# createMap
#
# Plots all the poses and lane boundaries on an image. One image with
# only the track, and one with the track, start/finish line, and centreline
# are saved. A yaml file for the track is also generated

def createMap(poses, lane_w, out_name):
	# World size
	x_min = np.min(poses[:, 0]) - lane_w
	x_max = np.max(poses[:, 0]) + lane_w
	y_min = np.min(poses[:, 1]) - lane_w
	y_max = np.max(poses[:, 1]) + lane_w
	
	world_w = x_max - x_min
	world_h = y_max - y_min
	world_c_x = (x_min + x_max)/2.0
	world_c_y = (y_min + y_max)/2.0
				
	# Determine image size and resolution
	ar = world_w/world_h
	max_num_pixels = 0.8*2000*1400  # 80% of maximum possible
	img_w = int(np.sqrt(max_num_pixels*ar))
	img_h = int(img_w/ar)
	res = world_w/img_w  # meters/pixel
	
	# Offset all poses to be centred
	poses -= np.array([world_c_x, world_c_y, 0.0])
	
	# Find the left and right track boundaries
	left_edges = []
	right_edges = []
	for pt in poses:
		left_x = pt[0] - 0.5*lane_w*math.sin(pt[2])
		left_y = pt[1] + 0.5*lane_w*math.cos(pt[2])
		left_edges.append([left_x, left_y])
		
		right_x = pt[0] + 0.5*lane_w*math.sin(pt[2])
		right_y = pt[1] - 0.5*lane_w*math.cos(pt[2])
		right_edges.append([right_x, right_y])
	
	left_edges = np.array(left_edges)
	right_edges = np.array(right_edges)
	
	# Create a blank image, and draw a line between all the points
	img_track = np.ones((img_h, img_w))*255
	prev_l_pixel = (int(left_edges[0, 0]/res + img_w/2), int(left_edges[0, 1]/res + img_h/2))
	prev_r_pixel = (int(right_edges[0, 0]/res + img_w/2), int(right_edges[0, 1]/res + img_h/2))
	for i in range(poses.shape[0]):
		# Draw the lane edges
		l_pixel = (int(left_edges[i, 0]/res + img_w/2), int(left_edges[i, 1]/res + img_h/2))
		r_pixel = (int(right_edges[i, 0]/res + img_w/2), int(right_edges[i, 1]/res + img_h/2))
		
		cv2.line(img_track, prev_l_pixel, l_pixel, 0, 3)
		cv2.line(img_track, prev_r_pixel, r_pixel, 0, 3)
		
		prev_l_pixel = l_pixel
		prev_r_pixel = r_pixel
	
	# Make another image with the driven line, and start finish line
	img_track_annotated = img_track.copy()
	prev_c_pixel = (int(poses[0, 0]/res + img_w/2), int(poses[0, 1]/res + img_h/2))
	for i in range(poses.shape[0]):
		# Draw the path point
		c_pixel = (int(poses[i, 0]/res + img_w/2), int(poses[i, 1]/res + img_h/2))
		cv2.line(img_track_annotated, prev_c_pixel, c_pixel, 0, 3)
		prev_c_pixel = c_pixel
		
	# Draw the start finish line on the track
	start_l_pixel = (int(left_edges[0, 0]/res + img_w/2), int(left_edges[0, 1]/res + img_h/2))
	start_r_pixel = (int(right_edges[0, 0]/res + img_w/2), int(right_edges[0, 1]/res + img_h/2))
	cv2.line(img_track_annotated, start_l_pixel, start_r_pixel, 0, 3)
	
	# Save the images to the current directory
	cwd = os.getcwd()
	cv2.imwrite(os.path.join(cwd, (out_name + ".jpg")), img_track)
	cv2.imwrite(os.path.join(cwd, (out_name + "_centreline.jpg")), img_track_annotated)
	
	# Create yaml file for the new track.
	# Need to determine the origin, which is the world coords of the ower 
	# left pixel, which should relative to the initial pose, so that start 
	# point is considered [0, 0]
	origin_x = -0.5*world_w - poses[0, 0]
	origin_y = -0.5*world_h - poses[0, 1]
	
	with open((out_name + ".yaml"), "w") as yaml_file:
		yaml_file.write("# Auto-generated file from map_from_odometry.py\n")
		yaml_file.write("image: %s.jpg\n" % out_name)
		yaml_file.write("resolution: %f  # meters/pixel\n" % res)
		yaml_file.write("origin: [%f, %f, 0.0]  # Lower left pixel in world frame [x, y, yaw]\n" % (origin_x, origin_y))
		yaml_file.write("occupied_thresh: 0.65  # Pixel occupied when probability greater than this\n")
		yaml_file.write("free_thresh: 0.196  # Pixel free when probability less than this\n")
		yaml_file.write("negate: 0  # Reverse white/black semantics\n")
		
	# Calculate the factor to resize image for displaying
	resize_factor = 500.0/max(img_w, img_w);
	if resize_factor < 1:
		img_track_disp = cv2.resize(img_track, (0,0), fx=resize_factor, fy=resize_factor)
		img_track_annotated_disp = cv2.resize(img_track_annotated, (0,0), fx=resize_factor, fy=resize_factor)
	else:
		img_track_disp = img_track.copy()
		img_track_annotated_disp = img_track_annotated.copy()
	
	cv2.imshow("Track", img_track_disp)
	cv2.imshow("Track Annotated", img_track_annotated_disp)
	cv2.waitKey(0)

###########################################

if __name__ == '__main__':
	# Get aruguments
	parser = argparse.ArgumentParser(usage = __usage__)
	parser.add_argument("path", help = "Path to bagfile", default = os.getcwd())
	parser.add_argument("width", help = "Width of track [meters]", type = float)
	parser.add_argument("--name", help = "Name of generated track", default = "new_track")
	parser.add_argument("--topic", help = "Topic containing odometry data", default = "/odometry/filtered")
	args = parser.parse_args()
		
	poses = getPoses(args.path, args.topic)
	poses = loopClosure(poses)
	createMap(poses, args.width, args.name);
			