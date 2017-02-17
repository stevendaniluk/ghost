#!/usr/bin/env python
from __future__ import print_function

import rospy
import rosbag
import rospkg
import tf
import argparse
import os
import math
import numpy as np
import yaml
import subprocess
import matplotlib.pyplot as plt

__usage__ = """
	 odometry_plotter.py produces plots of the odometry pose estimation
	 from the /arduino_state and/or /odometry/integrated topics.
	 
	 Odometry estimate via the /arduino_state topic will be performed
	 using the parameters in ghost/param/car_params.yaml.
	 
	 One plot will be produced for each bag in the directory.
"""

raw_topic = "/arduino_state"
integrated_topic = "/odometry/integrated"
valid_topics = [raw_topic, integrated_topic]

###########################################

def checkBags(bag_dir):
	# Find bagfiles
	bags = []
	for name in os.listdir(bag_dir):
		# Make sure it's a bag
		ext = os.path.splitext(name)[1]
		if ext.lower() != ".bag":
			continue
		bags.append(os.path.join(bag_dir, name))
		
	# Sort alphabetically
	bags = sorted(bags, key=str.lower)
		
	bags_checked = []
	for bag_name in bags:
		info_dict = yaml.load(subprocess.Popen(['rosbag', 'info', '--yaml', bag_name], stdout=subprocess.PIPE).communicate()[0])
		for topic_dict in info_dict["topics"]:
			if any(topic in topic_dict["topic"] for topic in valid_topics):
				bags_checked.append(bag_name)
				break
	
	if(len(bags_checked) == 0):
		print("No bags found with one of the required topics: %s, %s" % (raw_topic, integrated_topic))
		exit()
		
	print("Found %d bagfiles." % len(bags_checked))
	
	return bags_checked

###########################################

def getPoses(bag_name, params):
	
	# Get parameters
	d = float(params["chassis"]["d"])
	Cd = float(params["chassis"]["Cd"])
	b = float(params["chassis"]["b"])
	Cb = float(params["chassis"]["Cb"])
	delta_max = float(params["chassis"]["delta_max"])*np.pi/180.0
	m = float(params["odometry"]["pulses_per_rev"])
	
	# See which plots need to be made
	plot_raw = False
	plot_integrated = False
	info_dict = yaml.load(subprocess.Popen(['rosbag', 'info', '--yaml', bag_name], stdout=subprocess.PIPE).communicate()[0])
	for topic_dict in info_dict["topics"]:
		if raw_topic in topic_dict["topic"]:
			plot_raw = True
		if integrated_topic in topic_dict["topic"]:
			plot_integrated = True
	
	# Load the bag
	bag = rosbag.Bag(bag_name)
	
	# Loop through bag data
	raw_initialized = False
	int_initialized = False
	raw_poses = np.zeros((1,3))         # [x, y, psi]
	integrated_poses = np.zeros((1,3))  # [x, y, psi]
	for topic, msg, time in bag.read_messages(topics = valid_topics):			
		
		# Integrate raw data to get raw poses
		if plot_raw and topic == raw_topic:
			# Need a previous message
			if not raw_initialized:
				prev_raw_msg = msg
				raw_initialized = True
				continue
			
			# Find encoder pulses and steering angle for this time step
			n_FL = (msg.FL_pulse_count - prev_raw_msg.FL_pulse_count)
			n_FR = (msg.FR_pulse_count - prev_raw_msg.FR_pulse_count)
			n_RL = (msg.RL_pulse_count - prev_raw_msg.RL_pulse_count)
			n_RR = (msg.RR_pulse_count - prev_raw_msg.RR_pulse_count)
			delta = msg.steering*delta_max
			
			# Compute change in distance and orientation
			ds = (np.pi*d*Cd/(4*m))*((n_FL + n_FR)*math.cos(delta) + n_RL + n_RR)
			dpsi = ((np.pi*d*Cd)/(m*b*Cb))*(n_RR - n_RL)
			
			# Integrate for next pose
			dt = (msg.header.stamp - prev_raw_msg.header.stamp).to_sec()
			if abs(dpsi/dt) > 1e-6:
				# Exact
				x = raw_poses[-1][0] + (ds/dpsi)*(math.sin(raw_poses[-1][2] + dpsi) - math.sin(raw_poses[-1][2]))
				y = raw_poses[-1][1] - (ds/dpsi)*(math.cos(raw_poses[-1][2] + dpsi) - math.cos(raw_poses[-1][2]))
				psi = raw_poses[-1][2] + dpsi
			else:
				# 2nd order Runge Kutta
				x = raw_poses[-1][0] + ds*math.cos(raw_poses[-1][2] + 0.5*dpsi)
				y = raw_poses[-1][1] + ds*math.sin(raw_poses[-1][2] + 0.5*dpsi)
				psi = raw_poses[-1][2] + dpsi
			
			raw_poses = np.vstack([raw_poses, [x, y, psi]])
			prev_raw_msg = msg
						
		# Get integrated odometry poses
		if plot_integrated and topic == integrated_topic:
			
			# Need the first pose and orientation to adjust future messages 
			# (in case the initial pose is not at the origin)
			if not int_initialized:
				# Save the initial pose
				(r_dud, p_dud, initial_yaw) = tf.transformations.euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
				initial_pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, initial_yaw])
				
				# Precompute the rotation matrix to adjust future messages
				rot = np.array([[math.cos(initial_yaw), math.sin(initial_yaw), 0.0], [-math.sin(initial_yaw), math.cos(initial_yaw), 0.0], [0.0, 0.0, 1.0]])
				int_initialized = True
			
			# Get new pose and orientation
			(r_dud, p_dud, new_yaw) = tf.transformations.euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
			new_pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, new_yaw])
			
			# Correct the pose, and add to data
			corrected_pose = np.matmul(rot, new_pose - initial_pose)
			integrated_poses = np.vstack([integrated_poses, corrected_pose])
					
	bag.close()
		
	return raw_poses, integrated_poses

###########################################

def plotPoses(bag_name, raw_poses, integrated_poses, skip_pt):
	# Trim the data
	if skip_pt != 0:
		raw_poses = raw_poses[range(0, raw_poses.shape[0], skip_pt)]
		integrated_poses = integrated_poses[range(0, integrated_poses.shape[0], skip_pt)]
	
	plt.figure()	
	plt.title(os.path.basename(bag_name))
	plt.xlabel("X [m]")
	plt.ylabel("Y [m]")
	
	# Plot raw poses
	if raw_poses.shape[0] > 1:
		plt.plot(raw_poses[:,0], raw_poses[:, 1], "bx", label = "Raw")
	
	# Plot the integrated poses
	if integrated_poses.shape[0] > 1:
		plt.plot(integrated_poses[:,0], integrated_poses[:, 1], "rx", label = "Integrated")
	
	plt.legend()
	plt.axis('equal')
	
###########################################

if __name__ == '__main__':
	# Get aruguments
	parser = argparse.ArgumentParser(usage = __usage__)
	parser.add_argument("--path", help = "Path to directory containing bagfiles", default = os.getcwd())
	parser.add_argument("--skip_pt", help = "Plot every nth pose", type = int, default = 0)
	args = parser.parse_args()
		
	# Load bags
	bag_names = checkBags(args.path)
	
	# Get parameters
	param_path = os.path.join(rospkg.RosPack().get_path("ghost"), "param/car_params.yaml")	
	f = open(param_path, 'r')
	params = yaml.load(f)
	f.close()
	
	# Loop through bags	
	for name in bag_names:
		# Get poses from each bag
		raw_poses, integrated_poses = getPoses(name, params)
		# Plot the poses
		plotPoses(name, raw_poses, integrated_poses, args.skip_pt)
		
	plt.show()
		