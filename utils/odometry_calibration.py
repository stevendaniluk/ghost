#!/usr/bin/env python
from __future__ import print_function

import rospy
import rosbag
import rospkg
import argparse
import os
import math
import numpy as np
import yaml
import subprocess 

__usage__ = """
	 odometry_calibration.py performs calibration for odometry data, 
	 estimating correction factors for the wheel diameter, Cd, and 
	 the wheelbase, Cb.
	 
	 Maneuvers for calibration are straight driving a distance of s 
	 [meters], and driving in circles N times. Each maneuver must be 
	 recorded in seperate bagfiles, with the title containing either
	 "straight" or "rotate", depending on the maneuver type. Bags must 
	 contain the /arduino_state topic.
	 
	 Required arguments are the distance traveled, s, for the straight
	 line calibration, and the number of rotations, N, for the 
	 rotational calibration (the direction does not need to be indicated).
"""

required_topic = "/arduino_state"

###########################################

def checkBags(bag_dir):
	# Find bagfiles
	straight_bags = []
	rotate_bags = []
	for name in os.listdir(bag_dir):
		# Make sure it's a bag
		ext = os.path.splitext(name)[1]
		if ext.lower() != ".bag":
			continue
		if "straight" in os.path.basename(name).lower():
			straight_bags.append(os.path.join(bag_dir, name))
		elif "rotate" in os.path.basename(name).lower():
			rotate_bags.append(os.path.join(bag_dir, name))
		
	# Sort alphabetically
	straight_bags = sorted(straight_bags, key=str.lower)
	rotate_bags = sorted(rotate_bags, key=str.lower)
	
	straight_bags_checked = []
	for bag_name in straight_bags:
		info_dict = yaml.load(subprocess.Popen(['rosbag', 'info', '--yaml', bag_name], stdout=subprocess.PIPE).communicate()[0])
		for topic_dict in info_dict["topics"]:
			if required_topic in topic_dict["topic"]:
				straight_bags_checked.append(bag_name)
				break
	
	rotate_bags_checked = []
	for bag_name in rotate_bags:
		info_dict = yaml.load(subprocess.Popen(['rosbag', 'info', '--yaml', bag_name], stdout=subprocess.PIPE).communicate()[0])
		for topic_dict in info_dict["topics"]:
			if required_topic in topic_dict["topic"]:
				rotate_bags_checked.append(bag_name)
				break
	
	if((len(straight_bags_checked) + len(rotate_bags_checked)) == 0):
		print("No bags found with the required topics: %s" % required_topic)
		exit()
		
	print("Found %d straight, and %d rotate bagfiles." % (len(straight_bags_checked), len(rotate_bags_checked)))
	
	return straight_bags_checked, rotate_bags_checked

###########################################

def straightLineCalibration(straight_bags, s, params):
	# Default to value from yaml file
	Cd = float(params["chassis"]["Cd"])
	
	# Get car parameters
	d = float(params["chassis"]["d"])
	m = float(params["odometry"]["pulses_per_rev"])
	
	# Loop through each bag
	Cd_i = []
	s_i = []
	for bag_name in straight_bags:
		bag = rosbag.Bag(bag_name)
		
		# Get the pulse counts
		first_message = True
		for topic, msg, time in bag.read_messages(topics = required_topic):
			if first_message:
				n_initial = msg.FL_pulse_count + msg.FR_pulse_count + msg.RL_pulse_count + msg.RR_pulse_count
				first_message = False
			
			n_final = msg.FL_pulse_count + msg.FR_pulse_count + msg.RL_pulse_count + msg.RR_pulse_count
		bag.close()
		n_total = n_final - n_initial
				
		# Solve for calibration coefficient
		Cd_i.append(4*m*s/(np.pi*d*n_total))
		# Solve for estimated distance
		s_i.append(np.pi*d*n_total/(4*m))
	
	if(len(Cd_i) > 0):
		print("\n----- Straight Line Calibration Results -----")
		
		# Print estimate from each bag
		s_string = "  s Values: "
		for i in range(len(s_i)):
			s_string = s_string + ("%.3f" % s_i[i]) + ", "
		s_string = s_string[:-2]
		print(s_string)
		
		# Take the average of all measurements
		Cd = np.mean(Cd_i)
		print("  Cd = %.5f, Var = %.5f\n" % (Cd, np.var(Cd_i)))
	else:
		print("  No straight line calibration data.")
	
	return Cd		

###########################################

def rotationalCalibration(rotate_bags, N, Cd, params):
	# Default to value from yaml file
	Cb = float(params["chassis"]["Cd"])
	
	# Get parameters
	d = float(params["chassis"]["d"])
	b = float(params["chassis"]["b"])
	delta_max = float(params["chassis"]["delta_max"])*np.pi/180.0
	m = float(params["odometry"]["pulses_per_rev"])
	
	# Loop through each bag
	Cb_i = []
	N_i = []
	for bag_name in rotate_bags:
		bag = rosbag.Bag(bag_name)
		
		# Get the pulse counts
		delta_n_R = 0
		first_message = True
		for topic, msg, time in bag.read_messages(topics = required_topic):
			if first_message:
				prev_RL_pulse_count = msg.RL_pulse_count
				prev_RR_pulse_count = msg.RR_pulse_count
				first_message = False
				
			# Find pulses this time step
			n_RL = msg.RL_pulse_count - prev_RL_pulse_count
			n_RR = msg.RR_pulse_count - prev_RR_pulse_count
			prev_RL_pulse_count = msg.RL_pulse_count
			prev_RR_pulse_count = msg.RR_pulse_count
			
			# Find differences for front axle
			delta_n_R += (n_RR - n_RL)
			
		bag.close()
		
		# Solve for calibration coefficient
		Cb_i.append(abs((Cd*d/(2*m*N*b))*delta_n_R))
		# Store estimated N
		N_i.append((Cd*d/(2*m*b))*delta_n_R)
	
	if(len(Cb_i) > 0):
		print("----- Rotational Calibration Results -----")
		# Print estimate from each bag
		N_string = "  N Values: "
		for i in range(len(N_i)):
			N_string = N_string + ("%.3f" % N_i[i]) + ", "
		N_string = N_string[:-2]
		print(N_string)
		
		# Take the average of all measurements
		Cb = np.mean(Cb_i)
		print("  Cb = %.5f, Var = %.5f\n" % (Cb, np.var(Cb_i)))
	else:
		print("No rotational calibration data.")
		
	return Cb

###########################################

if __name__ == '__main__':
	# Get aruguments
	parser = argparse.ArgumentParser(usage = __usage__)
	parser.add_argument("s", help = "Distance traveled for straight line calibration", type = float)
	parser.add_argument("N", help = "Number of rotations for rotational calibration", type = float)
	parser.add_argument("--path", help = "Path to directory containing bagfiles", default = os.getcwd())
	args = parser.parse_args()
			
	# Load bags
	straight_bags, rotate_bags = checkBags(args.path)
	
	# Get parameters
	param_path = os.path.join(rospkg.RosPack().get_path("ghost"), "param/car_params.yaml")	
	f = open(param_path, 'r')
	params = yaml.load(f)
	f.close()	
	
	# Perform calibration
	Cd = straightLineCalibration(straight_bags, args.s, params)
	Cb = rotationalCalibration(rotate_bags, args.N, Cd, params)
	