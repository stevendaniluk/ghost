#!/usr/bin/env python
from __future__ import print_function
from __future__ import division

import argparse
import cv2
import math
import numpy as np
import os
import subprocess 
import yaml

__usage__ = """
racing_line_extractor.py is used to extract the world coordinates of a 
racing line drawn on a track map. The intended use is to have an 
additional copy of the track map with the desired racing line drawn on,
as well as the start/finish line.

It is expected that map has a white bakground, with the boundaries, start
line, and racing drawn, drawn in black. racing_line_extractor.py will erode
the image in an attempt to reduce the lines to single 8-connected pixels.
The default erosion iterations is 1, but can be set with erode_iters arg.

The annotated track image and corresponding yaml file must be supplied as
input arguments. When the map is shown, click on each side of the start/
finish line in a clockwise direction.

The world coordinates of each point of the racing line (starting at the
intersection with the start/finish line) will be saved to a text file.
"""	

# NOTES:
# To adapt racing_line_extractor.py to different maps, it may be nexessary
# to alter the following sections:
#   -Number of erosion iterations
#   -checkForTrackEdge
#   -checkForRacingLine

###########################################

# Names for display windows
input_window_name = "Input Image"
rline_window_name = "Identified Racing Line"

alpha = 0.9  # Exponential moving average coefficient for search directions
start_click_pts = [] # Placeholder for user click points set in callback

###########################################
# captureCallback
#
# Simply saves the pixel coordinates of the clicks

def captureCallback(event, u, v, flags, param):	
	if event == cv2.EVENT_LBUTTONDOWN:
		start_click_pts.append([u, v])
		if len(start_click_pts) >= 2:
			cv2.destroyWindow(input_window_name)
			return

###########################################
# getNextMove
#
# Scans the neighbouring pixels of the image to determine the
# direction of the next move.
#
# There are five possible directions, and they are evaluted 
# based on the initial direction variabls
#
# Example: For inputs of right = 0.99 and down = 0.001, the 
# ranking of potential moves would be:
#        [/ 5 3]
#        [/ / 1]
#        [/ 4 2]

def getNextMove(img_in, u, v, right, down):
	u = int(u)
	v = int(v)
		
	# Look diagonally first
	if abs(right) > 0.5 and abs(down) > 0.5:
		du = int(np.sign(right))
		dv = int(np.sign(down))
		if img_in[v + dv, u + du] == 255:
			#print("diag")
			return du, dv
	
	# Look directly in the suspected direction (straight)
	du = int(np.sign(right)*(abs(right) >= abs(down)))
	dv = int(np.sign(down)*(abs(right) < abs(down)))
	if img_in[v + dv, u + du] == 255:
		#print("straight")
		return du, dv
	
	# 45 degree turn, in suspected direction
	du = int(np.sign(right))
	dv = int(np.sign(down))
	if img_in[v + dv, u + du] == 255:
		#print("45-a")
		return du, dv
	
	# 45 degree turn, in opposite of suspected direction
	du = int(np.sign(right)*(-1 + 2*(abs(right) >= abs(down))))
	dv = int(np.sign(down)*(-1 + 2*(abs(right) < abs(down))))
	if img_in[v + dv, u + du] == 255:
		#print("45-b")
		return du, dv
	
	# 90 degree turn, in suspected direction
	du = int(np.sign(right)*(abs(right) < abs(down)))
	dv = int(np.sign(down)*(abs(right) >= abs(down)))
	if img_in[v + dv, u + du] == 255:
		#print("90-a")
		return du, dv
	
	# 90 degree turn, in opposite of suspected direction
	du = int(np.sign(right)*(-1 + 2*(abs(right) > abs(down))))
	dv = int(np.sign(down)*(-1 + 2*(abs(right) >= abs(down))))
	if img_in[v + dv, u + du] == 255:
		#print("90-b")
		return du, dv
		
	# Default to original pt
	print("Warning! Next point not found")
	return 0, 0

###########################################
# checkForTrackEdge
# Checks if the current pixel is at the track boundary
# 
# Track boundaries are identified by the next predicted position being a
# zero pixel (i.e. past the track boundary).

def checkForTrackEdge(img_in, pt, right, down):
	return img_in[pt[1] + int(round(down)), pt[0] + int(round(right))] == 0

###########################################
# checkForRacingLine
# Checks if the current pixel is the intersection of the start line
# with the racing line.
# 
# The intersection is identified by neighbouring pixels on both sides 
# of the start line, and pixels crossing all edges of a window 
# surrounding the point.	

def checkForRacingLine(img_in, pt, right, down):
	d = 7 # window size
	top_edge = img_in[pt[1] - d//2, (pt[0] - d//2):(pt[0] + 1 + d//2)]
	bottom_edge = img_in[pt[1] + 1 + d//2, (pt[0] - d//2):(pt[0] + 1 + d//2)]
	left_edge = img_in[(pt[1] - d//2):(pt[1] + 1 + d//2), pt[0] - d//2]
	right_edge = img_in[(pt[1] - d//2):(pt[1] + 1 + d//2), pt[0] + 1 + d//2]
	all_edges = (np.sum(top_edge) > 0) and (np.sum(bottom_edge) > 0) \
	            and (np.sum(left_edge) > 0) and (np.sum(right_edge) > 0)
	
	xing_vals = img_in[(pt[1] - int(abs(round(right)))):(pt[1] + 1 + int(abs(round(right)))), 
	                   (pt[0] - int(abs(round(down)))):(pt[0] + 1 + int(abs(round(down))))]
	neighbours_present = np.sum(np.nonzero(xing_vals)) == 3
	
	return neighbours_present and all_edges

###########################################

if __name__ == '__main__':
	# Get aruguments
	parser = argparse.ArgumentParser(usage = __usage__)
	parser.add_argument("img_path", help = "Path image file", default = None)
	parser.add_argument("yaml_path", help = "Path track yaml file", default = None)
	parser.add_argument("--erode_iters", help = "Number of iterations for erosion", type = int, default = 1)
	args = parser.parse_args()
	
	if not os.path.exists(args.yaml_path):
		print("Error: path to track yaml file is not valid")
		exit()
	
	# Get map resolution from yaml file
	full_yaml_path = os.path.abspath(args.yaml_path)
	f = open(full_yaml_path, 'r')
	yaml_file = yaml.load(f)
	f.close()
	img_res = float(yaml_file["resolution"])
	
	# Parse image name/path
	abs_path, img_name = os.path.split(args.img_path)
	img_name_base, img_ext = os.path.splitext(img_name)
	
	# Valid image?
	valid_extentions = [".jpg", ".png"]
	if img_ext.lower() not in valid_extentions:
		print("Error: Image extention of %s is not valid. Options are %s" % (img_ext, valid_extentions))
		exit()
	
	# Load the image
	img = cv2.imread(args.img_path, 0)
	
	# Calculate the factor to resize image for display purposes
	resize_factor = 1.0/round(img.shape[1]/700);
		
	# Get click points on either side of the start finish line
	print("Click image on each side of the finish line in a clock-wise " \
		"direction with respect to the track.\n")
	
	# May need to resize image
	if resize_factor < 1:
		img_disp = cv2.resize(img, (0,0), fx=resize_factor, fy=resize_factor)
	else:
		img_disp = np.copy(img)
	
	cv2.namedWindow(input_window_name)
	cv2.setMouseCallback(input_window_name, captureCallback)
	cv2.imshow(input_window_name, img_disp)
	cv2.waitKey(0)
	
	# Adjust the start points with scale factor
	start_click_pts = np.array(start_click_pts)/resize_factor
	
	# Convert to black background and threshold
	_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV);
	
	# Erode with cross kernel, to leave 8-connected pixels
	kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
	img = cv2.erode(img, kernel_cross, iterations = args.erode_iters)
	
	# Display the eroded image, for debugging
	#if resize_factor < 1:
	#	img_disp = cv2.resize(img, (0,0), fx=resize_factor, fy=resize_factor)
	#else:
	#	img_disp = np.copy(img)
	#cv2.imshow(input_window_name, img_disp)
	#cv2.waitKey(0)
	
	# Crawl between start points to find the finish line
	du_click = start_click_pts[1][0] - start_click_pts[0][0]
	dv_click = start_click_pts[1][1] - start_click_pts[0][1]
	dist = round(math.sqrt(du_click**2 + dv_click**2))
	#print("du_click=%.2f, dv_click=%.2f" % (du_click, dv_click))
	
	click_xing = np.zeros((1, 2))
	for i in range(int(dist)):
		u = int(start_click_pts[0][0] + (i/dist)*du_click)
		v = int(start_click_pts[0][1] + (i/dist)*dv_click)
		
		if img[v, u] == 255:
			#print("Found start line intersection: u=%d, v=%d" % (u, v))
			click_xing = np.array([u, v])
			break;
	
	if np.sum(click_xing) == 0:
		print("Warning: No line found between click points.")
		exit()
	
	# Normalize pixel distances, and make sure they are valid
	du_click = du_click/dist
	dv_click = dv_click/dist
	if du_click == 0:
		du_click = 0.001
	if dv_click == 0:
		dv_click = 0.001
		
	# Crawl along start line to find intersection with racing line, and end 
	# points at track boundaries. These will be used as the start points for
	# crawling along the racing lane and track boundaries.
	#
	# Track boundaries are identified by the next predicted position being a
	# zero pixel (i.e. past the track boundary).
	
	# Initialize pixel and direction variables for start points of the
	# right and left lane boundaries, and racing line
	start_l_found = False
	start_l = np.copy(click_xing)
	right_l = dv_click
	down_l = -du_click
	
	start_r_found = False
	start_r = np.copy(click_xing)
	right_r = -dv_click
	down_r = du_click
	
	start_rline = np.zeros((1, 2))
	rline_found = False
	
	# Crawl along start line
	while not (start_l_found and start_r_found):
		# Crawl towards left lane
		if not start_l_found:
			du_l, dv_l = getNextMove(img, start_l[0], start_l[1], right_l, down_l)
			
			# Update directions
			right_l = alpha*right_l + (1 - alpha)*du_l
			down_l = alpha*down_l + (1 - alpha)*dv_l
			
			# Check for intersaction with boundary
			if checkForTrackEdge(img, start_l, right_l, down_l):
				#print("Left Lane Start: u=%d, v=%d" % (start_l[0], start_l[1]))
				start_l_found = True
			else:
				start_l += np.array([du_l, dv_l])
			
			# Check for racing line
			if not rline_found:
				if checkForRacingLine(img, start_l, right_l, down_l):
					#print("Found racing line: u=%d, v=%d" % (start_l[0], start_l[1]))
					rline_found = True
					start_rline = np.copy(start_l)
				
		# Crawl towards right lane
		if not start_r_found:
			du_r, dv_r = getNextMove(img, start_r[0], start_r[1], right_r, down_r)
			
			# Update directions
			right_r = alpha*right_r + (1 - alpha)*du_r
			down_r = alpha*down_r + (1 - alpha)*dv_r
			
			# Check for intersaction with boundary
			if checkForTrackEdge(img, start_r, right_r, down_r):
				#print("Right Lane Start: u=%d, v=%d" % (start_r[0], start_r[1]))
				start_r_found = True
			else:
				start_r += np.array([du_r, dv_r])
			
			# Check for racing line
			if not rline_found:
				if checkForRacingLine(img, start_r, right_r, down_r):
					#print("Found racing line: u=%d, v=%d" % (start_r[0], start_r[1]))
					rline_found = True
					start_rline = np.copy(start_r)
		
	# Crawl around racing line, starting perpendicular to start line, and in
	# the direction of the click points
	
	# Make an image show the identified line, and draw the starting line
	rline_img = np.ones(img.shape)*255
	cv2.line(rline_img, tuple(start_l), tuple(start_r), 0, thickness=3)
	
	# Initialize pixel and lap variables for racing line
	rline_pt = np.copy(start_rline)
	rline = np.copy(start_rline)
	lap_complete = False
	
	# Rotate directions from finish line crossing by 90 degrees
	right = 0.5*(abs(down_l) + abs(down_r))*np.sign(du_click)
	down = 0.5*(abs(right_l) + abs(right_r))*np.sign(dv_click)
	#print("RL Starting Directions: right=%.2f, down=%.2f" % (right, down))
	
	while not lap_complete:
		du, dv = getNextMove(img, rline_pt[0], rline_pt[1], right, down)
		
		at_start_pt = ((rline_pt[0] + du) == start_rline[0] and (rline_pt[1] + dv) == start_rline[1])
		one_pt_away = ((abs(start_rline[0] - rline_pt[0]) <= 1) 
		               and (abs(start_rline[1] - rline_pt[1]) <= 1)
		               and rline.shape[0] > 5)
		if at_start_pt or one_pt_away:
			#print("Lap complete")
			lap_complete = True
		else:
			rline_pt += np.array([du, dv])
			rline = np.vstack([rline, rline_pt])
			
			# Update directions
			right = alpha*right + (1 - alpha)*du
			down = alpha*down + (1 - alpha)*dv
			
			# Draw line on image (make a little bigger so it shows up)
			rline_img[(rline_pt[1] - 1):(rline_pt[1] + 2), (rline_pt[0] - 1):(rline_pt[0] + 2)] = 0
		
	# Make all racing line points relative to centre of the start/finish line
	# (have to negate the y-direction since image and world Y is reversed)
	origin_pixel = (start_l + start_r)/2
	rline -= np.array([int(origin_pixel[0]), int(origin_pixel[1])])
	rline[:, 1] *= -1
	
	# Scale racing line to world units [meters]
	rline = rline*img_res
	
	# Save racing line to text file
	textfile_name = img_name_base + ".txt"
	textfile_full_name = os.path.join(abs_path, textfile_name)
	if os.path.exists(textfile_full_name):
		print("File %s already exists. Overwriting.\n" % textfile_name)
	
	with open(textfile_full_name, "w") as text_file:
		text_file.write("x [m] \ty [m]\n")
		for pt in rline:
			text_file.write("%.4f\t%.4f\n" % (pt[0], pt[1]))
	
	print("The identified racing line and starting line is shown in the figure.")
	print("The world coordinates (in meters) of each point in the racing line have been saved to %s, " \
		"with the x and y coordinates in the first and second columns, respectively.\n" % (textfile_name))	
	print("Please set the origin value in the track yaml file to: ")
	print("[%.5f, %.5f, 0.0]\n" % (-img_res*origin_pixel[0], -img_res*(img.shape[0] - origin_pixel[1])))
	
	# Show the identified line
	if resize_factor < 1:
		rline_img = cv2.resize(rline_img, (0,0), fx=resize_factor, fy=resize_factor)
	cv2.namedWindow(rline_window_name)
	cv2.imshow(rline_window_name, rline_img)
	cv2.waitKey(0)
	