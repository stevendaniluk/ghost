define lidar ranger (
	sensor(       
		# laser-specific properties
		range [0.3  4.0]
		fov 70.0
		samples 100
	)
	model(
		# generic model properties
		size [ 0.07 0.07 0.05 ]  
		color "red"
	)
)

define car position(
	pose [ 0.0 0.0 0.0 0.0 ]
	
	size [ 0.2552 0.2552 0.40 ]
	origin [ 0.0 0.0 0.0 0.0 ]
	gui_nose 1
	drive "car"
	wheelbase 0.260
	color "blue"
	
	# [ xmin xmax ymin ymax zmin zmax amin amax ]				
  velocity_bounds [-1 15 -5 5 0 0 -360 360]					
  acceleration_bounds [-10 10 -10 10 0 0 -90 90]
	
	localization "odom"
	odom_error [0.03 0.03 0.00 0.05]
	
	lidar(pose [ -0.1 0.0 -0.11 0.0 ])
)