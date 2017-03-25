/* LaneDetector Nodelet
*
* Images from the camera are processed to detect edges (i.e. the edges
* of the floor), the pixel coordinates of which are then projected 
* into real world coordinates.
*
* The resulting edges are then scanned at predefined rows (based on seperation
* distance) from the centre towards the left and right. The centre point used
* in lane detection changes as an exponential moving average of previous rows.
*
* The detected lane points are published either as a XY pointcloud or laserscan
* message. Additionally, the points are projected on an image to be published (along
* with the lane centre), and as a 3D pointcloud to be visible in RVIZ.
*
* The camera pose is determined from the TF frames, then the roll and pitch are
* adjusted based on IMU measurements. With each camera pose update the rotation
* matrices for pixel-to-world and world-to-pixel are updated.
*
* ----------------------------------------------------------------------------------
*/
#include <cv_bridge/cv_bridge.h>
#include <dynamic_reconfigure/server.h>
#include <geometry_msgs/Quaternion.h>
#include <ghost/LaneDetectorConfig.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/LaserScan.h>
#include <nodelet/nodelet.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <ros/console.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Header.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>

#define PI 3.14159265

typedef pcl::PointCloud<pcl::PointXY> PointCloudXY;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloudXYZ;

namespace ghost {

class LaneDetector : public nodelet::Nodelet {
	// Nodehandles
	ros::NodeHandle nh_;
	ros::NodeHandle pnh_;
	
	// Image transport
	boost::shared_ptr<image_transport::ImageTransport> it_;
	boost::shared_ptr<image_transport::ImageTransport> private_it_;
	
	// Pubs, Subs, and tf
	image_transport::CameraSubscriber sub_camera_;  // Camera
	ros::Subscriber sub_imu_;                       // IMU
	ros::Publisher pub_pc_;                         // Lane pointcloud
	ros::Publisher pub_pc_vis_;                     // Lane pointcloud only for visualization
	ros::Publisher pub_laserscan_;                  // Laserscan message
	image_transport::Publisher pub_lane_pts_;       // Image with lane hits and centreline
	image_transport::Publisher pub_edges_;          // Image with edges
	image_transport::Publisher pub_bev_;            // Bird's eye view image
	image_transport::Publisher pub_histo_;          // Image histogram
	image_transport::Publisher pub_otsu_;           // Image with Otsu thresholding
	tf::TransformListener tf_listener_;             // Camera transform
		
	// Camera properties
	tf::StampedTransform cam_tf_;  // Most recent transform between camera and base_footprint
	double cam_Z_;                 // Height of camera from ground
	double cam_roll_;              // Angle of camera (about the X axis)
	double cam_pitch_;             // Angle of camera (about the Y axis)
	double cam_yaw_;               // Angle of camera (about the Z axis)
	int cam_w_;                    // Width of camera image (ROI)
	int cam_h_;                    // Height of camera image (ROI)
	int cam_x_offset_;             // Offset from left pixel for ROI
	int cam_y_offset_;             // Offset from top pixel for ROI
	double cam_FOV_;               // Field of View
	double cam_max_width_;         // World width at max scan range
	double cam_min_depth_;         // World width at bottom pixel row
	cv::Mat K_;                    // Camera intrinsic matrix
	cv::Mat K_i_;                  // Inverse of camera intrinsic matrix
	cv::Mat t_cw_;                 // Translation of camera wrt world frame
	cv::Mat Rz_cw_;                // Rotation of camera wrt world Z axis
	cv::Mat A_cw_;                 // Intermediate transformation [R_cw][K^-1]
	cv::Mat A_wc_;                 // Intermediate transformation [R_cw^-1][K_i]
	
	// Bird's Eye View image properties
	int bev_w_;           // Image width
	int bev_h_;           // Image height
	int bev_ppm_;         // Pixel-per-meter for transformed image
	cv::Mat H_bev_;       // BEV transformation matrix		
	
	// Image filtering parameters 
	enum canny_thresh_enum {OTSU, MEDIAN, MEAN, MANUAL};
	bool apply_gauss_;              // Flag for applying Gaussian filter
	int gauss_k_size_;              // Size of kernal for Gaussian blur
	double gauss_sigma_x_;          // Gaussian kernel standar deviation in the X direction
	double gauss_sigma_y_;          // Gaussian kernel standar deviation in the Y direction
	int canny_type_;                // Enum for canny threshold calculation method
	int canny_lower_;               // Lower threshold for Canny edge detection (during manual mode)
	int canny_upper_;               // Upper threshold for Canny edge detection (during manual mode)    
	
	// Detection properties
	double max_range_;                        // Maximum detectable range
	double min_range_;                        // Minimum detectable range
	std::string detect_frame_id_;             // Name of the frame to attach to the lane points message
	std::vector<int> scan_rows_;              // Pixel rows to detect lanes in
	double detect_dx_;                        // Depth distance between lane detections
	double lane_centre_alpha_;                // Coefficient for exponential moving average of lane centreline
	PointCloudXY pc_msg_;                     // Pointcloud message containing lane points
	PointCloudXYZ pc_vis_msg_;                // 3D version of pc_msg_ for visualization with RVIZ
	sensor_msgs::LaserScan laserscan_msg_;    // LaserScan message containing lane points

	// Miscellaneous
	bool initialized_;       // Flag for if the transforms and detector have been initialized
	bool use_imu_;           // If the IMU should be used to adjust camera pitch and roll
	std::string imu_topic_;  // Topic name to subscribe to
	bool imu_effect_reset_;  // Flag for resetting the transformations when IMU not used
	
	// Method declarations
	void onInit();
	void reconfigureCallback(ghost::LaneDetectorConfig &config, uint32_t level);
	void imuCallback(const sensor_msgs::Imu::ConstPtr &msg);
	void connectCb();
	void imageCb(const sensor_msgs::ImageConstPtr& image_msg, const sensor_msgs::CameraInfoConstPtr& info_msg);
	bool initializeTransforms(const sensor_msgs::CameraInfoConstPtr& info_msg);
	bool initializeDetector();
	void detectLanes(cv::Mat &img, const std_msgs::Header &header);
	void imageDebugging(cv::Mat &img, const std_msgs::Header &header);
	cv::Point2f convertPixelToWorld(const cv::Point2f &pixel);
	cv::Point2f convertWorldToPixel(const cv::Point2f &p);
};// end LaneDetector class def

/* Initializer
*
*/
void LaneDetector::onInit(){	
	// Get the nodehandles
	nh_ = getNodeHandle();
	pnh_ = getPrivateNodeHandle();
	
	initialized_ = false;
	imu_effect_reset_ = false;
	
	// Load configuration parameters from the parameter server
	bool debug;
	pnh_.param<bool>("debug", debug, false);
	pnh_.param<int>("bev_ppm", bev_ppm_, 100);
	pnh_.param<double>("min_range", min_range_, 0.3);
	pnh_.param<double>("max_range", max_range_, 5.0);
	pnh_.param<double>("detect_dx", detect_dx_, 0.1);
	pnh_.param<std::string>("detect_frame_id", detect_frame_id_, "camera_ground");
	pnh_.param<std::string>("imu_topic", imu_topic_, "/imu/data");
	
	// Setup dynamic reconfigure
	dynamic_reconfigure::Server<ghost::LaneDetectorConfig> *server;
	server = new dynamic_reconfigure::Server<ghost::LaneDetectorConfig>(pnh_);
	dynamic_reconfigure::Server<ghost::LaneDetectorConfig>::CallbackType f;
	f = boost::bind(&LaneDetector::reconfigureCallback, this, _1, _2);
	server->setCallback(f);	
	
	// Manage debugging logger level
	if(debug){
		if(ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug) ) {
			ros::console::notifyLoggerLevelsChanged();
		}
	}
	
	// Setup image transport
	it_.reset(new image_transport::ImageTransport(nh_));
	private_it_.reset(new image_transport::ImageTransport(pnh_));

	// Monitor whether anyone is subscribed to the output
	ros::SubscriberStatusCallback connect_cb = boost::bind(&LaneDetector::connectCb, this);
	image_transport::SubscriberStatusCallback image_connect_cb = boost::bind(&LaneDetector::connectCb, this);
	
	// Setup the publishers (linking them to connectCb)
	pub_pc_ = pnh_.advertise<PointCloudXY>("lane_pointcloud", 5, connect_cb, connect_cb);
	pub_pc_vis_ = pnh_.advertise<PointCloudXYZ>("lane_pointcloud_vis", 5, connect_cb, connect_cb);
	pub_laserscan_ = pnh_.advertise<sensor_msgs::LaserScan>("lane_scan", 5, connect_cb, connect_cb);
	pub_lane_pts_  = private_it_->advertise("lane_pts_image",  1, image_connect_cb, image_connect_cb);
	pub_edges_  = private_it_->advertise("edges_image",  1, image_connect_cb, image_connect_cb);
	pub_bev_  = private_it_->advertise("bev_image",  1, image_connect_cb, image_connect_cb);
	pub_histo_  = private_it_->advertise("histogram",  1, image_connect_cb, image_connect_cb);
	pub_otsu_  = private_it_->advertise("otsu_image",  1, image_connect_cb, image_connect_cb);
}// end onInit

/* reconfigureCallback
*
* Callback for updating parameters with Dynamic Reconfigure
*/
void LaneDetector::reconfigureCallback(ghost::LaneDetectorConfig &config, uint32_t level) {	
	if(use_imu_ && !config.use_imu)
		imu_effect_reset_ = true;
	
	use_imu_ = config.use_imu;
		
	apply_gauss_ = config.apply_gauss;
	gauss_k_size_ = config.gauss_k_size;
	gauss_sigma_x_ = config.gauss_sigma_x;
	gauss_sigma_y_ = config.gauss_sigma_y;
	
	canny_type_ = config.canny_threshold;
	canny_lower_ = config.canny_lower;
	canny_upper_ = config.canny_upper;
	
	lane_centre_alpha_ = config.lane_centre_alpha;
}

/* imuCallback
*
* Callback for IMU messages to account for changes in camera pitch and roll.
* When the IMU is not used the transformations will need to be reset once.
*/
void LaneDetector::imuCallback(const sensor_msgs::Imu::ConstPtr &msg) {
	if(initialized_) {
		if(use_imu_){
			// Convert from geometry_msgs to tf quaternion
			tf::Quaternion quat;
			tf::quaternionMsgToTF(msg->orientation, quat);
			
			// Convert from quaternion to roll-pitch-yaw
			double delta_roll, delta_pitch, yaw_dud;
			tf::Matrix3x3(quat).getRPY(delta_roll, delta_pitch, yaw_dud);
							
			// Repeat the formation of the intermediate transforms from initializeTransforms, 
			// accounting for changes int the pitch and roll, and using the static yaw angle
			// in Rz_cw_. See initializeTransforms for more comments.
			const double gamma = -(PI/2.0 + cam_pitch_ + delta_pitch);
			const cv::Mat Rx_cw = (cv::Mat_<double>(3, 3) <<
				1.0,     0.0,        0.0,
				0.0, cos(gamma), -sin(gamma),
				0.0, sin(gamma),  cos(gamma));
			
			const double beta = cam_roll_ + delta_roll;
			const cv::Mat Ry_cw = (cv::Mat_<double>(3, 3) <<
				 cos(beta), 0.0, sin(beta),
						0.0,    1.0,    0.0,
				-sin(beta), 0.0, cos(beta));
			
			const cv::Mat R_cw = Rz_cw_*Ry_cw*Rx_cw;
			
			A_cw_ = R_cw*K_i_;
			A_wc_ = K_*R_cw.t();
		}else if(imu_effect_reset_) {
			// Reset the transformations to their static value
			
			const double gamma = -(PI/2.0 + cam_pitch_);
			const cv::Mat Rx_cw = (cv::Mat_<double>(3, 3) <<
				1.0,     0.0,        0.0,
				0.0, cos(gamma), -sin(gamma),
				0.0, sin(gamma),  cos(gamma));
			
			const double beta = cam_roll_;
			const cv::Mat Ry_cw = (cv::Mat_<double>(3, 3) <<
				 cos(beta), 0.0, sin(beta),
						0.0,    1.0,    0.0,
				-sin(beta), 0.0, cos(beta));
			
			const cv::Mat R_cw = Rz_cw_*Ry_cw*Rx_cw;
			
			A_cw_ = R_cw*K_i_;
			A_wc_ = K_*R_cw.t();
			
			imu_effect_reset_ = false;
		}
	}
}

/* connectCb
*
* Handles (un)subscribing when clients (un)subscribe
*/
void LaneDetector::connectCb() {
	if(pub_pc_.getNumSubscribers() == 0 &&
		 pub_pc_vis_.getNumSubscribers() == 0 &&
		 pub_laserscan_.getNumSubscribers() == 0 &&
		 pub_lane_pts_.getNumSubscribers() == 0 &&
		 pub_edges_.getNumSubscribers() == 0 &&
		 pub_bev_.getNumSubscribers() == 0 && 
		 pub_histo_.getNumSubscribers() == 0 &&
		 pub_otsu_.getNumSubscribers() == 0) {
		NODELET_INFO("No listeners, shutting down camera subscriber.");
		sub_camera_.shutdown();
		sub_imu_.shutdown();
	}else if (!sub_camera_) {
		NODELET_INFO("Subscribing to camera topic.");
		image_transport::TransportHints hints("raw", ros::TransportHints(), getPrivateNodeHandle());
		sub_camera_ = it_->subscribeCamera("image_rect_mono", 1, &LaneDetector::imageCb, this, hints);
		sub_imu_ = nh_.subscribe(imu_topic_.c_str(), 50, &LaneDetector::imuCallback, this);
	}
}

/* imageCb
*
* Performs all processing on the image to find the lane edges.
*
*/
void LaneDetector::imageCb(const sensor_msgs::ImageConstPtr& image_msg, const sensor_msgs::CameraInfoConstPtr& info_msg){
	// Make sure the BEV transform and image scan points are initialized
	if (!initialized_){
		// Need tf data to get camera height and angle
		// (transform assumed to be constant)
		std::string camera_frame = image_msg->header.frame_id;
		try {
			tf_listener_.waitForTransform("/base_footprint", "/" + camera_frame, image_msg->header.stamp, ros::Duration(5.0));
			tf_listener_.lookupTransform("/base_footprint", "/" + camera_frame, ros::Time(0), cam_tf_);
			cam_Z_ = cam_tf_.getOrigin().z();
			tf::Quaternion cam_quat = cam_tf_.getRotation();
			tf::Matrix3x3(cam_quat).getRPY(cam_roll_, cam_pitch_, cam_yaw_);
			NODELET_DEBUG("Got tf data: cam_Z_=%.2f, cam_roll_=%.2f, cam_pitch_=%.2f, cam_yaw_=%.2f", cam_Z_, cam_roll_, cam_pitch_, cam_yaw_);
		}catch (tf::TransformException &ex) {
			NODELET_WARN("Waiting for tf data between /base_footprint and /%s.", camera_frame.c_str());
			return;
		}
		
		bool T_init = initializeTransforms(info_msg);
		bool D_init = initializeDetector();
		initialized_ = T_init && D_init;
	}
	
	// Get the image
	cv::Mat img;
	try {
		img = cv_bridge::toCvShare(image_msg, sensor_msgs::image_encodings::MONO8)->image;
	} catch (cv_bridge::Exception& e) {
		NODELET_WARN("cv_bridge exception: %s", e.what());
		return;
	}
	
	// Perform any image debugging
	if(pub_bev_.getNumSubscribers() ||
		 pub_histo_.getNumSubscribers() ||
		 pub_otsu_.getNumSubscribers())
		imageDebugging(img, image_msg->header);
	
	// Perform lane detection
	if(pub_pc_.getNumSubscribers() ||
		 pub_pc_vis_.getNumSubscribers() ||
		 pub_laserscan_.getNumSubscribers() ||
		 pub_lane_pts_.getNumSubscribers() ||
		 pub_edges_.getNumSubscribers())
		detectLanes(img, image_msg->header);
}// end imageCb

/* initializeTransforms
*
* Computes intermediate transforms necessary for converting image pixels to
* world coordinates, and world coordinates to image pixels.
*
* Also computes the homography transform to produce a Bird's Eye View image
* (based on the min and max range of the camera/scan)
*/
bool LaneDetector::initializeTransforms(const sensor_msgs::CameraInfoConstPtr& info_msg) {
	if(info_msg->K[0] == 0 || info_msg->width == 0 || info_msg->height == 0)
		return false;
	
	// Set camera properties
	if(info_msg->roi.width == 0 && info_msg->roi.height == 0) {
		cam_w_ = info_msg->width;
		cam_h_ = info_msg->height;
	}else {
		cam_w_ = info_msg->roi.width;
		cam_h_ = info_msg->roi.height;
	}
	cam_y_offset_ = info_msg->roi.y_offset;
	cam_x_offset_ = info_msg->roi.x_offset;
	NODELET_DEBUG("ROI: %dx%d, x_offset=%d, y_offset=%d", cam_w_, cam_h_, cam_x_offset_, cam_y_offset_);
	
	// Form intrinsic matrix, and invert it
	K_ = (cv::Mat_<double>(3,3) <<
		info_msg->K[0], info_msg->K[1], info_msg->K[2],
		info_msg->K[3], info_msg->K[4], info_msg->K[5],
		info_msg->K[6], info_msg->K[7], info_msg->K[8]);
	K_i_ = K_.inv();
	
	// Form transformation expressing camera wrt world coordinates
	// (considering world origin as directly below the camera)
	t_cw_ = (cv::Mat_<double>(3,1) << 0.0, 0.0, cam_Z_);
	
	// Form rotation matrics to convert from car camera frame, to the optical
	// frame, accounting for camera roll, pitch, and yaw
	const double gamma = -(PI/2.0 + cam_pitch_);
	const cv::Mat Rx_cw = (cv::Mat_<double>(3, 3) <<
		1.0,     0.0,        0.0,
		0.0, cos(gamma), -sin(gamma),
		0.0, sin(gamma),  cos(gamma));
	
	const double beta = cam_roll_;
	const cv::Mat Ry_cw = (cv::Mat_<double>(3, 3) <<
		 cos(beta), 0.0, sin(beta),
				0.0,    1.0,    0.0,
		-sin(beta), 0.0, cos(beta));
	
	const double alpha = -PI/2.0 + cam_yaw_;
	Rz_cw_ = (cv::Mat_<double>(3, 3) <<
		cos(alpha), -sin(alpha), 0.0,
		sin(alpha), cos(alpha),  0.0,
				0.0,        0.0,     1.0);
	
	// Precompute rotation matrix for camera wrt world frame
	const cv::Mat R_cw = Rz_cw_*Ry_cw*Rx_cw;
	
	// Precompute intermediate transformations where R_cw is the rotation of the camera
	// optical frame wrt world camera frame, and K is the intrinsic matric
	A_cw_ = R_cw*K_i_;    // For converting pixel coords to world coords
	A_wc_ = K_*R_cw.t();  // For converting world coords to pixel coords
	
	/* Homography is be determined from the max range and minimum possible camera depth.
	*  It is assumed that the yaw and roll angles are zero.
	*
	*    Points on Image Plane      Points in world frame
	*                                  _____________
	*       _____________              *           *
	*       |           |               \         /
	*       *           *                \       /
	*       |           |                 \     /
	*       |           |                  \   /
	*  *    -------------    *              *-*
	*/
	// Find pixel height at max depth (at centre)
	const cv::Point2f TC_w = cv::Point2f(max_range_, 0.0);
	const cv::Point2f TC_p = convertWorldToPixel(TC_w);
	const double v_max_range = TC_p.y;
	
	// Form top left and right points
	const cv::Point2f TL_p = cv::Point2f(cam_x_offset_ + 1, v_max_range);
	const cv::Point2f TR_p = cv::Point2f(cam_w_ + cam_x_offset_, v_max_range);
	
	// Find left and right world widths at max range
	const cv::Point2f TL_w = convertPixelToWorld(TL_p);
	const cv::Point2f TR_w = convertPixelToWorld(TR_p);
	cam_max_width_ = TL_w.y - TR_w.y;
	NODELET_DEBUG("Camera view width at max range: %.3fm", cam_max_width_);
			
	// Find world depth at bottom pixel (at centre)
	const cv::Point2f BC_p = cv::Point2f(K_.at<double>(0,2), cam_h_ + cam_y_offset_);
	const cv::Point2f BC_w = convertPixelToWorld(BC_p);
	cam_min_depth_ = BC_w.x;
	NODELET_DEBUG("Camera minimum depth: %.3fm", cam_min_depth_);
	if(min_range_ < cam_min_depth_){
		min_range_ = std::max(min_range_, cam_min_depth_);
		NODELET_INFO("min_range parameter less than minimum possible camera depth. Increasing to %.3f", min_range_);
	}
	
	// Find pixel coords at min depth and world width at max range
	const cv::Point2f BR_w = cv::Point2f(cam_min_depth_, TR_w.y);
	const cv::Point2f BL_w = cv::Point2f(cam_min_depth_, TL_w.y);
	const cv::Point2f BR_p = convertWorldToPixel(BR_w);
	const cv::Point2f BL_p = convertWorldToPixel(BL_w);
			
	// Form source points vector (Order: TL, TR, BR, BL, and form [col, row])
	std::vector<cv::Point2f> src(4);
	src[0] = TL_p;
	src[1] = TR_p;
	src[2] = BR_p;
	src[3] = BL_p;
	
	// Form destination points vector (Order: TL, TR, BR, BL, and form [col, row])
	// Scaled according to pixels-per-meter
	bev_w_ = cam_max_width_*bev_ppm_;
	bev_h_ = (max_range_ - cam_min_depth_)*bev_ppm_;
	
	std::vector<cv::Point2f> dst(4);
	dst[0] = cv::Point2f(1, 1);
	dst[1] = cv::Point2f(bev_w_, 1);
	dst[2] = cv::Point2f(bev_w_, bev_h_);
	dst[3] = cv::Point2f(1, bev_h_);
	
	NODELET_DEBUG("BEV image size: %dx%d", static_cast<int>(bev_w_), static_cast<int>(bev_h_));
	NODELET_DEBUG("BEV Points (in the order TL, TR, BR, BL):");
	NODELET_DEBUG("SRC: [%.2f, %.2f], [%.2f, %.2f], [%.2f, %.2f], [%.2f, %.2f]", src[0].x, src[0].y, src[1].x, src[1].y, src[2].x, src[2].y, src[3].x, src[3].y);
	NODELET_DEBUG("DST: [%.2f, %.2f], [%.2f, %.2f], [%.2f, %.2f], [%.2f, %.2f]", dst[0].x, dst[0].y, dst[1].x, dst[1].y, dst[2].x, dst[2].y, dst[3].x, dst[3].y);
	
	// Compute the homography
	H_bev_ = cv::getPerspectiveTransform(src, dst);
	
	// Calculate the field of view
	cam_FOV_ = 2*atan2(0.5*cam_max_width_, max_range_);
	NODELET_DEBUG("Camera FOV: %.3f deg", (cam_FOV_)*180.0/PI);
	
	NODELET_INFO("Initialized the camera transformations");
	return true;
}// end initializeTransforms

/* initializeDetector
* 
* Finds which rows to scan in the image based on the min and max range, and the
* desired spatial resolution. Also fills in portions of the pointcloud messages
* that will not change.
*/
bool LaneDetector::initializeDetector() {			
	// Find which rows to scan
	const int num_pts = std::floor((max_range_ - min_range_)/detect_dx_);
	double x = min_range_;
	NODELET_DEBUG("%d pixel rows will be scanned.", num_pts);
	for(int i = 0; i < num_pts; i++){
		// Find pixel coord
		const cv::Point2f pt_w = cv::Point2f(x, 0);
		const cv::Point2f pt_p = convertWorldToPixel(pt_w);
		scan_rows_.push_back(pt_p.y);
		
		NODELET_DEBUG("%d) Depth=%.2f, Pixel Row=%.1f", (i + 1), x, pt_p.y);
		x += detect_dx_;
	}
	
	// Fill in constant sections of messages
	pc_msg_.header.frame_id = detect_frame_id_;
	pc_msg_.width = 2;
	pc_msg_.height = scan_rows_.size();
	pc_msg_.points.resize(pc_msg_.width*pc_msg_.height);
	
	pc_vis_msg_.header.frame_id = detect_frame_id_;
	pc_vis_msg_.width = 2;
	pc_vis_msg_.height = scan_rows_.size();
	pc_vis_msg_.points.resize(pc_vis_msg_.width*pc_vis_msg_.height);
	
	const int laserscan_res = 100;
	laserscan_msg_.header.frame_id = detect_frame_id_;
	laserscan_msg_.angle_min = -cam_FOV_/2 - 0.1*PI;
	laserscan_msg_.angle_max = cam_FOV_/2 + 0.1*PI;
	laserscan_msg_.angle_increment = (laserscan_msg_.angle_max - laserscan_msg_.angle_min)/(laserscan_res - 1);
	laserscan_msg_.time_increment = 0.0;
	laserscan_msg_.range_min = min_range_;
	laserscan_msg_.range_max = max_range_/cos(1.1*cam_FOV_/2);  // Bump up a bit to account for distortion
	laserscan_msg_.ranges.resize(laserscan_res);
	
	NODELET_INFO("Detector initialized.");
	return true;
}

/* detectLanes
* 
* Applys a gaussian filter to smooth the image, then performs Canny edge detection.
* The resulting image is scanned at pre defined rows to detect the left and right
* lane edges. The centreline from which the scans start is updated with an exponential
* moving average of the left and right detections. The world coordinates of the 
* edge points are then computed, and published either as a 2D pointcloud or a
* laser scan message.
*
* Also has capability to publish an image with the detected lanes points with
* the lane centreline (from which the scans start), and a 3D pointcloud to be
* viewed with RVIZ.
*/
void LaneDetector::detectLanes(cv::Mat &img, const std_msgs::Header &header) {
	// What needs to be published
	const bool edges_requested = pub_edges_.getNumSubscribers();
	const bool lane_points_requested = pub_lane_pts_.getNumSubscribers();
	const bool pc_requested = pub_pc_.getNumSubscribers();
	const bool pc_vis_requested = pub_pc_vis_.getNumSubscribers();
	const bool laserscan_requested = pub_laserscan_.getNumSubscribers();

	// Gaussian smoothing
	if(apply_gauss_) {
		cv::GaussianBlur(img, img, cv::Size(gauss_k_size_, gauss_k_size_), gauss_sigma_x_, gauss_sigma_y_);
	}
	
	// Apply Canny edge detection
	if(canny_type_ == OTSU) {
		// Apply Otsu thesholding
		cv::Mat otsu_img;
		const double otsu_thresh = cv::threshold(img, otsu_img, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
		cv::Canny(img, img, 0.5*otsu_thresh, otsu_thresh);
	}else if(canny_type_ == MEDIAN) {
		// Use +- 33% of the median of pixels for threshold values
		
		// Compute histogram of image
		const int num_bins = 256;
		float range[] = {0, num_bins};
		const float* hist_range = {range};
		const bool uniform = true;
		const bool accumulate = false;
		cv::Mat hist;
		cv::calcHist(&img, 1, 0, cv::Mat(), hist, 1, &num_bins, &hist_range, uniform, accumulate);
		
		int bin = 0;
		double median = -1.0;
		const double m = (img.rows*img.cols)/2;
		for(int i = 0; i < num_bins && median < 0.0; ++i) {
			bin += cvRound(hist.at<float>(i));
			if(bin > m && median < 0.0)
				median = i;
		}
		
		cv::Canny(img, img, 0.66*median, 1.33*median);
	}else if(canny_type_ == MEAN) {
		// Use +- 33% of the mean of all pixels for threshold values
		cv::Scalar temp_mean = cv::mean(img);
		float mean = temp_mean.val[0];
		cv::Canny(img, img, 0.66*mean, 1.33*mean);
	}else if(canny_type_ == MANUAL) {
		cv::Canny(img, img, canny_lower_, canny_upper_);
	}
	
	// Publish the edges image (if requested)
	if(edges_requested) {
		sensor_msgs::ImagePtr edges_msg = cv_bridge::CvImage(header, sensor_msgs::image_encodings::MONO8, img).toImageMsg();
		pub_edges_.publish(edges_msg);
	}
	
	if(!lane_points_requested && !pc_requested && !pc_vis_requested && !laserscan_requested)
		return;
	
	// Detect lane points in image by scanning specified rows to the left and right
	// looking for an edge pixel, then convert to a pointcloud and publish. The 
	// centre point in the row from which the left and right scans starts will change
	// as a moving average of the detected centreline
	
	// Necessary variables
	int index;                            // Current index in scan_rows_
	int left_n, right_n;                  // Left and right pixel hits in the current row
	std::vector<cv::Point> left_pixels(scan_rows_.size(), cv::Point(0, 0));   // For storing left lane pixels
	std::vector<cv::Point> right_pixels(scan_rows_.size(), cv::Point(0, 0));  // For storing right lane pixels
	std::vector<cv::Point> centre_pixels(scan_rows_.size(), cv::Point(0, 0)); // For storing centreline pixels
	std::vector<cv::Point2f> left_world(scan_rows_.size(), cv::Point(0, 0));  // For storing left lane world coords
	std::vector<cv::Point2f> right_world(scan_rows_.size(), cv::Point(0, 0)); // For storing lerightft lane world coords
	
	double start_col = (cam_w_ - 1)/2;    // Initialize which column to start from
	
	std::vector<int>::iterator row_it;
	for(index = 0, row_it = scan_rows_.begin(); row_it < scan_rows_.end(); ++row_it, index++) {
		const unsigned char* row = img.ptr<unsigned char>(*row_it);
		
		// Iterate left for lane hits
		for(left_n = start_col - 1; left_n >= 0; left_n--) {
			if(row[left_n] == 255) {
				// Save pixel and world values
				left_pixels[index] = cv::Point(left_n, *row_it);
				left_world[index] = convertPixelToWorld(left_pixels[index]);
				break;
			}
		}
		
		// Iterate right for lane hits
		for(right_n = start_col + 1; right_n < cam_w_; right_n++) {
			if(row[right_n] == 255) {
				// Save pixel and world values
				right_pixels[index] = cv::Point(right_n, *row_it);
				right_world[index] = convertPixelToWorld(right_pixels[index]);
				break;
			}
		}
		
		// Update the start column
		const double prev_mid_pt = (left_n + right_n)/2.0;
		start_col = lane_centre_alpha_*start_col + (1 - lane_centre_alpha_)*prev_mid_pt;
		centre_pixels[index] = cv::Point(start_col, *row_it);
	}
	
	// Publish the 2D pointcloud (if requested)
	if(pc_requested) {
		// Need to convert to XY pointcloud
		for(index = 0; index < scan_rows_.size(); index++) {
			// Convert left lane points
			if(left_world[index].x != 0.0 && left_world[index].y != 0.0) {
				pc_msg_.points[pc_msg_.width*index].x = left_world[index].x;
				pc_msg_.points[pc_msg_.width*index].y = left_world[index].y;
			}else {
				pc_msg_.points[pc_msg_.width*index].x = std::numeric_limits<double>::infinity();
				pc_msg_.points[pc_msg_.width*index].y = std::numeric_limits<double>::infinity();
			}
			
			// Convert right lane points
			if(right_world[index].x != 0.0 && right_world[index].y != 0.0) {
				pc_msg_.points[pc_msg_.width*index + 1].x = right_world[index].x;
				pc_msg_.points[pc_msg_.width*index + 1].y = right_world[index].y;
			}else {
				pc_msg_.points[pc_msg_.width*index + 1].x = std::numeric_limits<double>::infinity();
				pc_msg_.points[pc_msg_.width*index + 1].y = std::numeric_limits<double>::infinity();
			}
		}
		
		pcl_conversions::toPCL(header.stamp, pc_msg_.header.stamp);
		pub_pc_.publish(pc_msg_);
	}
	
	// Publish the pointcloud for visualization (if requested)
	if(pc_vis_requested) {
		// Need to convert to XYZ pointcloud for RVIZ
		for(index = 0; index < scan_rows_.size(); index++) {
			// Convert left lane points
			if(left_world[index].x != 0.0 && left_world[index].y != 0.0) {
				pc_vis_msg_.points[pc_vis_msg_.width*index].x = left_world[index].x;
				pc_vis_msg_.points[pc_vis_msg_.width*index].y = left_world[index].y;
			}else {
				pc_vis_msg_.points[pc_vis_msg_.width*index].x = std::numeric_limits<double>::infinity();
				pc_vis_msg_.points[pc_vis_msg_.width*index].y = std::numeric_limits<double>::infinity();
			}
			
			// Convert right lane points
			if(right_world[index].x != 0.0 && right_world[index].y != 0.0) {
				pc_vis_msg_.points[pc_vis_msg_.width*index + 1].x = right_world[index].x;
				pc_vis_msg_.points[pc_vis_msg_.width*index + 1].y = right_world[index].y;
			}else {
				pc_vis_msg_.points[pc_vis_msg_.width*index + 1].x = std::numeric_limits<double>::infinity();
				pc_vis_msg_.points[pc_vis_msg_.width*index + 1].y = std::numeric_limits<double>::infinity();
			}
		}
		
		pcl_conversions::toPCL(header.stamp, pc_vis_msg_.header.stamp);
		pub_pc_vis_.publish(pc_vis_msg_);
	}
	
	// Publish the laserscan points (if requested)
	if(laserscan_requested) {
		// Requires binning the pts into scan rays based on their angle
		// (will lose some accuracy, but with enough rays it should be alright)
		
		// Default to infinity
		std::fill(laserscan_msg_.ranges.begin(), laserscan_msg_.ranges.end(), std::numeric_limits<double>::infinity());
		
		// Convert right lane points
		for(index = 0; index < scan_rows_.size(); index++) {
			if(right_world[index].x != 0.0 && right_world[index].y != 0.0) {
				const double angle = atan2(right_world[index].y, right_world[index].x);
				const int ray_index = round((angle - laserscan_msg_.angle_min)/laserscan_msg_.angle_increment);
				laserscan_msg_.ranges[ray_index] = sqrt(pow(right_world[index].x, 2) + pow(right_world[index].y, 2));
			}
		}
		
		// Convert left lane points
		for(index = 0; index < scan_rows_.size(); index++) {
			if(left_world[index].x != 0.0 && left_world[index].y != 0.0) {
				const double angle = atan2(left_world[index].y, left_world[index].x);
				const int ray_index = round((angle - laserscan_msg_.angle_min)/laserscan_msg_.angle_increment);
				laserscan_msg_.ranges[ray_index] = sqrt(pow(left_world[index].x, 2) + pow(left_world[index].y, 2));
			}
		}
		
		laserscan_msg_.header.stamp = header.stamp;
		pub_laserscan_.publish(laserscan_msg_);
	}
	
	// Publish the lane points on an image (if requested)
	if(lane_points_requested) {
		cv::Mat pts_img = cv::Mat::zeros(cam_h_, cam_w_, CV_8UC1);
		
		// Draw circles for all the points
		std::vector<cv::Point>::iterator point_it;
		for(point_it = left_pixels.begin() ; point_it < left_pixels.end(); ++point_it) {
			if(point_it->x != 0 && point_it->y != 0) {
				cv::circle(pts_img, *point_it, 1, 255);
			}
		}
		for(point_it = right_pixels.begin() ; point_it < right_pixels.end(); ++point_it) {
			if(point_it->x != 0 && point_it->y != 0) {
				cv::circle(pts_img, *point_it, 1, 255);
			}
		}
		for(point_it = centre_pixels.begin() ; point_it < centre_pixels.end(); ++point_it) {
			cv::circle(pts_img, *point_it, 5, 255, -1);
		}
		
		// Publish the image
		sensor_msgs::ImagePtr pts_msg = cv_bridge::CvImage(header, sensor_msgs::image_encodings::MONO8, pts_img).toImageMsg();
		pub_lane_pts_.publish(pts_msg);
	}
}

/* imageDebugging
* 
* Performs additional operations on the input image and publishes the results.
* Intended for debugging and observing image characteristics.
*
* Currently performs:
*   -Birds eye view projection
*   -Histogram of pixel itnensities
*   -Otsu threshold
*/
void LaneDetector::imageDebugging(cv::Mat &img, const std_msgs::Header &header) {	
	// Birds Eye View projection
	if (pub_bev_.getNumSubscribers()) {
		cv::Mat bev_img;
		cv::warpPerspective(img, bev_img, H_bev_, cv::Size(bev_w_, bev_h_), cv::BORDER_CONSTANT);
		
		sensor_msgs::ImagePtr bev_msg;
		bev_msg = cv_bridge::CvImage(header, sensor_msgs::image_encodings::MONO8, img).toImageMsg();
		pub_bev_.publish(bev_msg);
	}
	
	// Image histogram
	if(pub_histo_.getNumSubscribers()) {
		// Establish the number of bins
		int hist_size = 256;
		
		// Set the ranges
		float range[] = {0, 256};
		const float* hist_range = {range};
		
		// Compute the histogram
		cv::Mat hist;
		bool uniform = true; 
		bool accumulate = false;
		cv::calcHist(&img, 1, 0, cv::Mat(), hist, 1, &hist_size, &hist_range, uniform, accumulate);
		
		// Make the output image
		int hist_w = 512;
		int hist_h = 400;
		cv::Mat hist_img = cv::Mat::zeros(hist_h, hist_w, CV_8UC1);
		
		// Normalize the result to [0, hist_img.rows]
		cv::normalize(hist, hist, 0, hist_img.rows, cv::NORM_MINMAX, -1, cv::Mat());
		
		// Draw histogram lines
		int bin_w = round((double)hist_w/hist_size );
		for(int i = 0; i < hist_size; i++) {
				cv::line(hist_img, cv::Point(bin_w*i, hist_h - round(hist.at<float>(i))),
									cv::Point(bin_w*(i), hist_h), 255, 2, 8, 0);
		}
		
		sensor_msgs::ImagePtr histo_msg = cv_bridge::CvImage(header, sensor_msgs::image_encodings::MONO8, hist_img).toImageMsg();
		pub_histo_.publish(histo_msg);
	}
	
	// Otsu threshold image
	if(pub_otsu_.getNumSubscribers()) {
		cv::Mat otsu_img;
		cv::threshold(img, otsu_img, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
		sensor_msgs::ImagePtr otsu_msg = cv_bridge::CvImage(header, sensor_msgs::image_encodings::MONO8, otsu_img).toImageMsg();
		pub_otsu_.publish(otsu_msg);
	}
}
	
/* convertPixelToWorld
* 
* Converts pixel positions to world coordinates using the precomputed
* transform and camera height. World X and Y is relative to camera origin.
*/
cv::Point2f LaneDetector::convertPixelToWorld(const cv::Point2f &pixel) {
	// Form pixel coordinate
	const cv::Mat p_p = (cv::Mat_<double>(3, 1) << pixel.x + cam_x_offset_, pixel.y + cam_y_offset_, 1.0);
	
	// Get intermediate point
	const cv::Mat p_i = A_cw_*p_p;
	
	// Solve for scaling factor
	const double s = -cam_Z_/p_i.at<double>(2,0);
	
	// Solve for world coordinates
	const double x = s*p_i.at<double>(0,0);
	const double y = s*p_i.at<double>(1,0);
	
	const cv::Point2f p_w = cv::Point2f(x, y);
	return p_w;
}

/* convertWorldToPixel
* 
* Converts world coordinates to pixel positions using the precomputed
* transform and camera height. World X and Y is relative to camera origin.
*/
cv::Point2f LaneDetector::convertWorldToPixel(const cv::Point2f &p) {
	// Form world coordinate
	const cv::Mat p_w = (cv::Mat_<double>(3, 1) << p.x, p.y, 0.0);
	
	// Get non-homogeneous pixel coordinate
	const cv::Mat p_i = A_wc_*(p_w - t_cw_);
	
	// Solve for scaling factor
	const double s = 1.0/p_i.at<double>(2,0);
	
	// Solve for homogeneous pixel coordinates
	const double u = s*p_i.at<double>(0,0) - cam_x_offset_;
	const double v = s*p_i.at<double>(1,0) - cam_y_offset_;
	
	const cv::Point2f p_p = cv::Point2f(u, v);
	return p_p;
}

}// end ghost namespace 

// Register the nodelet
PLUGINLIB_DECLARE_CLASS(ghost, LaneDetector, ghost::LaneDetector, nodelet::Nodelet);