/* LaneDetector Nodelet
*
* Images from the camera are processed to detect edges (i.e. the edges
* of the floor), the pixel coordinates of which are then projected 
* into real world coordinates.
*
* The resulting edges are then scanned as if a laser scanner were placed
* at the camera's optical origin, where the first edge detected by a ray 
* is returned as hit. The result is then published as a LaserScan message.
*
* The camera pose is determined from the TF frames. Based on the pose, the
* transformations for pixel-to-world, world-to-pixel, and homography, are
* precomputed.
*
* ----------------------------------------------------------------------------------
*/

#include <ros/ros.h>
#include <ros/console.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <dynamic_reconfigure/server.h>
#include <ghost/LaneDetectorConfig.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/LaserScan.h>
#include <std_msgs/Header.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>

#define PI 3.14159265

namespace ghost {

class LaneDetector : public nodelet::Nodelet {
	// Image transport
	boost::shared_ptr<image_transport::ImageTransport> it_;
	boost::shared_ptr<image_transport::ImageTransport> private_it_;

	// Pubs, Subs, and tf
	image_transport::CameraSubscriber sub_camera_;  // Camera
	ros::Publisher pub_scan_;                       // Laserscan message
	image_transport::Publisher pub_bev_;            // Bird's eye view image
	image_transport::Publisher pub_edges_;          // Image with edges
	image_transport::Publisher pub_histo_;          // Image histogram
	image_transport::Publisher pub_otsu_;           // Image with Otsu thresholding
	tf::TransformListener tf_listener_;             // Camera transform
		
	// Camera properties
	tf::StampedTransform cam_tf_;  // Most recent transform between camera and base_footprint
	double cam_Z_;                 // Height of camera from ground
	double cam_roll_;              // Angle of camera (about the X axis)
	double cam_pitch_;             // Angle of camera (about the Y axis)
	double cam_yaw_;               // Angle of camera (about the Z axis)
	int cam_w_;                    // Width of camera image
	int cam_h_;                    // Height of camera image
	int cam_y_offset_;             // Offset from top pixel for ROI
	double cam_FOV_;               // Field of View
	double cam_max_width_;         // World width at max scan range
	double cam_min_depth_;         // World width at bottom pixel row
	cv::Mat A_cw_;                 // Intermediate transformation [R_cw][K^-1]
	cv::Mat A_wc_;                 // Intermediate transformation [R_cw^-1][K_i]	
	
	// Bird's Eye View image properties
	bool use_BEV_image_;  // Flag if the BEV image should be used to edge detection and scanning
	int bev_w_;           // Image width
	int bev_h_;           // Image height
	int bev_ppm_;         // Pixel-per-meter for transformed image
	cv::Mat H_bev_;       // BEV transformation matrix		
	
	// Image filtering parameters 
	enum canny_thresh_enum {OTSU, MEDIAN, MEAN};
	bool apply_clahe_;              // Flag for applying CLAHE histogram equalization
	double clahe_clip_limit_;       // Clipping limit for CLAHE histogram equalization
	bool apply_gauss_;              // Flag for applying Gaussian filter
	int gauss_k_size_;              // Size of kernal for Gaussian blur
	double gauss_sigma_x_;          // Gaussian kernel standar deviation in the X direction
	double gauss_sigma_y_;          // Gaussian kernel standar deviation in the Y direction
	bool apply_bilateral_;          // Flag for applying bilateral filter
	int bilateral_k_size_;          // Size of kernel for bilateral filter
	double bilateral_sigma_color_;  // Filter sigma in the color space
	double bilateral_sigma_space_;  // Filter sigma in the coordinate space
	int canny_thresh_;              // Enum for canny threshold calculation method
	bool apply_vert_edge_;          // Flag if vertical edges should be removed
	
	// Scaner properties
	double max_range_;                        // Maximum range of scan rays
	double min_range_;                        // Minimum range of scan rays
	int scan_res_;                            // Number of scan points within FOV
	std::string scan_frame_id_;               // Name of the frame to attach to the scan message
	std::vector<cv::Point2f> scan_start_pts_; // Pixel start points of each scan ray
	std::vector<cv::Point2f> scan_end_pts_;   // Pixel end points of each scan ray
	sensor_msgs::LaserScan scan_msg_;         // Message containing scan to publish
	
	// Miscellaneous
	bool initialized_;   // Flag for if the transforms and scanner points have been initialized
	
	// Method declarations
	void onInit();
	void reconfigureCallback(ghost::LaneDetectorConfig &config, uint32_t level);
	void connectCb();
	void imageCb(const sensor_msgs::ImageConstPtr& image_msg, const sensor_msgs::CameraInfoConstPtr& info_msg);
	void initializeTransforms(const sensor_msgs::CameraInfoConstPtr& info_msg);
	void initializeScanner();
	void getScanWorldPoints(std::vector<cv::Point2f> &start, std::vector<cv::Point2f> &end);
	void detectEdges(cv::Mat &img, const std_msgs::Header &header);
	void imageDebugging(cv::Mat &img, const std_msgs::Header &header);
	void scanImage(cv::Mat &img);
	cv::Point2f convertPixelToWorld(const cv::Point2f &pixel);
	cv::Point2f convertWorldToPixel(const cv::Point2f &p);
	
};// end LaneDetector class def

/* Initializer
*
*/
void LaneDetector::onInit(){
	// Get the nodehandles
	ros::NodeHandle &nh = getNodeHandle();
	ros::NodeHandle &pnh = getPrivateNodeHandle();

	// Setup image transport
	it_.reset(new image_transport::ImageTransport(nh));
	private_it_.reset(new image_transport::ImageTransport(pnh));

	// Monitor whether anyone is subscribed to the output
	ros::SubscriberStatusCallback scan_connect_cb = boost::bind(&LaneDetector::connectCb, this);
	image_transport::SubscriberStatusCallback image_connect_cb = boost::bind(&LaneDetector::connectCb, this);

	// Setup the publishers (linking them to connectCb)
	pub_scan_ = pnh.advertise<sensor_msgs::LaserScan>("scan", 50, scan_connect_cb, scan_connect_cb);
	pub_bev_  = private_it_->advertise("bev_image",  1, image_connect_cb, image_connect_cb);
	pub_edges_  = private_it_->advertise("edges_image",  1, image_connect_cb, image_connect_cb);
	pub_histo_  = private_it_->advertise("histogram",  1, image_connect_cb, image_connect_cb);
	pub_otsu_  = private_it_->advertise("otsu_image",  1, image_connect_cb, image_connect_cb);
	
	// Load configuration parameters from the parameter server
	bool debug;
	pnh.param<bool>("debug", debug, false);
	pnh.param<int>("bev_ppm", bev_ppm_, 100);
	pnh.param<bool>("use_BEV_image", use_BEV_image_, false);
	pnh.param<double>("min_range", min_range_, 0.5);
	pnh.param<double>("max_range", max_range_, 5.0);
	pnh.param<int>("scan_res", scan_res_, 100);
	pnh.param<std::string>("scan_frame_id", scan_frame_id_, "camera");
	
	// Setup dynamic reconfigure
	dynamic_reconfigure::Server<ghost::LaneDetectorConfig> *server;
	server = new dynamic_reconfigure::Server<ghost::LaneDetectorConfig>(pnh);
  dynamic_reconfigure::Server<ghost::LaneDetectorConfig>::CallbackType f;
  f = boost::bind(&LaneDetector::reconfigureCallback, this, _1, _2);
  server->setCallback(f);
	
	initialized_ = false;
	
	// Manage debugging logger level
	if(debug){
		if(ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug) ) {
			ros::console::notifyLoggerLevelsChanged();
		}
	}
	
}// end onInit

/* reconfigureCallback
*
* Callback for updating parameters with Dynamic Reconfigure
*/
void LaneDetector::reconfigureCallback(ghost::LaneDetectorConfig &config, uint32_t level) {
	apply_clahe_ = config.apply_clahe;
	clahe_clip_limit_ = config.clahe_clip_limit;
	
	apply_gauss_ = config.apply_gauss;
	gauss_k_size_ = config.gauss_k_size;
	gauss_sigma_x_ = config.gauss_sigma_x;
	gauss_sigma_y_ = config.gauss_sigma_y;
	
	apply_bilateral_ = config.apply_bilateral;
	bilateral_k_size_ = config.bilateral_k_size;
	bilateral_sigma_color_ = config.bilateral_sigma_color;
	bilateral_sigma_space_ = config.bilateral_sigma_space;
	
	canny_thresh_ = config.canny_threshold;
		
	apply_vert_edge_ = config.apply_vert_edge;
}

/* connectCb
*
* Handles (un)subscribing when clients (un)subscribe
*/
void LaneDetector::connectCb() {
	if (pub_scan_.getNumSubscribers() == 0 && 
		  pub_bev_.getNumSubscribers() == 0 && 
		  pub_edges_.getNumSubscribers() == 0 && 
		  pub_histo_.getNumSubscribers() == 0 &&
		  pub_otsu_.getNumSubscribers() == 0) {
		NODELET_INFO("No listeners, shutting down camera subscriber.");
		sub_camera_.shutdown();
	} else if (!sub_camera_) {
		NODELET_INFO("Subscribing to camera topic.");
		image_transport::TransportHints hints("raw", ros::TransportHints(), getPrivateNodeHandle());
		sub_camera_ = it_->subscribeCamera("image_rect_mono", 1, &LaneDetector::imageCb, this, hints);
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
		
		NODELET_INFO("Initializing detector.");
		initializeTransforms(info_msg);
		initializeScanner();
		initialized_ = true;
	}
	
	// Get the image
	cv::Mat img = cv_bridge::toCvShare(image_msg, sensor_msgs::image_encodings::MONO8)->image;
	
	if(use_BEV_image_)
		cv::warpPerspective(img, img, H_bev_, cv::Size(bev_w_, bev_h_), cv::BORDER_CONSTANT);
	
	// Publish the BEV image (if requested)
	if (pub_bev_.getNumSubscribers()) {
		sensor_msgs::ImagePtr bev_msg;
		if(!use_BEV_image_) {
			// Need to apply BEV transform
			cv::Mat bev_image;
			cv::warpPerspective(img, bev_image, H_bev_, cv::Size(bev_w_, bev_h_), cv::BORDER_CONSTANT);
			bev_msg = cv_bridge::CvImage(image_msg->header, sensor_msgs::image_encodings::MONO8, bev_image).toImageMsg();
		}else {
			bev_msg = cv_bridge::CvImage(image_msg->header, sensor_msgs::image_encodings::MONO8, img).toImageMsg();
		}
		pub_bev_.publish(bev_msg);
	}
	
	// Perform any image debugging
	if(pub_histo_.getNumSubscribers() || pub_otsu_.getNumSubscribers())
		imageDebugging(img, image_msg->header);
	
	// Only proceed if the scan or the edges image is subscribed to
	if(!pub_scan_.getNumSubscribers() && !pub_edges_.getNumSubscribers())
		return;
	
	// Get edges from the image
	detectEdges(img, image_msg->header);
		
	// Only proceed if the scan is subscribed to
	if(!pub_scan_.getNumSubscribers())
		return;
	
	// Scan image and publish
	scanImage(img);
	pub_scan_.publish(scan_msg_);
	
}// end imageCb

/* initializeTransforms
*
* Computes intermediate transforms necessary for converting image pixels to
* world coordinates, and world coordinates to image pixels.
*
* Also computes the homography transform to produce a Bird's Eye View image
* (based on the min and max range of the camera/scan)
*
*   Points on Image      Points in world frame
*        _____                 _______
*       *     *               *       *
*      /       \              |       |
*     /         \             |       |
*    *           *            *       *
*    |           |             \     /
*    |           |              \   /
*    *-----------*               *-*
*/
void LaneDetector::initializeTransforms(const sensor_msgs::CameraInfoConstPtr& info_msg) {
	NODELET_DEBUG("Initializing the camera transformations");
	
	// Set camera properties
	cam_w_ = info_msg->width;
	cam_h_ = info_msg->height;
	cam_y_offset_ = info_msg->roi.y_offset;
	
	// Form intrinsic matrix, and invert it
	const cv::Mat K = (cv::Mat_<double>(3,3) <<
		info_msg->K[0], info_msg->K[1], info_msg->K[2],
		info_msg->K[3], info_msg->K[4], info_msg->K[5],
		info_msg->K[6], info_msg->K[7], info_msg->K[8]);
	const cv::Mat K_i = K.inv();
	
	// Form transformation expressing camera wrt world coordinates
	// (considering world origin as directly below the camera)
	const cv::Mat t_cw = (cv::Mat_<double>(3,1) << 0.0, 0.0, cam_Z_);
	
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
				 0.0,   1.0,    0.0,
		-sin(beta), 0.0, cos(beta));
	
	const double alpha = -PI/2.0 + cam_yaw_;
	const cv::Mat Rz_cw = (cv::Mat_<double>(3, 3) <<
		cos(alpha), -sin(alpha), 0.0,
		sin(alpha), cos(alpha),  0.0,
				0.0,        0.0,     1.0);
	
	const cv::Mat R_cw = Rz_cw*Ry_cw*Rx_cw;
	
	// Precompute intermediate transformations where R_cw is the rotation of the camera
	// optical frame wrt world camera frame, and K is the intrinsic matric
	A_cw_ = R_cw*K_i;    // For converting pixel coords to world coords
	A_wc_ = K*R_cw.t();  // For converting world coords to pixel coords
	
	// Homography is be determined from the max range and minimum possible camera depth.
	// It is assumed that the yaw and roll angles are zero.
	
	// Find pixel height at max depth (at centre)
	const cv::Point2f TC_w = cv::Point2f(max_range_, 0.0);
	const cv::Point2f TC_p = convertWorldToPixel(TC_w);
	const double v_max_range = TC_p.y;
	
	// Form top left and right points
	const cv::Point2f TL_p = cv::Point2f(1, v_max_range);
	const cv::Point2f TR_p = cv::Point2f(cam_w_, v_max_range);
	
	// Find left and right world widths at max range
	const cv::Point2f TL_w = convertPixelToWorld(TL_p);
	const cv::Point2f TR_w = convertPixelToWorld(TR_p);
	cam_max_width_ = TL_w.y - TR_w.y;
	NODELET_DEBUG("Camera view width at max range: %.3fm", cam_max_width_);
			
	// Find world depth at bottom pixel (at centre)
	const cv::Point2f BC_p = cv::Point2f(cam_w_/2.0, cam_h_ - cam_y_offset_);
	const cv::Point2f BC_w = convertPixelToWorld(BC_p);
	cam_min_depth_ = BC_w.x;
	NODELET_DEBUG("Camera minimum depth: %.3fm", cam_min_depth_);
	if(min_range_ < cam_min_depth_){
		min_range_ = std::max(min_range_, cam_min_depth_);
		NODELET_INFO("min_range parameter less than minimum possible camera depth. Increasing to %.3f", min_range_);
	}
	
	// Find pixel coords and min depth and world width at max range
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
	
}// end initializeTransforms

/* initializeScanner
* 
* Initializes start and end points for scan rays in pixel coordinates for the
* camera image. Rays go from the camera origin to a depth of max_range_, and
* are equally divided across the camera FOV. Start and end points are first 
* found in world coordinates, then converted to pixel coordinates (and constrained 
* to be valid).
*/
void LaneDetector::initializeScanner() {
	NODELET_DEBUG("Initializing scanner.");
	
	// Get the scan points in the world frame
	std::vector<cv::Point2f> w_i, w_f;
	getScanWorldPoints(w_i, w_f);
	
	scan_start_pts_.clear();
	scan_end_pts_.clear();
	scan_start_pts_.resize(scan_res_);
	scan_end_pts_.resize(scan_res_);
	
	// Loop through points and convert to pixel coords
	NODELET_DEBUG("Scan pixel points:");
	if(use_BEV_image_){
		for(int i = 0; i < scan_res_; i++) {
			scan_start_pts_[i].x = 0.5*bev_w_ - w_i[i].y*bev_ppm_;
			scan_start_pts_[i].y = bev_h_ - (w_i[i].x - cam_min_depth_)*bev_ppm_;
			scan_start_pts_[i].x = std::max(1.0, std::min(ceil(scan_start_pts_[i].x), static_cast<double>(bev_w_)));
			scan_start_pts_[i].y = std::max(1.0, std::min(ceil(scan_start_pts_[i].y), static_cast<double>(bev_h_)));
			
			scan_end_pts_[i].x = 0.5*bev_w_ - w_f[i].y*bev_ppm_;
			scan_end_pts_[i].y = bev_h_ - (w_f[i].x - cam_min_depth_)*bev_ppm_;
			scan_end_pts_[i].x = std::max(1.0, std::min(ceil(scan_end_pts_[i].x), static_cast<double>(bev_w_)));
			scan_end_pts_[i].y = std::max(1.0, std::min(ceil(scan_end_pts_[i].y), static_cast<double>(bev_h_)));
			NODELET_DEBUG("Start: [%.1f, %.1f], End: [%.1f, %.1f]", scan_start_pts_[i].x, scan_start_pts_[i].y, scan_end_pts_[i].x, scan_end_pts_[i].y);
		}
	}else {
		for(int i = 0; i < scan_res_; i++) {
			scan_start_pts_[i] = convertWorldToPixel(w_i[i]);
			scan_start_pts_[i].x = std::max(1.0, std::min(round(scan_start_pts_[i].x), static_cast<double>(cam_w_)));
			scan_start_pts_[i].y = std::max(1.0, std::min(round(scan_start_pts_[i].y), static_cast<double>(cam_h_)));
			
			scan_end_pts_[i] = convertWorldToPixel(w_f[i]);
			scan_end_pts_[i].x = std::max(1.0, std::min(round(scan_end_pts_[i].x), static_cast<double>(cam_w_)));
			scan_end_pts_[i].y = std::max(1.0, std::min(round(scan_end_pts_[i].y), static_cast<double>(cam_h_)));
			NODELET_DEBUG("Start: [%.1f, %.1f], End: [%.1f, %.1f]", scan_start_pts_[i].x, scan_start_pts_[i].y, scan_end_pts_[i].x, scan_end_pts_[i].y);
		}
	}
	
	// Fill in constant LaserScan message properties
	scan_msg_.header.frame_id = scan_frame_id_;
	scan_msg_.angle_min = -cam_FOV_/2;
	scan_msg_.angle_max = cam_FOV_/2;
	scan_msg_.angle_increment = cam_FOV_/scan_res_;
	scan_msg_.time_increment = 0.0;
	scan_msg_.range_min = min_range_;
	scan_msg_.range_max = max_range_;
	scan_msg_.ranges.resize(scan_res_);
}// end initializeScanner
	
/* getScanWorldPoints
* 
* Forms start and end points in world coordinates for scan rays from the 
* camera origin to a depth of max_range_. Rays are equally divided across
* the camera FOV.
*/
void LaneDetector::getScanWorldPoints(std::vector<cv::Point2f> &start, std::vector<cv::Point2f> &end) {
	NODELET_DEBUG("Forming scanner world points.");
	
	start.clear();
	end.clear();
	start.resize(scan_res_);
	end.resize(scan_res_);
	double theta = -cam_FOV_/2.0;
	
	for(int i = 0; i < scan_res_; i++) {
		start[i].x = min_range_;
		start[i].y = tan(theta)*min_range_;
		
		end[i].x = max_range_;
		end[i].y = tan(theta)*max_range_;
		
		theta = -cam_FOV_/2.0 + cam_FOV_*(i + 1)/(scan_res_ - 1);
		//NODELET_DEBUG("Start: [%.2f, %.2f], End: [%.2f, %.2f]", start[i].x, start[i].y, end[i].x, end[i].y);
	}
}

/* detectEdges
* 
* Performs processing and edge detection on image.
*/
void LaneDetector::detectEdges(cv::Mat &img, const std_msgs::Header &header) {
	// CLAHE Histogram equalization
	if(apply_clahe_) {
		cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
		clahe->setClipLimit(clahe_clip_limit_);
		clahe->apply(img, img);
	}
	
	// Apply bilateral filter
	if(apply_bilateral_) {
		cv::Mat bilat_in = img.clone();  // bilateralFilter cannot work inline
		cv::bilateralFilter(bilat_in, img, bilateral_k_size_, bilateral_sigma_color_, bilateral_sigma_space_);
	}
	
	// Gaussian smoothing
	if(apply_gauss_) {
		cv::GaussianBlur(img, img, cv::Size(gauss_k_size_, gauss_k_size_), gauss_sigma_x_, gauss_sigma_y_);  
	}
	
	// Apply Canny edge detection
	if(canny_thresh_ == OTSU) {
		// Apply Otsu thesholding
		cv::Mat otsu_img;
		const double otsu_thresh = cv::threshold(img, otsu_img, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
		cv::Canny(img, img, 0.5*otsu_thresh, otsu_thresh);
	}else if(canny_thresh_ == MEDIAN) {
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
	}else if (canny_thresh_ == MEAN) {
		// Use +- 33% of the mean of all pixels for threshold values
		cv::Scalar temp_mean = cv::mean(img);
		float mean = temp_mean.val[0];
		cv::Canny(img, img, 0.66*mean, 1.33*mean);
	}	
	
	// Remove vertical edges
	if(apply_vert_edge_) {
		cv::Mat v_edge_kernel = (cv::Mat_<double>(5, 1) << -1, -1, 1, -1, -1);
		cv::filter2D(img, img, -1, v_edge_kernel);
	}
	
	// Publish the edges image (if requested)
	if(pub_edges_.getNumSubscribers()) {
		sensor_msgs::ImagePtr edges_msg = cv_bridge::CvImage(header, sensor_msgs::image_encodings::MONO8, img).toImageMsg();
		pub_edges_.publish(edges_msg);
	}
}

/* imageDebugging
* 
* Performs additional operations on the input image and publishes the results.
* Intended for debugging and observing image characteristics.
*
* Currently performs:
*   -Histogram of pixel itnensities
*   -Otsu threshold
*/
void LaneDetector::imageDebugging(cv::Mat &img, const std_msgs::Header &header) {
	// Publish the image histogram (if requested)
	if(pub_histo_.getNumSubscribers()) {
	  // Establish the number of bins
	  int hist_size = 256;
	  
	  /// Set the ranges
	  float range[] = {0, 256};
	  const float* hist_range = {range};
	  
	  /// Compute the histogram
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
	
	// Publish the Otsu threshold image (if requested)
	if(pub_otsu_.getNumSubscribers()) {
		cv::Mat otsu_img;
		cv::threshold(img, otsu_img, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
		sensor_msgs::ImagePtr otsu_msg = cv_bridge::CvImage(header, sensor_msgs::image_encodings::MONO8, otsu_img).toImageMsg();
		pub_otsu_.publish(otsu_msg);
	}
}

/* scanImage
* 
* Scans the edges image as if a laser scanner was placed at the camera optical 
* origin, using the start and end points computed with initializeScanner. A hit
* is considered as a white pixel (=255).	
*/
void LaneDetector::scanImage(cv::Mat &img) {
	scan_msg_.header.stamp = ros::Time::now();
		
	double x, y;
	cv::LineIterator iter(img, cv::Point(0, 0), cv::Point(0, 0), 4);
	for(int j = 0; j < scan_res_; j++) {
		iter = cv::LineIterator(img, scan_start_pts_[j], scan_end_pts_[j], 4);

		// Default to inifinity
		scan_msg_.ranges[j] = std::numeric_limits<double>::infinity();

		// Crawl along line
		for(int k = 0; k < iter.count; k++, iter++){
			// Check for a hit
			if (*iter.operator*() == 255) {
				cv::Point hit_p = iter.pos();
				
				if(use_BEV_image_){
					y = (hit_p.x - 0.5*bev_w_)/bev_ppm_;
					x = (bev_h_ - hit_p.y)/bev_ppm_ + cam_min_depth_;
				}else{
					hit_p.x++;
					hit_p.y++;
					cv::Point2f hit_w = convertPixelToWorld(hit_p);
					x = hit_w.x;
					y = hit_w.y;
				}
				
				scan_msg_.ranges[j] = sqrt(pow(x, 2) + pow(y, 2));
				break;
			}
		} 
	}
}
	
/* convertPixelToWorld
* 
* Converts pixel positions to world coordinates using the precomputed
* transform and camera height. World X and Y is relative to camera origin.
*/
cv::Point2f LaneDetector::convertPixelToWorld(const cv::Point2f &pixel) {
	// Form pixel coordinate
	const cv::Mat p_p = (cv::Mat_<double>(3, 1) << pixel.x, pixel.y + cam_y_offset_, 1.0);
	
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
	const cv::Mat t_cw = (cv::Mat_<double>(3,1) << 0.0, 0.0, cam_Z_);
	const cv::Mat p_i = A_wc_*(p_w - t_cw); 
	
	// Solve for scaling factor
	const double s = 1.0/p_i.at<double>(2,0);
	
	// Solve for homogeneous pixel coordinates
	const double u = s*p_i.at<double>(0,0);
	const double v = s*p_i.at<double>(1,0) - cam_y_offset_;
	
	const cv::Point2f p_p = cv::Point2f(u, v);
	return p_p;
}

}// end ghost namespace 

// Register the nodelet
PLUGINLIB_DECLARE_CLASS(ghost, LaneDetector, ghost::LaneDetector, nodelet::Nodelet);