/* LaneDetector Nodelet
*
* Images from the camera are processed to detect edges (i.e. the edges
* of the floor), the pixel coordinates of which are then projected 
* into real world coordinates.
*
* The resulting edges are then scanned as if a laser scanner were placed
* at the camera's position, where the first edge detected by a ray is returned
* as hit. The result is then published as a LaserScan message.
*
* Inputted parameters dictacte the camera's position, as will as the range
* and resolution of the outputted scan.
*
* In addition to the LaserScan message, image topics are also available for
* the detected edges, and the Bird's Eye View of the transformed image. All
* topics are only published when subscribed to.
*
* ----------------------------------------------------------------------------------
*
* Subscribed Topics
*   image_rect_mono (sensor_msgs/Image)
*     -Rectified grayscale image from camera
*
* Published Topics
*   scan (sensor_msgs/LaserScan)
*     -Detected edge points in the image
*   bev_image (sensor_msgs/Image)
*     -Cropped version of image_rect_mono with Birds Eye View transformation applied
*   edges_image (sensor_msgs/Image)
*     -Edges detected in the subscribed image_rect_mono
*
* Parameters
*   ~cam_height (double, default: 0.1)
*     -Height of camera from ground plan [m]
*   ~cam_angle (double, default: 0.0)
*     -Angle of camera about the X axis [Deg]
*   ~min_range (double, defualt: 0.5)
*     -Minimum detactable range for scanner [m]
*   ~max_range (double, defualt: 5.0)
*     -Maximum detactable range for scanner [m]
*   ~angle_increment (double, default: 1.0)
*     -Angular resolution of scan [Deg]
*   ~gauss_k_size (int, default 3)
*     -Size of kernal for Gaussian blur
*   ~gauss_sigma_x (double, default: 10.0)
*     -Gaussian kernel standar deviation in the X direction
*   ~gauss_sigma_y (double, default: 10.0)
*     -Gaussian kernel standar deviation in the Y direction
*   ~bilateral_k_size (int, default: 3)
*     -Size of kernel for bilateral filter
*   ~bilateral_sigma_color (int, default 1000)
*     -Filter sigma in the color space
*   ~bilateral_sigma_space (int, default 1000)
*     -Filter sigma in the coordinate space
*   ~clahe_clip_limit (double, default: 1.0)
*     -Clipping limit for CLAHE histogram equalization
*
* ----------------------------------------------------------------------------------
*/

#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/LaserScan.h>

#define PI 3.14159265

namespace ghost {

class LaneDetector : public nodelet::Nodelet {

  // Image transport
  boost::shared_ptr<image_transport::ImageTransport> it_;
  boost::shared_ptr<image_transport::ImageTransport> private_it_;

  // Pubs and Subs
  image_transport::CameraSubscriber sub_camera_;   // To camera
  ros::Publisher pub_scan_;                        // Laserscan message
  image_transport::Publisher pub_bev_;             // Bird's eye view image
  image_transport::Publisher pub_edges_;           // Image with edges

  // BEV image properties
  double v_max_depth_;  // Pixel height for max range (for cropping and indexing image)
  int bev_w_;           // BEV image width
  int bev_h_;           // BEV image height
  double mpp_;          // BEV meter-per-pixel
  cv::Mat M_;           // BEV transformation matrix

  // Scaner properties
  int scan_num_points_;                     // Total number of scan points
  std::vector<cv::Point2f> scan_start_pts_; // Pixel start points of each scan ray
  std::vector<cv::Point2f> scan_end_pts_;   // Pixel end points of each scan ray

  bool initialized_;  // Flag for if the BEV transform and scanner have been calculated

  // Camera parameters (loaded from parameter server)
  double cam_Y_;                 // Height of camera from ground
  double cam_angle_;             // Angle of camera (about the x axis)

  // Scanner parameters (loaded from parameter server)
  double max_range_;             // Maximum range of camera/scan
  double min_range_;             // Minimum range of camera/scan
  double scan_angle_increment_;  // Angle between scans

  // Image filtering parameters (loaded from parameter server)
  double clahe_clip_limit_;      // Clipping limit for CLAHE histogram equalization
  int gauss_k_size_;             // Size of kernal for Gaussian blur
  double gauss_sigma_x_;         // Gaussian kernel standar deviation in the X direction
  double gauss_sigma_y_;         // Gaussian kernel standar deviation in the Y direction
  int bilateral_k_size_;         // Size of kernel for bilateral filter
  int bilateral_sigma_color_;    // Filter sigma in the color space
  int bilateral_sigma_space_;    // Filter sigma in the coordinate space
  
  //--------------------------------------

  // Initializer
  virtual void onInit(){

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

    // Load configuration parameters from the parameter server
    pnh.param<double>("cam_height", cam_Y_, 0.10);
    pnh.param<double>("cam_angle", cam_angle_, 0.0);
    pnh.param<double>("min_range", min_range_, 0.5);
    pnh.param<double>("max_range", max_range_, 5.0);
    pnh.param<double>("angle_increment_", scan_angle_increment_, 1.0);
    pnh.param<int>("gauss_k_size", gauss_k_size_, 3);
    pnh.param<double>("gauss_sigma_x", gauss_sigma_x_, 10.0);
    pnh.param<double>("gauss_sigma_y", gauss_sigma_y_, 10.0);
    pnh.param<int>("bilateral_k_size", bilateral_k_size_, 3);
    pnh.param<int>("bilateral_sigma_color", bilateral_sigma_color_, 1000);
    pnh.param<int>("bilateral_sigma_space", bilateral_sigma_space_, 1000);
    pnh.param<double>("clahe_clip_limit", clahe_clip_limit_, 1.0);

    initialized_ = false;

  }// end onInit

  //--------------------------------------

  /* connectCb
  *
  * Handles (un)subscribing when clients (un)subscribe
  */
  void connectCb() {
    if (pub_scan_.getNumSubscribers() == 0 && pub_bev_.getNumSubscribers() == 0 && pub_edges_.getNumSubscribers() == 0){
      NODELET_INFO("No listeners, shutting down camera subscriber.");
      sub_camera_.shutdown();
    } else if (!sub_camera_) {
      NODELET_INFO("Subscribing to camera topic.");
      image_transport::TransportHints hints("raw", ros::TransportHints(), getPrivateNodeHandle());
      sub_camera_ = it_->subscribeCamera("image_rect_mono", 1, &LaneDetector::imageCb, this, hints);
    }
  }
  
  //--------------------------------------

  /* imageCb
  *
  * Performs all processing on the image to find the lane edges.
  *
  * First, the image is filtered to detect the edges, then the BEV transform
  * is applied, then the image is scanned from the origin as if a laser scanner
  * were present, and the detect edges are outputted as a LaserScan message.
  *
  * The image is filtered in 5 stages
  *    -Stage 1: Crop
  *    -Stage 2: Histogram equalization (modifies input image)
  *    -Stage 3: Smoothing with bilateral and Gaussian filters
  *    -Stage 4: Canny edge detection
  *    -Stage 5: Remove vertical edges
  */
  void imageCb(const sensor_msgs::ImageConstPtr& image_msg, const sensor_msgs::CameraInfoConstPtr& info_msg){
    // Make sure the BEV transform and image scan points are initialized
    if (!initialized_){
      NODELET_INFO("Initializing detector.");
      initializeTransform(info_msg);
      initializeScanner();
      initialized_ = true;
    }

    // Crop the input (apply ROI to the source image, to avoiding copy the unused area)
    const cv::Mat source = cv_bridge::toCvShare(image_msg, sensor_msgs::image_encodings::MONO8)->image;
    cv::Rect crop_ROI(0, v_max_depth_, image_msg->width, image_msg->height - v_max_depth_);
    cv::Mat image = source(cv::Rect(crop_ROI));

    // Convert to grayscale if necessary

    // Publish the BEV image (if requested)
    if (pub_bev_.getNumSubscribers()) {
      // Must make a copy since image may be modified further
      cv::Mat bev = image.clone();
      cv::warpPerspective(bev, bev, M_, cv::Size(bev_w_, bev_h_), cv::BORDER_CONSTANT);
      sensor_msgs::ImagePtr bev_msg = cv_bridge::CvImage(image_msg->header, sensor_msgs::image_encodings::MONO8, bev).toImageMsg();
      pub_bev_.publish(bev_msg);
    }
    
    // Histogram equalization
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(clahe_clip_limit_);
    clahe->apply(image, image);

    // Only proceed if the scan or the edges image is subscribed to
    if (!pub_scan_.getNumSubscribers() && !pub_edges_.getNumSubscribers())
      return;

    // Apply bilateral and Gaussian filter
    cv::Mat bilat_in = image.clone();  // bilateralFilter cannot work inline
    cv::bilateralFilter(bilat_in, image, bilateral_k_size_, bilateral_sigma_color_, bilateral_sigma_space_);
    cv::GaussianBlur(image, image, cv::Size(gauss_k_size_, gauss_k_size_), gauss_sigma_x_, gauss_sigma_y_);    

    // Apply Canny edge detection using Otsu's method
    cv::Mat dud; // throwaway for threshold
    double otsu_thresh = cv::threshold(image, dud, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    cv::Canny(image, image, otsu_thresh, 0.5*otsu_thresh);
    
    // Remove vertical edges
    cv::Mat v_edge_kernel = (cv::Mat_<double>(5, 1) << -1, -1, 1, -1, -1);
    cv::filter2D(image, image, -1, v_edge_kernel);

    // Publish the edges image (if requested)
    if (pub_edges_.getNumSubscribers()) {
      // Must make a copy since image may be modified further
      cv::Mat edges = image.clone();
      sensor_msgs::ImagePtr edges_msg = cv_bridge::CvImage(image_msg->header, sensor_msgs::image_encodings::MONO8, edges).toImageMsg();
      pub_edges_.publish(edges_msg);
    }

    // Only proceed if the scan is subscribed to
    if (!pub_scan_.getNumSubscribers())
      return;
    
    // Apply transform for the Bird's Eye View
    cv::warpPerspective(image, image, M_, cv::Size(bev_w_, bev_h_), cv::BORDER_CONSTANT);

    // Form the LaserScan message
    sensor_msgs::LaserScan scan;
    scan.header.stamp = ros::Time::now();
    scan.header.frame_id = image_msg->header.frame_id;
    scan.angle_min = -PI/2;
    scan.angle_max = PI/2;
    scan.angle_increment = PI/scan_num_points_;
    scan.time_increment = 0.0;
    scan.range_min = min_range_;
    scan.range_max = max_range_;
    scan.ranges.resize(scan_num_points_);
    
    // Scan the BEV image with edges as if a laser scanner was placed at the origin
    double x, y;
    cv::LineIterator iter(image, cv::Point(0, 0), cv::Point(0, 0), 4);
    for(int j = 0; j < scan_num_points_; j++) {
      iter = cv::LineIterator(image, scan_start_pts_[j], scan_end_pts_[j], 4);

      // Default to inifinity
      scan.ranges[j] = std::numeric_limits<double>::infinity();

      // Crawl along line
      for(int k = 0; k < iter.count; k++, iter++){
        // Check for a hit
        if (*iter.operator*() == 255) {
          cv::Point hit = iter.pos();

          x = (hit.x - 0.5*bev_w_)*mpp_;
          y = (bev_h_ - hit.y)*mpp_;

          scan.ranges[j] = sqrt(pow(x, 2) + pow(y, 2));
          break;
        }
      } 
    }

    // Publish the laserscan
    pub_scan_.publish(scan);

  }// end imageCb

  //--------------------------------------

  /* initializeTransform
  *
  * Computes the transform for the Bird's Eye View image based on the max range
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
  void initializeTransform(const sensor_msgs::CameraInfoConstPtr& info_msg) {
    // Get the camera parameters
    double fp_w = info_msg->K[0];
    double fp_h = info_msg->K[4];
    double u_o = info_msg->K[2];
    double v_o = info_msg->K[5];
    double w_og = info_msg->width;
    double h_og = info_msg->height;

    // Find pixel height for max depth (for cropping and indexing image)
    v_max_depth_ = fp_h*(cam_Y_/max_range_) + v_o;

    // Set the (cropped) image size
    double w = w_og;
    double h = int(h_og - v_max_depth_) + 1;

    // Find world width at the bottom of the image, then find the L and R pixel
    // positions with that same world width but at depth max_range_.
    // These will be used for the homography.
    double X_low_l = cam_Y_*(fp_h/fp_w)*(0.0 - u_o)/(h_og - 1.0 - v_o);
    double X_low_r = cam_Y_*(fp_h/fp_w)*(w_og - 1.0 - u_o)/(h_og - 1.0 - v_o);
    double u_high_l = (X_low_l/cam_Y_)*(v_max_depth_ - v_o)*(fp_w/fp_h) + u_o;
    double u_high_r = (X_low_r/cam_Y_)*(v_max_depth_ - v_o)*(fp_w/fp_h) + u_o;

    // Compute the meters-per-pixel for the BEV image
    double delta_X = X_low_r - X_low_l;
    double delta_u = u_high_r - u_high_l;
    mpp_ = delta_X/delta_u;

    // New width well be equal to original image width at max depth, but height
    // will be scaled to match resolution
    double dst_w = delta_X/mpp_;
    double dst_h = max_range_/mpp_;

    // Form arrays of points to get the homography 
    // In order TL, TR, BR, BL, and of the form [col, row]
    std::vector<cv::Point2f> src(4);
    src[0] = cv::Point2f(u_high_l, 0);
    src[1] = cv::Point2f(u_high_r, 0);
    src[2] = cv::Point2f(w - 1, h - 1);
    src[3] = cv::Point2f(0, h - 1);

    std::vector<cv::Point2f> dst(4);
    dst[0] = cv::Point2f(0, 0);
    dst[1] = cv::Point2f(dst_w - 1, 0);
    dst[2] = cv::Point2f(dst_w - 1, dst_h - 1);
    dst[3] = cv::Point2f(0, dst_h - 1);

    // Compute the homography
    cv::Mat H = cv::getPerspectiveTransform(src, dst);

    // Use the BEV image resolution to determine the output size
    // BEV width will be limited by either the max range, or the FOV
    bev_h_ = int(max_range_/mpp_);
    double fov_max_w = (bev_h_*w_og)/fp_w + 0.5*dst_w;
    bev_w_ = int(std::min(2*max_range_/mpp_, fov_max_w));

    // Rotation about the X-axis to account for camera angle
    // (have to add translations to rotate about the centre)
    cv::Mat Rx_T1 = (cv::Mat_<double>(3, 3) <<
              1.0, 0.0, -w/2,
              0.0, 1.0, -h/2,
              0.0, 0.0,  1.0);
    cv::Mat Rx_T2 = (cv::Mat_<double>(3, 3) <<
              1.0, 0.0, w/2,
              0.0, 1.0, h/2,
              0.0, 0.0, 1.0);

    double alpha = cam_angle_*PI/180.0;
    cv::Mat Rx = (cv::Mat_<double>(3, 3) <<
              1.0,     0.0,        0.0,
              0.0, cos(alpha), -sin(alpha),
              0.0, sin(alpha),  cos(alpha));

    cv::Mat R = Rx_T2*Rx*Rx_T1;

    // Translation to centre the image and shift to the bottom
    double x_offset = (bev_w_ - dst_w)/2;
    double y_offset = 0;
    cv::Mat T = (cv::Mat_<double>(3, 3) <<
              1.0, 0.0, x_offset,
              0.0, 1.0, y_offset,
              0.0, 0.0, 1.0);

    // Form the full transformation
    M_ = T*H*R;
    
  }// end initializeTransform

  //--------------------------------------

  /* initializeScanner
  * 
  * Creates start and end points for each scan ray, beginning from the original
  * of the BEV image (i.e. middle of the bottom row)
  */
  void initializeScanner() {
    // TODO: Reformulate to only scan up to the max range, or the edge of the
    // valid area of the BEV image, whichever is closer

    // Set scanner parameters (sweep across the entire image)
    double angle_min = 0.0;
    double angle_max = 180.0;
    scan_num_points_ = floor((angle_max - angle_min)/scan_angle_increment_);

    // Create start and end points
    // X and Y components are multiplied by the diagonal size of the image,
    // since the points will clipped to the image boundaries anyways
    cv::Point2f scan_origin = cv::Point2f(int(bev_w_/2), bev_h_ - 1);
    scan_start_pts_.resize(scan_num_points_);
    scan_end_pts_.resize(scan_num_points_);

    double x_start, y_start, x_end, y_end, x_comp, y_comp, min_length, max_length, angle;
    max_length = max_range_/mpp_;
    min_length = min_range_/mpp_;
    angle = angle_min + 0.5*scan_angle_increment_;

    for(int i = 0; i < scan_num_points_; i++) {
      // Set x and y components
      x_comp = cos(angle*PI/180.0);
      y_comp = -sin(angle*PI/180.0);

      // Set start and end points
      x_start = x_comp*min_length;
      y_start = y_comp*min_length;
      x_end = x_comp*max_length;
      y_end = y_comp*max_length;

      scan_start_pts_[i] = scan_origin + cv::Point2f(x_start, y_start);
      scan_end_pts_[i] = scan_origin + cv::Point2f(x_end, y_end);
      angle += scan_angle_increment_;
    }    
    
  }// end initializeScanner

};// end LaneDetector

}// end ghost namespace 

// Register the nodelet
PLUGINLIB_DECLARE_CLASS(ghost, LaneDetector, ghost::LaneDetector, nodelet::Nodelet);