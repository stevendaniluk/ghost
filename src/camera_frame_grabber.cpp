/* Saves images from camera at 2Hz 

   Accepts a single input argument for the camera topic name.
   If no argument is provided, a default name is used.
   Saves images in the directory from the which the script is called.
   (Must be called with rosrun, since it won't have write permission with roslaunch)
*/

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <highgui.h>

class ImageConverter {
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;

  int img_count_;
  
public:
  ImageConverter(std::string camera_topic) : it_(nh_) {
    img_count_ = 0;

    // Subscribe to input video feed
    image_sub_ = it_.subscribe(camera_topic, 1, &ImageConverter::imageCb, this);
  }// end ImageConverter

  void imageCb(const sensor_msgs::ImageConstPtr& msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    // Save image
    char buf[255];
    std::string name;
    sprintf(buf, "img_%05d.jpg", img_count_);
    name = buf;
    cv::imwrite(name, cv_ptr->image );

    img_count_ ++;
  }// end imageCb
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "frame_grabber");

  // Check if input argument has camera topic
  std::string camera_topic;
  if (argc == 2){
    camera_topic = argv[1];
  }else {
    camera_topic = "/camera/color/image_raw";
  }

  ImageConverter ic(camera_topic);

  ROS_INFO("Beginning recording...");

  // Get and save images at 2Hz
  ros::Rate loop_rate(2);
  while (ros::ok()) {
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}