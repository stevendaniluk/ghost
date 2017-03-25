/* odometry_integration
*
* Integrates encoder messages from all four wheels, as well as using the 
* steering angle to form an estimate of the pose and velocity.
*
* The motion model is ased off of Automotive Control Systems - For Engine, 
* Driveline, and Vehicle, Sections 8.3, 9.1, and 9.2.
*
* The velocity estimate is formed from the average velocity of all four
* wheels, while the yaw rate is determined solely from the yaw rate of the
* rear axle (since the car currently has a one-way diff in the front).
*/

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <ghost/ArduinoState.h>
#include <boost/assign.hpp>

#define PI 3.141592653589793

class Odometry {
	public:
		ros::NodeHandle nh_;
		
		// Pubs/Subs/Broadcasters
		ros::Subscriber arduino_sub_;
		ros::Publisher odom_pub_;
		tf::TransformBroadcaster odom_broadcaster_;
		
		// Outgoing messages
		geometry_msgs::TransformStamped odom_trans_;
		nav_msgs::Odometry odom_msg_;
		
		// Parameters
		double d_;          // Measured wheel diameter [m]
		double Cd_;         // Wheel diameter calibration coefficient
		double b_;          // Measured track width [m]
		double Cb_;         // Track width calibration coefficient
		double delta_max_;  // Max average steering angle between L and R wheels [radians]
		int m_;             // Number of encoder pulses for one revolution
		bool publish_tf_;
		
		// Data variables
		double x_;                 // Pose X coordinate
		double y_;                 // Pose Y coordinate
		double psi_;               // Pose yaw angle
		double v_;                 // Estimated velocity
		double psi_dot_;           // Estimated yaw rate
		uint32_t FL_prev_pulses_;  // FL wheel ncoder pulses from previous message
		uint32_t FR_prev_pulses_;  // FR wheel ncoder pulses from previous message
		uint32_t RL_prev_pulses_;  // RL wheel ncoder pulses from previous message
		uint32_t RR_prev_pulses_;  // RR wheel ncoder pulses from previous message
		
		// Update related variables
		ros::Time update_time_, last_update_time_;  // Time of state updates
		ros::Duration publish_period_;              // Time between publishing odom messages
		bool update_flag_;                          // Flag for if a new state is available
		bool first_message_flag_;                   // Flag for this is the first message received
		
		Odometry();
		void arduinoStateCallback(const ghost::ArduinoState& msg);
		void publishOdom();
};

Odometry::Odometry() {
	// Get relevant parameters
	int update_rate;
	double delta_max_deg;
	std::vector<double> pose_cov_diag(6, 1e-6);
	std::vector<double> twist_cov_diag(6, 1e-6);
	std::string base_frame, world_frame;
	nh_.param<double>("chassis/d", d_, 0.060);
	nh_.param<double>("chassis/Cd", Cd_, 1.0);
	nh_.param<double>("chassis/b", b_, 0.162);
	nh_.param<double>("chassis/Cb", Cb_, 1.0);
	nh_.param<double>("chassis/delta_max", delta_max_deg, 30.0);
	nh_.param<int>("odometry/pulses_per_rev", m_, 24);
	nh_.param<int>("odometry/update_rate", update_rate, 30);
	nh_.param<std::string>("odometry/base_frame", base_frame, "odom");
	nh_.param<std::string>("odometry/world_frame", world_frame, "base_footprint");
	nh_.param<bool>("odometry/publish_tf", publish_tf_, false);
	
	if(nh_.hasParam("odometry/pose_covariance_diagonal")) {
		XmlRpc::XmlRpcValue pose_cov_list;
		nh_.getParam("odometry/pose_covariance_diagonal", pose_cov_list);
		ROS_ASSERT(pose_cov_list.getType() == XmlRpc::XmlRpcValue::TypeArray);
		ROS_ASSERT(pose_cov_list.size() == 6);
		for (int i = 0; i < pose_cov_list.size(); ++i){
			// Read as string to handle no decimals and scientific notation
			std::ostringstream ostr;
      ostr << pose_cov_list[i];
      std::istringstream istr(ostr.str());
      istr >> pose_cov_diag[i];    
		}
	}else {
		ROS_WARN("Pose covariance diagonals not specified for odometry integration. Defaulting to 1e-6.");
	}
	
	if(nh_.hasParam("odometry/twist_covariance_diagonal")) {
		XmlRpc::XmlRpcValue twist_cov_list;
		nh_.getParam("odometry/twist_covariance_diagonal", twist_cov_list);
		ROS_ASSERT(twist_cov_list.getType() == XmlRpc::XmlRpcValue::TypeArray);
		ROS_ASSERT(twist_cov_list.size() == 6);
		for (int i = 0; i < twist_cov_list.size(); ++i){
			// Read as string to handle no decimals and scientific notation
			std::ostringstream ostr;
      ostr << twist_cov_list[i];
      std::istringstream istr(ostr.str());
      istr >> twist_cov_diag[i];
		}
	}else {
		ROS_WARN("Twist covariance diagonals not specified for odometry integration. Defaulting to 1e-6.");
	}
			
	delta_max_ = delta_max_deg*PI/180.0;
	publish_period_ = ros::Duration(1.0/update_rate);
	
	// Subscribe to cmd_car_executed and encoder_pulses messsages
	arduino_sub_ = nh_.subscribe("arduino_state", 50, &Odometry::arduinoStateCallback, this);
	
	// Setup publisher of odometry message
	odom_pub_ = nh_.advertise<nav_msgs::Odometry>("odometry/integrated", 50);
	
	// Initialize the states
	x_ = 0.0;
	y_ = 0.0;
	psi_ = 0.0;
	v_ = 0.0;
	psi_dot_ = 0.0;
	update_time_ = ros::Time::now();
	
	// Initialize pulse counters
	FL_prev_pulses_ = 0;
	FR_prev_pulses_ = 0;
	RL_prev_pulses_ = 0;
	RR_prev_pulses_ = 0;
	
	// Initialize flags
	update_flag_ = true;
	first_message_flag_ = true;
	
	// Initialize message fields, and publish an initial transform
	if(publish_tf_) {
		odom_trans_.header.frame_id = world_frame;
		odom_trans_.child_frame_id = base_frame;
		odom_trans_.transform.translation.z = 0.0;
		odom_trans_.transform.translation.x = x_;
		odom_trans_.transform.translation.y = y_;
		
		geometry_msgs::Quaternion odom_quat = tf::createQuaternionMsgFromYaw(psi_);
		odom_trans_.transform.rotation = odom_quat;
		
		odom_broadcaster_.sendTransform(odom_trans_);
  }
  
	odom_msg_.header.frame_id = world_frame;
	odom_msg_.child_frame_id = base_frame;
	odom_msg_.pose.pose.position.z = 0.0;
	odom_msg_.pose.covariance = boost::assign::list_of
        (pose_cov_diag[0]) (0)  (0)  (0)  (0)  (0)
        (0)  (pose_cov_diag[1]) (0)  (0)  (0)  (0)
        (0)  (0)  (pose_cov_diag[2]) (0)  (0)  (0)
        (0)  (0)  (0)  (pose_cov_diag[3]) (0)  (0)
        (0)  (0)  (0)  (0)  (pose_cov_diag[4]) (0)
        (0)  (0)  (0)  (0)  (0)  (pose_cov_diag[5]);
	odom_msg_.twist.twist.linear.y = 0.0;
	odom_msg_.twist.twist.linear.z = 0.0; 
	odom_msg_.twist.twist.angular.x = 0.0;
	odom_msg_.twist.twist.angular.y = 0.0;
	odom_msg_.twist.covariance = boost::assign::list_of
        (twist_cov_diag[0]) (0)  (0)  (0)  (0)  (0)
        (0)  (twist_cov_diag[1]) (0)  (0)  (0)  (0)
        (0)  (0)  (twist_cov_diag[2]) (0)  (0)  (0)
        (0)  (0)  (0)  (twist_cov_diag[3]) (0)  (0)
        (0)  (0)  (0)  (0)  (twist_cov_diag[4]) (0)
        (0)  (0)  (0)  (0)  (0)  (twist_cov_diag[5]);
}

void Odometry::arduinoStateCallback(const ghost::ArduinoState& msg) {
	update_time_ = msg.header.stamp;
	
	// Determine steering angle
	const double delta = msg.steering*delta_max_;
	
	// Track pulses from each wheel
	const int n_FL = msg.FL_pulse_count - FL_prev_pulses_;
	const int n_FR = msg.FR_pulse_count - FR_prev_pulses_;
	const int n_RL = msg.RL_pulse_count - RL_prev_pulses_;
	const int n_RR = msg.RR_pulse_count - RR_prev_pulses_;
	FL_prev_pulses_ = msg.FL_pulse_count;
	FR_prev_pulses_ = msg.FR_pulse_count;
	RL_prev_pulses_ = msg.RL_pulse_count;
	RR_prev_pulses_ = msg.RR_pulse_count;
	
	// Skip updating on the first message (in the event that this node is started
	// after some encode pulses have already been recorded)
	if(first_message_flag_) {
		first_message_flag_ = false;
		last_update_time_ = update_time_;
		return;
	}
	
	const double dt = (update_time_ - last_update_time_).toSec();
	
	// Determine distance travelled (average of each wheel)
	const double ds = (PI*d_*Cd_/(4*m_))*((n_FL + n_FR)*cos(delta) + n_RL + n_RR);
	
	// Calculate change in yaw angle from the rear axle
	const double dpsi = (PI*d_*Cd_/(m_*b_*Cb_))*(n_RR - n_RL);
	
	// Record velocities
	v_ = ds/dt;
	psi_dot_ = dpsi/dt;
	
	// Integrate velocity
	if(fabs(psi_dot_) > 1e-6) {
		// Perform exact integration
		const double r = ds/dpsi;
		x_ += r*(sin(psi_ + dpsi) - sin(psi_));
		y_ -= r*(cos(psi_ + dpsi) - cos(psi_));
		psi_ += dpsi;
	}else {
		// Perform 2nd order Runge-Kutta integration
		const double delta_V = v_*dt;
		x_ += ds*cos(psi_ + 0.5*dpsi);
		y_ += ds*sin(psi_ + 0.5*dpsi);
		psi_ += dpsi;
	}
	
	last_update_time_ = update_time_;
	update_flag_ = true;
}

void Odometry::publishOdom() {
	// Fill in the message fields
	
	// Need a quaternion created from the yaw angle
	geometry_msgs::Quaternion odom_quat = tf::createQuaternionMsgFromYaw(psi_);

	// Transform message
	if(publish_tf_){
		odom_trans_.header.stamp = update_time_;
		odom_trans_.transform.translation.x = x_;
		odom_trans_.transform.translation.y = y_;
		odom_trans_.transform.rotation = odom_quat;
		odom_broadcaster_.sendTransform(odom_trans_);
  }

	// Odometry message
	odom_msg_.header.stamp = update_time_;
	odom_msg_.pose.pose.position.x = x_;
	odom_msg_.pose.pose.position.y = y_;
	odom_msg_.pose.pose.orientation = odom_quat;
	odom_msg_.twist.twist.linear.x = v_;
	odom_msg_.twist.twist.angular.z = psi_dot_;
	odom_pub_.publish(odom_msg_);
}

int main(int argc, char** argv){
	// Initialize
	ros::init(argc, argv, "odometry_integration");
	
	// Create the odometry object
	Odometry odom;
		
	// Spin and publish
	ros::Time last_publish_time;
	while(ros::ok()) {
		ros::spinOnce();
		
		// Need to wait long enough, and have a new message to publish
		if(((last_publish_time + odom.publish_period_) < ros::Time::now()) && odom.update_flag_) {
			odom.update_flag_ = false;
			odom.publishOdom();
			last_publish_time += odom.publish_period_;
		}
	}
}
