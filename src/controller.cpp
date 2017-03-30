/* Controller
*
* Compares the car's position and velocity to the racing line, determines
* steering and throttle controls through a set of PID controllers
*
* A text file with racing line coordinates is loaded, then based on the
* defined velocity and acceleration limits, the corresponding velocity and
* yaw rate for each point on the racing line are determined.
*
* The car's ucrrent position and velocity are compared to the desired values
* from the racing line, which are used to determine the error signals for the
* PID controllers.
*
* The controls are converted to the range [-1,1] and published as a 
* CarControl message.
*
* The error signals for steering and throttle are also published as Float32
* messages, with the values normalized to the range [-1,1].
*
* --------------------------------------------------------------------------
*/

#include <boost/thread.hpp>
#include <dynamic_reconfigure/server.h>
#include <fstream>
#include <geometry_msgs/PoseWithCovariance.h>
#include <geometry_msgs/TwistWithCovariance.h>
#include <ghost/CarControl.h>
#include <ghost/ControllerConfig.h>
#include <nav_msgs/Odometry.h>
#include <ros/console.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <std_msgs/Float32.h>
#include <tf/transform_datatypes.h>

#define PI 3.14159265

typedef ghost::ControllerConfig Config;
typedef dynamic_reconfigure::Server<Config> ReconfigureServer;

// Structure containing the information for every point in the racing line
struct RL_pt {
	double x;     // X-coordinate
	double y;     // Y-coordinate
	double psi;   // Yaw angle
	double v;     // Velocity
	double dpsi;  // Yaw rate
};

class Controller {
	public:
		ros::NodeHandle nh_;
		ros::NodeHandle pnh_;
		
		ros::Subscriber pose_sub_;
		ros::Publisher ctrl_pub_;
		ros::Publisher e_psi_pub_;
		ros::Publisher e_psi_integral_pub_;
		ros::Publisher e_v_pub_;
		ros::Publisher e_v_integral_pub_;
		
		// Car variables
		double l_;          // Wheelbase
		double delta_max_;  // Max steering angle [radians]
		double v_max_;      // Maximum possible car velocity
		
		// State variables
		double x_;      // Current X position
		double y_;      // Current Y position
		double psi_;    // Current yaw angle
		double v_;      // Current velocity
		double dpsi_;   // Current yaw rate
		int RL_index_;  // Current index in racing line
		
		// Track variables
		std::string track_name;  // Name of file to load
		double dpsi_max_;        // Maximum yaw rate (from v_limit and car geometry)
		std::vector<RL_pt> RL_;  // Track point coordinates, velocities, and yaw rates
		int RL_num_pts_;         // Total number of points in track
		double ds_dpt_;          // Average distance between points
		
		// PID variables
		bool first_pose_;           // Flag for if this is the first pose message
		ros::Time t_now_, t_prev_;  // Tracking time between controls
		double e_v_;                // Velocity error
		double e_psi_;              // Heading error
		double v_integral_;         // Integral of velocity error
		double psi_integral_;       // Integral of heading error
		
		// Dynamic reconfigure
		boost::recursive_mutex config_mutex_;
		boost::shared_ptr<ReconfigureServer> reconfigure_server_;
		Config config_;
		
		Controller();
		void reconfigureCallback(Config &config, uint32_t level);
		int offsetIndex(const int &start, const int &offset);
		void RLInitialization(std::string &track_name, bool &CW);
		void setRL();
		void poseCallback(const nav_msgs::Odometry::ConstPtr& msg);
		void localIndexSearch();
		void globalIndexSearch();
		void PIDControl();
};

/* Controller
*
* Constructor
*/
Controller::Controller() : pnh_("~") {
	// Get parameters
	std::string track_name;
	if(!pnh_.getParam("track", track_name)){
		ROS_ERROR("Failed to get param 'track'");
		ros::shutdown();
	}
	bool CW;
	bool debug;
	pnh_.param<bool>("debug", debug, false);
	pnh_.param<bool>("CW", CW, true);
	
	// Need some chassis parameters, possible in another namespace
	std::string key;
	if (nh_.searchParam("chassis/l", key))
	  nh_.getParam(key, l_);
	else
		l_ = 0.260;
	
	if (nh_.searchParam("chassis/delta_max", key))
	  nh_.getParam(key, delta_max_);
	else
		delta_max_ = 25.0;
	delta_max_ *= PI/180;
	
	if(nh_.searchParam("chassis/v_max", key))
    nh_.getParam(key, v_max_);
  else
    v_max_ = 15.0;
	
	// Initialization
	RLInitialization(track_name, CW);
	first_pose_ = true;
	v_integral_ = 0.0;
	psi_integral_ = 0.0;
	
	// Setup dynamic reconfigure
	reconfigure_server_.reset(new ReconfigureServer(config_mutex_, pnh_));
	ReconfigureServer::CallbackType f = boost::bind(&Controller::reconfigureCallback, this, _1, _2);
	reconfigure_server_->setCallback(f);
	
	// Manage debugging logger level
	if(debug){
		if(ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug) ) {
			ros::console::notifyLoggerLevelsChanged();
		}
	}
	
	// Setup pubs and subs
	pose_sub_ = nh_.subscribe("track_pose", 50, &Controller::poseCallback, this);
	ctrl_pub_ = nh_.advertise<ghost::CarControl>("cmd_car", 50);
	e_psi_pub_ = nh_.advertise<std_msgs::Float32>("controller_e_psi", 50);
	e_psi_integral_pub_ = nh_.advertise<std_msgs::Float32>("controller_e_psi_integral", 50);
	e_v_pub_ = nh_.advertise<std_msgs::Float32>("controller_e_v", 50);
	e_v_integral_pub_ = nh_.advertise<std_msgs::Float32>("controller_e_v_integral", 50);
}

/* reconfigureCallback
*
* Callback for updating parameters with Dynamic Reconfigure. If the maximum
* velocities or accelerations are changed, the racing velocity and yaw rate
* need to be updated.
*/
void Controller::reconfigureCallback(Config &config_in, uint32_t level) {
	bool update_RL = false;
	if(config_.v_limit != config_in.v_limit ||
		 config_.ax_accel_limit != config_in.ax_accel_limit ||
		 config_.ax_braking_limit != config_in.ax_braking_limit ||
		 config_.ay_limit != config_in.ay_limit)
		update_RL = true;
	
	config_ = config_in;
	if(update_RL)
		setRL();
}

/* offsetIndex
*
* Helper function for finding a new index in the racing line. Accounts
* for wrapping around the start when the end is reached.
*/
int Controller::offsetIndex(const int &start, const int &offset) {
	return std::max((start + offset)%RL_num_pts_, (RL_num_pts_ + (start + offset)%RL_num_pts_)%RL_num_pts_);
}

/* RLInitialization
*
* Initializes the data for the racing line (but does not set the velocities
* or yaw rates). Loads the track file, reverses it if necessary, and stores
* it in RL_.
*/
void Controller::RLInitialization(std::string &track_name, bool &CW) {	
	// Load the track file
	std::string full_file = ros::package::getPath("ghost") + "/tracks/" + track_name;	
	std::ifstream data(full_file.c_str());
		
	// Load the coordinates into the racing line (ignore first line of file)
	int i = 0;
	std::string line;
	std::getline(data, line);
	while(std::getline(data, line)) {
		std::stringstream linestream(line);
		RL_pt pt;
		linestream >> pt.x >> pt.y;
		RL_.push_back(pt);
    i++;
	}
	RL_num_pts_ = RL_.size();
	
	// Reverse the track for CCW lap, when needed
	if(!CW) {
		std::vector<RL_pt> new_RL(RL_num_pts_);
		for(int index = 0; index < RL_num_pts_; index++) {
			new_RL[RL_num_pts_ - index - 1] = RL_[index];
		}
		RL_ = new_RL;
	}
	
	// Calculate average distance between points
	double d = sqrt(pow((RL_[0].x - RL_[RL_num_pts_ - 1].x), 2) + pow((RL_[0].y - RL_[RL_num_pts_ - 1].y), 2));
	for(int index = 1; index < RL_num_pts_; index++) {
			d += sqrt(pow((RL_[index].x - RL_[index - 1].x), 2) + pow((RL_[index].y - RL_[index - 1].y), 2));
	}
	ds_dpt_ = d/RL_num_pts_;
	
}

/* setRL
* 
* Crawls through the racing line coordinates, and determines the maximum
* yaw rate and velocity, given the acceleration and velocity contstraints.
*
* First, any points where the velocity is limited by lateral accelerated are
* set. Then, crawl away from these points accelerating at the forward/braking
* limits until all points are filled.
*/
void Controller::setRL() {
	Config config;
	{
		boost::lock_guard<boost::recursive_mutex> lock(config_mutex_);
		config = config_;
	}
	
	// Update the max possible yaw rate
	dpsi_max_ = (config.v_limit/l_)*tan(delta_max_);
	
	int index;
	const double alpha = 0.9;                  // Coeff for exponential moving avg's
	const int k = 5;                           // Distance to look on each side of target point for angle calc
	std::vector<double> dpsi_dm(RL_num_pts_);  // Vector of derivative of yaw wrt distance
		
	// Initialize the previous pt so the derivative can be found
	double prev_angle = atan2(RL_[k - 1].y - RL_[RL_num_pts_ - k + 1].y, RL_[k - 1].x - RL_[RL_num_pts_ - k + 1].x);
	RL_pt prev_pt = RL_.back();
	
	// Initialize the first pts for updating moving average
	double new_angle = prev_angle;
	double dm = sqrt(pow(RL_[0].x - prev_pt.x, 2) + pow(RL_[0].y - prev_pt.y, 2));
	
	// Crawl through coords, determining the derivative of yaw wrt distance
	for(index = 0; index < RL_num_pts_; index++) {
		// Get points ahead a backwards
		const int a_index = offsetIndex(index, k);
		const int b_index = offsetIndex(index, -k);
				
		// Compute the change in angle and distance
		double new_angle_raw = atan2(RL_[a_index].y - RL_[b_index].y, RL_[a_index].x - RL_[b_index].x);
		const double dm_raw = sqrt(pow(RL_[index].x - prev_pt.x, 2) + pow(RL_[index].y - prev_pt.y, 2));
		
		// Find change in angle (must account for PI transition)
		double raw_d_angle = new_angle_raw - fmod(new_angle, 2*PI);
		if(fabs(raw_d_angle) > PI) {
			// PI boundary crossed
			raw_d_angle += (raw_d_angle < 0)*2*PI - (raw_d_angle > 0)*2*PI;
		}
		
		// Apply exponential moving average to smooth
		new_angle = alpha*new_angle + (1 - alpha)*(new_angle + raw_d_angle);
		RL_[index].psi = new_angle;
		dm = alpha*dm + (1 - alpha)*dm_raw;
		
		// Compute derivative of yaw wrt distance
		dpsi_dm[index] = alpha*dpsi_dm[std::max(index - 1, 0)] + (1 - alpha)*((new_angle - prev_angle)/dm);
		
		prev_pt = RL_[index];
		prev_angle = new_angle;
	}
	
	// Now crawl through and set the velocity and yaw rate
	int num_pts_set = 0;
	std::vector<int> ay_limited_pts;
	std::vector<RL_pt>::iterator iter;
	
	// Find all the points where the velocity is limited by lateral 
	// acceleration (i.e. a_y = V*psi_dot > ay_limit), and set the
	// appropriate velocity
	for(index = 0, iter = RL_.begin(); iter < RL_.end(); ++iter, index++) {
		if(dpsi_dm[index]*config.v_limit*config.v_limit > config.ay_limit) {
			// Set to limited value
			iter->v = sqrt(config.ay_limit/dpsi_dm[index]);
			iter->dpsi = dpsi_dm[index]*iter->v;
			
			num_pts_set++;
			ay_limited_pts.push_back(index);
		}else {
			// Set V to be negative so it can be identified later
			iter->v = -1.0;
		}
	}
	
	// When never limited by lateral acceleration, all pts are max velocity
	if(num_pts_set == 0) {
		for(index = 0, iter = RL_.begin(); iter < RL_.end(); ++iter, index++) {
			iter->v = config.v_limit;
			iter->dpsi = dpsi_dm[index]*iter->v;
		}
		return;
	}
		
	// Otherwise, some points are limited by lateral acceleration, so 
	// crawl outwards from these points accelerating at the a_x limit 
	// until all points are set
	int offset = 1;
	//for(int temp = 0; temp < 1; temp++) {
	while(num_pts_set < RL_num_pts_) {
		std::vector<int> ay_limited_pts_new;  // Points that need to be kept for next round
		
		// Loop through each limited pt, crawling outwards. When both outward
		// points are already set, remove the limited point
		std::vector<int>::iterator lim_iter;
		for(lim_iter = ay_limited_pts.begin(); lim_iter < ay_limited_pts.end(); ++lim_iter) {
			
			// Get points offset to each side
			const int a_index = offsetIndex(*lim_iter, offset);
			const int b_index = offsetIndex(*lim_iter, -offset);
			
			// Check if both need to be set
			bool a_set = (RL_[a_index].v != -1.0);
			bool b_set = (RL_[b_index].v != -1.0);
			
			// Acceleration and deceleration need to match up
			bool override_b = false;
			if(b_set) {
				const int prev_b_index = offsetIndex(b_index, 1);
				if(RL_[b_index].v > RL_[prev_b_index].v) {
					// Need to override this with a lower velocity due to braking limits
					b_set = false;
					override_b = true;
				}
			}
			
			// Handle point a
			if(!a_set) {
				// Accelerate at ax_max_accel
				const int prev_index = offsetIndex(a_index, -1);
				const double d = sqrt(pow(RL_[prev_index].x - RL_[a_index].x, 2) + pow(RL_[prev_index].y - RL_[a_index].y, 2));
				RL_[a_index].v = std::min(config.v_limit, config.ax_accel_limit*d/RL_[prev_index].v + RL_[prev_index].v);
				RL_[a_index].dpsi = dpsi_dm[a_index]*RL_[a_index].v;
				num_pts_set++;
			}
			
			// Handle point b
			if(!b_set) {
				// decelerate at ax_max_braking
				const int prev_index = offsetIndex(b_index, 1);
				const double d = sqrt(pow(RL_[prev_index].x - RL_[b_index].x, 2) + pow(RL_[prev_index].y - RL_[b_index].y, 2));
				RL_[b_index].v = std::min(config.v_limit, config.ax_braking_limit*d/RL_[prev_index].v + RL_[prev_index].v);
				RL_[b_index].dpsi = dpsi_dm[b_index]*RL_[b_index].v;
				if(!override_b)
					num_pts_set++;
			}
			
			// Keep the point for next round?
			if(!a_set || !b_set)
				ay_limited_pts_new.push_back(*lim_iter);
		}
		
		// Update for next round
		ay_limited_pts = ay_limited_pts_new;
		offset++;
	}
	
	// Smooth V and psi_dot with exponential moving average
	for(index = 1; index < RL_num_pts_; index++) {
		RL_[index].v = alpha*RL_[index - 1].v + (1 - alpha)*RL_[index].v;
		RL_[index].dpsi = alpha*RL_[index - 1].dpsi + (1 - alpha)*RL_[index].dpsi;
	}
	
	ROS_INFO("Racing line points set for v_limit=%.1f m/s, ax_accel_limit=%.1f m/s^2, ax_braking_limit=%.1f m/s^2, ay_limit=%.1f m/s^2", config.v_limit, config.ax_accel_limit, config.ax_braking_limit, config.ay_limit);
}

/* localIndexSearch
*
* Finds the racing index closest to the car by crawling away from the 
* current racing line index until the distance between the current 
* position and that index's position begins to increase.
*/
void Controller::localIndexSearch() {
	int best_index = RL_index_;
	double d_best = sqrt(pow((x_ - RL_[RL_index_].x), 2) + pow((y_ - RL_[RL_index_].y), 2));
	
	// Get new distances in each direction
	int a_index, b_index;
	double d_a, d_b;
	
	a_index = offsetIndex(RL_index_, 1);
	d_a = sqrt(pow((x_ - RL_[a_index].x), 2) + pow((y_ - RL_[a_index].y), 2));
	
	b_index = offsetIndex(RL_index_, -1);
	d_b = sqrt(pow((x_ - RL_[b_index].x), 2) + pow((y_ - RL_[b_index].y), 2));
		
	// Pick the better direction
	bool getting_worse = false;
	if(d_a < d_best) {
		best_index = a_index;
		// Go in a direction (higher index)
		while(!getting_worse) {
			a_index = offsetIndex(best_index, 1);
			d_a = sqrt(pow((x_ - RL_[a_index].x), 2) + pow((y_ - RL_[a_index].y), 2));
			if(d_a < d_best) {
				d_best = d_a;
				best_index = a_index;
			}else {
				RL_index_ = best_index;
				return;
			}
		}
		
	}else if(d_b < d_best) {
		// Go in b direction (lower index)
		while(!getting_worse) {
			b_index = offsetIndex(best_index, -1);
			d_b = sqrt(pow((x_ - RL_[b_index].x), 2) + pow((y_ - RL_[b_index].y), 2));
			if(d_b < d_best) {
				d_best = d_b;
				best_index = b_index;
			}else {
				RL_index_ = best_index;
				return;
			}
		}
	}else {
		// Already at the minimum
		return;
	}
}

/* localIndexSearch
*
* Finds the racing index closest to the car by iterating through the
* entire racing line and selecting the closest point.
*/
void Controller::globalIndexSearch() {	
	double d_min = 9999.0;
	int index = 0;
	int i;
	std::vector<RL_pt>::iterator iter;
	for(i = 0, iter = RL_.begin(); iter < RL_.end(); ++iter, i++) {
		const double d = sqrt(pow((x_ - iter->x), 2) + pow((y_ - iter->y), 2));
		if(d < d_min){
			d_min = d;
			index = i;
		}
	}
	
	RL_index_ =  index;
}

/* poseCallback
*
* Callback for car pose. Stores the vehicles state, then calls PIDControl
* to issue the necessary controls.
*/
void Controller::poseCallback(const nav_msgs::Odometry::ConstPtr& msg) {
	// Save new state
	x_ = msg->pose.pose.position.x;
	y_ = msg->pose.pose.position.y;
	v_ = sqrt(pow(msg->twist.twist.linear.x, 2) + pow(msg->twist.twist.linear.y, 2));
	dpsi_ = msg->twist.twist.angular.z;
	
	// Convert from geometry_msgs to tf quaternion, then to roll-pitch-yaw
	tf::Quaternion quat;
	tf::quaternionMsgToTF(msg->pose.pose.orientation, quat);
	double dud_roll, dud_pitch;
	tf::Matrix3x3(quat).getRPY(dud_roll, dud_pitch, psi_);
	
	t_prev_ = t_now_;
	t_now_ = msg->header.stamp;
		
	// Search for the closest position on the racing line
	if(first_pose_) {
		globalIndexSearch();
		first_pose_ = false;
		
		// Need a previous message to continue and issue controls
		return;
	} else {
		localIndexSearch();
	}
	
	// Drive
	PIDControl();
}

/* PIDControl
*
* A simple PID controller to control the car;s steering and throttle. The 
* steering error is the difference between the angle to a look-ahead point,
* and the current heading, while velocity error is simply the difference
* between the desired and current velocities.
*
* The control output from the PID's are converted to steering and throttle
* controls [-1,1] by using the defined maximum velocity and steering angle.
*/
void Controller::PIDControl() {
	Config config;
	{
		boost::lock_guard<boost::recursive_mutex> lock(config_mutex_);
		config = config_;
	}
	const double dt = (t_now_ - t_prev_).toSec();
	
	// Find heading error
	const int lookahead_index = offsetIndex(RL_index_, config.lookahead_dist/ds_dpt_);
	double angle = atan2(RL_[lookahead_index].y - y_, RL_[lookahead_index].x - x_);
	
	const double prev_e_psi = e_psi_;
	e_psi_ = angle - psi_;
	if(fabs(e_psi_) > PI) {
		// PI boundary crossed
		e_psi_ += (e_psi_ < 0)*2*PI - (e_psi_ > 0)*2*PI;
	}
	ROS_DEBUG("e_psi=%.2f [%.2f - %.2f]", e_psi_, angle, fmod(psi_, 2*PI));
	
	// Compute steering control
	psi_integral_ += e_psi_*dt;
	const double de_dpsi = (e_psi_ - prev_e_psi)/dt;
	double steering = config.K_psi*(e_psi_ + psi_integral_/config.Ti_psi + config.Td_psi*de_dpsi);
	steering = std::max(std::min(steering, 1.0), -1.0);
	ROS_DEBUG("dpsi_integral=%.2f, de_dpsi=%.2f", psi_integral_, de_dpsi);
	
	// Publish the heading error and integral (normalized to [-1,1])
	std_msgs::Float32 e_psi_msg;
	e_psi_msg.data = e_psi_/PI;
	e_psi_pub_.publish(e_psi_msg);
	
	std_msgs::Float32 e_psi_integral_msg;
	e_psi_integral_msg.data = psi_integral_/PI;
	e_psi_integral_pub_.publish(e_psi_integral_msg);
	
	// Find velocity error
	const double prev_e_v = e_v_;
	double e_v_ = (RL_[RL_index_].v - v_)/v_max_;
	ROS_DEBUG("e_v=%.2f [%.2f - %.2f]", e_v_, RL_[RL_index_].v, v_);
		
	// Compute throttle control
	v_integral_ += e_v_*dt;
	const double de_v = (e_v_ - prev_e_v)/dt;
	double delta_throttle = config.K_v*(e_v_ + v_integral_/config.Ti_v + config.Td_v*de_v);
	double throttle = v_/v_max_ + delta_throttle;
	throttle = std::max(std::min(throttle, 1.0), -1.0);
	ROS_DEBUG("v_integral_=%.2f, de_v=%.2f", v_integral_, de_v);
	
	// Publish the velocity error and integral (normalized to [-1,1])
	std_msgs::Float32 e_v_msg;
	e_v_msg.data = e_v_;
	e_v_pub_.publish(e_v_msg);
	
	std_msgs::Float32 e_v_integral_msg;
	e_v_integral_msg.data = v_integral_;
	e_v_integral_pub_.publish(e_v_integral_msg);
		
	// Publish control
	ghost::CarControl msg;
	msg.steering = steering;
	msg.throttle = throttle;
	ctrl_pub_.publish(msg);
	
	ROS_DEBUG("Steering=%.2f, Throttle=%.2f", steering, throttle);
}

/* main
*
* Controller entry point.
*/ 
int main(int argc, char** argv){
	// Initialize
	ros::init(argc, argv, "controller");
	
	// Create the controller object
	Controller ctrl;
	
	ros::spin();
}
