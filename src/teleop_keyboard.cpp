/* Reads in arrow keystrokes, and publishes car control commands

   Simply increments the steering angle and velocity, then publishes it as
   a CarControl message.
*/

#include <ros/ros.h>
#include <signal.h>
#include <termios.h>
#include <stdio.h>
#include <algorithm>
#include <ghost/CarControl.h>

#define KEYCODE_R 0x43 
#define KEYCODE_L 0x44
#define KEYCODE_U 0x41
#define KEYCODE_D 0x42
#define KEYCODE_S 0x20
#define KEYCODE_Q 0x71
#define KEYCODE_A 0x61
#define KEYCODE_NULL 0x00

//------------------------------------

class Teleop {
  public:
    Teleop();
    void keyLoop();
        
    // Message to be published
    ghost::CarControl msg;

  private:
  ros::NodeHandle nh_;
  ros::Publisher car_control_pub_;
  
};// end class

// Constructor
Teleop::Teleop() {  
  // Setup publisher
  car_control_pub_ = nh_.advertise<ghost::CarControl>("cmd_car", 1, true);
}

// Global variables for key inputs
int kfd = 0;
struct termios cooked, raw;

// Loop for reading in keys
void Teleop::keyLoop() {
  char key;
  bool unknown_key = false;
  
  float steering = 0;
  float throttle = 0;

  // Get the console in raw mode                          
  tcgetattr(kfd, &cooked);
  memcpy(&raw, &cooked, sizeof(struct termios));
  raw.c_lflag &=~ (ICANON | ECHO);
  
  // Setting a new line, then end of file                         
  raw.c_cc[VEOL] = 1;
  raw.c_cc[VEOF] = 2;
  tcsetattr(kfd, TCSANOW, &raw);
  
  // Set timeout for reading keystrokes
  struct timeval timeout;
  timeout.tv_sec = 0.01;
  timeout.tv_usec = 0;
  
  puts(" ");
  puts("---------------------------");
  puts("Reading from keyboard");
  puts("Use arrow keys to move the car.");
  puts(" ");
  puts("Up Arrow = Increment Throttle Up");
  puts("Down Arrow = Increment Throttle Down");
  puts("Left Arrow = Increment Steering Left");
  puts("Right Arrow = Increment Steering Right");
  puts("Space Bar = Set Throttle To Zero");
  puts(" ");
  
  // Set precision for printing max velocity
  std::cout.setf(std::ios::fixed, std::ios::floatfield);
  std::cout.precision(2);
  
  // Variables for controlling publish rate
  float pub_rate = 20;
  ros::Time prev_pub_time = ros::Time::now();
  
  // Loop while reading key inputs
  while(1) {
    
    // Initialize file descriptor sets (necessary for select method)
    fd_set read_fds, write_fds, except_fds;
    FD_ZERO(&read_fds);
    FD_ZERO(&write_fds);
    FD_ZERO(&except_fds);
    FD_SET(kfd, &read_fds);
    
    // Wait for input to become ready or until the time out
    if (select(kfd + 1, &read_fds, &write_fds, &except_fds, &timeout) == 1) {
      // kfd is ready for reading
      if(read(kfd, &key, 1) < 1) {
        perror("read():");
        exit(-1);
      }
    } else {
      // Timeout
      key = KEYCODE_NULL;
    }
    
    ROS_DEBUG("value: 0x%02X\n", key);
    unknown_key = false;
    switch(key) {
      case KEYCODE_L:
        steering -= 0.1;
        break;
      case KEYCODE_R:
        steering += 0.1;
        break;
      case KEYCODE_U:
        throttle += 0.05;
        break;
      case KEYCODE_D:
        throttle -= 0.05;
        break;
      case KEYCODE_S:
  	    throttle = 0;
        break;
      default:
        unknown_key = true;
        break;
    }// end switch
    
    // Only process known key inputs
    if (!unknown_key) {
      // Flush remaining inputs in queue
      tcflush(kfd, TCIFLUSH);
      
      // Make sure steering is within bounds
      steering = std::min(steering, 1.0f);
      steering = std::max(steering, -1.0f);
      
      // Make sure throttle is within bounds
      throttle = std::min(throttle, 1.0f);
      throttle = std::max(throttle, -1.0f);
      
      msg.steering = steering;
      msg.throttle = throttle;
    }// end unknown_key if
    
    // Publish at desired frequency
    if (ros::Time::now() > (prev_pub_time + ros::Duration(1/pub_rate))) {
      prev_pub_time = ros::Time::now();
      msg.header.stamp = ros::Time::now();
      car_control_pub_.publish(msg);
      ROS_DEBUG("Steering=%.2f, Throttle=%.2f", steering, throttle);
    }
    
  }// end while

  return;
}

//------------------------------------

void quit(int sig) {
  tcsetattr(kfd, TCSANOW, &cooked);
  ros::shutdown();
  exit(0);
}// end quit


int main(int argc, char** argv) {
  // Initialize
  ros::init(argc, argv, "teleop_keyboard");
  
  // Create the teleop object
  Teleop teleop;
  
  // Process key inputs
  signal(SIGINT,quit);
  teleop.keyLoop();
  
  return(0);
}


