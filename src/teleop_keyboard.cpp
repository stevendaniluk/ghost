/* Reads in arrow keystrokes, and publishes car control commands

   Simply increments the steering angle and velocity, then publishes it as
   a CarControl message.
*/

#include <ros/ros.h>
#include <signal.h>
#include <termios.h>
#include <stdio.h>
#include <ghost/CarControl.h>

#define KEYCODE_R 0x43 
#define KEYCODE_L 0x44
#define KEYCODE_U 0x41
#define KEYCODE_D 0x42
#define KEYCODE_S 0x20
#define KEYCODE_Q 0x71

//------------------------------------

class Teleop {
  public:
    Teleop();
    void keyLoop();
    
    // For holding loaded parameters
    double vel_max_;
    double max_steering_angle_;
    
    // Message to be published
    ghost::CarControl msg;

  private:
  ros::NodeHandle nh_;
  ros::Publisher car_control_pub_;
  
};// end class

// Constructor
Teleop::Teleop() {
  // Get relevant parameters
  nh_.param("vel_max", vel_max_, 10.0);
  nh_.param("max_steering_angle", max_steering_angle_, 40.0);
  
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
  
  int steering_angle = 0;
  float velocity = 0;

  // get the console in raw mode                          
  tcgetattr(kfd, &cooked);
  memcpy(&raw, &cooked, sizeof(struct termios));
  raw.c_lflag &=~ (ICANON | ECHO);
  
  // Setting a new line, then end of file                         
  raw.c_cc[VEOL] = 1;
  raw.c_cc[VEOF] = 2;
  tcsetattr(kfd, TCSANOW, &raw);
  
  puts(" ");
  puts("---------------------------");
  puts("Reading from keyboard");
  puts("Use arrow keys to move the car.");
  puts(" ");
  puts("Up Arrow = Increase Speed");
  puts("Down Arrow = Decrease Speed");
  puts("Left Arrow = Increment Steering Angle Left");
  puts("Right Arrow = Increment Steering Angle Right");

  for(;;) {
    // Get the next event from the keyboard  
    if(read(kfd, &key, 1) < 0) {
      perror("read():");
      exit(-1);
    }

    ROS_DEBUG("value: 0x%02X\n", key);
    unknown_key = false;
    switch(key) {
      case KEYCODE_L:
        steering_angle -= 2;
        break;
      case KEYCODE_R:
        steering_angle += 2;
        break;
      case KEYCODE_U:
        velocity += 0.5;
        break;
      case KEYCODE_D:
        velocity -= 0.5;
        break;
      case KEYCODE_S:
  	velocity = 0;
        break;
      default:
        unknown_key = true;
        break;
    }
    
    // Only process known key inputs
      if (!unknown_key) {
      // Make sure steering is within bounds
      if (steering_angle > max_steering_angle_) {
        steering_angle = max_steering_angle_;
      } else if (steering_angle < -max_steering_angle_) {
        steering_angle = -max_steering_angle_;
      }// end angle if
      
      // Make sure velocity is within bounds
      if (velocity > vel_max_) {
        velocity = vel_max_;
      } else if (velocity < 0) {
        velocity = 0;
      }// end if
      
      // Publish the message
      msg.steering_angle = steering_angle;
      msg.velocity = velocity;
      car_control_pub_.publish(msg);
    }// end unknown_key if

  }// end for

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



