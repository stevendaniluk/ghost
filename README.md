# Ghost
A ROS package for racing an autonomous RC car.

This project is a work in progress. The goal is to enable an RC car to race autonomously around an on-road RC track (or some other area). You can follow the progress at [my Medium account](https://medium.com/@stevendaniluk), where I am documenting the development process.

The name Ghost comes from the ghost car feature in racing video games.

---

## Project Status
Below are the main project components and their status
* Car/Computer Interface - Complete
* Odometry Estimation - In Progress
* Lane Detection - In Progress, prototype version in place
* Localization - TODO
* Path Tracking Controller - TODO

## Install
All software is developed in ROS Indigo. Perform the commands below from within your ROS workspace.

To download and build:
```
git clone https://github.com/stevendaniluk/ghost.git
catkin_make
```

To build and upload the arduino code:
```
rosrun rosserial_arduino make_libraries.py build/ghost/
catkin_make ghost_arduino_controller_interface
catkin_make ghost_arduino_controller_interface-upload
```

This will build the ghost messages, compile the arduino code, and upload it to the arduino (which should be connected via USB). Note: in order to build the messages for the arduino, your workspace must be re-sourced after the package is built with `catkin_make` (either open a new terminal window, or run `source /workspace_path/devel/setup.bash`).

## Hardware Used
* Kyosho TF-5 Stallion 1/10th On-road RC Car
* Arduino Nano Microcontroller
* Phidgets Spatial 3/3/3 IMU
* PointGrey Blackfly Camera
* Intel NUC i5
