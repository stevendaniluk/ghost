cmake_minimum_required(VERSION 2.8.3)
project(ghost)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  dynamic_reconfigure
  image_transport
  message_generation
  nav_msgs
  nodelet
  pcl_ros
  roscpp
  rosserial_arduino
  rosserial_client
  sensor_msgs
  std_msgs
  tf
)

################################################
## Declare ROS messages, services and actions ##
################################################

add_message_files(
  FILES
  CarControl.msg
  CarControlStamped.msg
  ControlState.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

generate_dynamic_reconfigure_options(
  cfg/Controller.cfg
  cfg/LaneDetector.cfg
)

###################################
## catkin specific configuration ##
###################################

catkin_package(
  LIBRARIES lane_detector_nodelet
  CATKIN_DEPENDS nodelet roscpp
)

## Specify additional locations of header files
include_directories(
  ${catkin_INCLUDE_DIRS}
  include
)

#############
## Arduino ##
#############

# Library generation for Arduino
rosserial_generate_ros_lib(
  PACKAGE rosserial_arduino
  SCRIPT make_libraries.py
)
add_dependencies(${PROJECT_NAME}_ros_lib ${PROJECT_NAME}_generate_messages_cpp)

# Arduino target setup
rosserial_configure_client(
  DIRECTORY arduino
  TOOLCHAIN_FILE ${ROSSERIAL_ARDUINO_TOOLCHAIN}
)
rosserial_add_client_target(arduino car_interface)
rosserial_add_client_target(arduino car_interface-upload)

###########
## Build ##
###########

# Odometry library
add_library(car_odometry
  src/car_odometry.cpp
)
target_link_libraries(car_odometry
  ${catkin_LIBRARIES}
)

# Keyboard teleop executable
add_executable(teleop_keyboard src/teleop_keyboard.cpp)
add_dependencies(teleop_keyboard ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(teleop_keyboard ${catkin_LIBRARIES})

# PS3 teleop executable
add_executable(teleop_ps3 src/teleop_ps3.cpp)
add_dependencies(teleop_ps3 ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(teleop_ps3 ${catkin_LIBRARIES})

# Odometry integration executable
add_executable(odometry_integration src/odometry_integration.cpp)
add_dependencies(odometry_integration ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(odometry_integration ${catkin_LIBRARIES})

# Lane detector nodelet
add_library(lane_detector_nodelet src/lane_detector_nodelet.cpp)
add_dependencies(lane_detector_nodelet ${PROJECT_NAME}_gencfg)
target_link_libraries(lane_detector_nodelet ${catkin_LIBRARIES})

# Node wrapper for lane detector nodelet
add_executable(lane_detector_node src/lane_detector_node.cpp)
add_dependencies(lane_detector_node ${PROJECT_NAME}_gencfg)
target_link_libraries(lane_detector_node ${catkin_LIBRARIES})

# Controller executable
add_executable(controller src/controller.cpp)
add_dependencies(controller ${PROJECT_NAME}_gencfg)
target_link_libraries(controller ${catkin_LIBRARIES})

# CarCommand to twist message converter executable
add_executable(cmd_car_to_twist src/cmd_car_to_twist.cpp)
target_link_libraries(cmd_car_to_twist ${catkin_LIBRARIES})

#############
## Testing ##
#############

# Unit test for car odometry library
add_executable(test_car_odometry test/test_car_odometry.cpp)
gtest_add_tests(test_car_odometry "" test/test_car_odometry.cpp)
target_link_libraries(test_car_odometry
    car_odometry
    ${GTEST_BOTH_LIBRARIES}
    ${catkin_LIBRARIES}
)
