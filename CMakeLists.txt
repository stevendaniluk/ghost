cmake_minimum_required(VERSION 2.8.3)
project(ghost)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  message_generation
  nodelet
  roscpp
  rosserial_arduino
  rosserial_client
  sensor_msgs
  std_msgs
  tf
)

## Declare custom messages
add_message_files(
  FILES
  CarControl.msg
  ArduinoControl.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

# Declare the catkin package
catkin_package(
  LIBRARIES lane_detector_nodelet
  CATKIN_DEPENDS nodelet roscpp
)

## Specify additional locations of header files
include_directories(
  ${catkin_INCLUDE_DIRS}
)

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
rosserial_add_client_target(arduino controller_interface)
rosserial_add_client_target(arduino controller_interface-upload)

# Keyboard teleop executable
add_executable(teleop_keyboard src/teleop_keyboard.cpp)
add_dependencies(teleop_keyboard ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(teleop_keyboard ${catkin_LIBRARIES})

# PS3 teleop executable
add_executable(teleop_ps3 src/teleop_ps3.cpp)
add_dependencies(teleop_ps3 ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(teleop_ps3 ${catkin_LIBRARIES})

# Car-to-Servo Converter executable
add_executable(car_to_servo_converter src/car_to_servo_converter.cpp)
add_dependencies(car_to_servo_converter ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(car_to_servo_converter ${catkin_LIBRARIES})

# Executable for saving camera images
add_executable(camera_frame_grabber src/camera_frame_grabber.cpp)
target_link_libraries(camera_frame_grabber ${catkin_LIBRARIES})

# Nodelet library
add_library(lane_detector_nodelet src/lane_detector_nodelet.cpp)
target_link_libraries(lane_detector_nodelet ${catkin_LIBRARIES})

# Node wrapper for lane detector nodelet
add_executable(lane_detector_node src/lane_detector_node.cpp)
target_link_libraries(lane_detector_node ${catkin_LIBRARIES})
