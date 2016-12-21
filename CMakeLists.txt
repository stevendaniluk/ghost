cmake_minimum_required(VERSION 2.8.3)
project(ghost)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  message_generation
  nodelet
  roscpp
  sensor_msgs
  std_msgs
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

## Declare C++ executables
add_executable(teleop_keyboard src/teleop_keyboard.cpp)
add_executable(teleop_ps3 src/teleop_ps3.cpp)
add_executable(car_to_servo_converter src/car_to_servo_converter.cpp)
add_executable(camera_frame_grabber src/camera_frame_grabber.cpp)

# Declare the dependencies
add_dependencies(teleop_keyboard ghost_generate_messages_cpp)
add_dependencies(teleop_ps3 ghost_generate_messages_cpp)
add_dependencies(car_to_servo_converter ghost_generate_messages_cpp)

# Specify libraries
target_link_libraries(teleop_keyboard ${catkin_LIBRARIES})
target_link_libraries(teleop_ps3 ${catkin_LIBRARIES})
target_link_libraries(car_to_servo_converter ${catkin_LIBRARIES})
target_link_libraries(camera_frame_grabber ${catkin_LIBRARIES})

# Nodelet library
add_library(lane_detector_nodelet src/lane_detector_nodelet.cpp)
target_link_libraries(lane_detector_nodelet ${catkin_LIBRARIES})

# Node wrapper for lane detector nodelet
add_executable(lane_detector_node src/lane_detector_node.cpp)
target_link_libraries(lane_detector_node ${catkin_LIBRARIES})
