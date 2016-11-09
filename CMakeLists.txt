cmake_minimum_required(VERSION 2.8.3)
project(ghost)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  message_generation
  roscpp
  sensor_msgs
  std_msgs
)

# Declare custom messages
add_message_files(
  FILES
  CarControl.msg
  ServoControl.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

# Declare the catkin package
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES 
#  CATKIN_DEPENDS
#  DEPENDS system_lib
)

#set(SOMETHING_HEADER_PATH "../something/include/")

## Specify additional locations of header files
# include_directories(include)
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

# Declare a C++ library
# add_library(foo src/${PROJECT_NAME}/foo.cpp
#)


