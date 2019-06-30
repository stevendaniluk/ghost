cmake_minimum_required(VERSION 2.8.3)

include_directories(${ROS_LIB_DIR})

generate_arduino_firmware(car_interface
  SRCS car_interface.cpp ${ROS_LIB_DIR}/time.cpp
  HDRS PinChangeInt.h
  BOARD nano328
  PORT /dev/ttyUSB0
)
