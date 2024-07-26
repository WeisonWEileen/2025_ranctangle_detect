cmake_minimum_required(VERSION 3.0)

project(DianSai)
set(CMAKE_CXX_STANDARD 11)


set(CMAKE_BUILD_TYPE Debug)


find_package(OpenCV REQUIRED)
add_executable(HSV_Filter hsv_white_filtering.cpp)


target_link_libraries(HSV_Filter ${OpenCV_LIBS})


