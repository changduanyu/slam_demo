cmake_minimum_required(VERSION 2.8.3)
project(slam_from_scratch)
set (CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
find_package(Pangolin REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Pangolin_INCLUDE_DIRS})

add_executable(main main.cpp)
target_link_libraries(main ${OpenCV_LIBS} ${Pangolin_LIBRARIES})