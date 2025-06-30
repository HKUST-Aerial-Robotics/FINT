cmake_minimum_required(VERSION 2.8.3)
project(marker_detector)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14 -march=native -DEIGEN_DONT_PARALLELIZE")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    cv_bridge
	message_filters
    )

find_package(OpenCV REQUIRED)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3  REQUIRED)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  include
)

set(CMAKE_MODULE_PATH /usr/local/lib/cmake/)
# find_package(aruco REQUIRED)


catkin_package(
    )

add_executable(${PROJECT_NAME}
    src/marker_detector.cpp
    src/marker_detector_node.cpp)

target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBS} ${aruco_LIBS})

add_executable(entrance_publisher
    src/entrance_publisher.cpp)

target_link_libraries(entrance_publisher ${catkin_LIBRARIES})
