cmake_minimum_required(VERSION 2.8.3)
project(tunnel_planner)

set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++14 -O3 -Wall")

find_package(OpenCV REQUIRED)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  visualization_msgs
  cv_bridge
  message_filters
  message_generation
)

find_package(Eigen3 REQUIRED)
find_package(PCL 1.7 REQUIRED)

add_message_files(
  FILES
  Bspline_with_retiming.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
 INCLUDE_DIRS include
 CATKIN_DEPENDS roscpp std_msgs
#  DEPENDS system_lib
)

include_directories( 
    SYSTEM 
    include
    ${catkin_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS} 
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})

add_executable(${PROJECT_NAME} 
    src/tunnel_planner_node.cpp
    src/tunnel_planner.cpp
    src/parameters.cpp
    src/edf_map/edf_map_generator.cpp
    src/edf_map/occ_map_fusion.cpp
    src/non_uniform_bspline.cpp
    src/bspline_optimizer.cpp
    src/hough/hough_circle.cpp
    src/hough/hough_rectangle.cpp
    src/disturbance_estimator/regression_net.cpp
    src/kinodynamic_astar.cpp
    src/optical_flow_estimator/optical_flow_estimator.cpp
    src/raycast/raycast.cpp
    src/shape_classifier/cnn.cpp
)

target_link_libraries( ${PROJECT_NAME}
    ${catkin_LIBRARIES} 
    ${PCL_LIBRARIES}
    /usr/local/lib/libnlopt.so
)

add_dependencies(${PROJECT_NAME} ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)

add_executable(b_spline_traj_server
    src/traj_server.cpp
    src/non_uniform_bspline.cpp
)

target_link_libraries(b_spline_traj_server
    ${catkin_LIBRARIES} 
    ${PCL_LIBRARIES}
)  

add_dependencies(b_spline_traj_server ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
