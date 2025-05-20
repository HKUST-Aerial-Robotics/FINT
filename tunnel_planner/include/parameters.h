/*******************************************************
Copyright (C) 2025, Aerial Robotics Group, Hong Kong University of Science and Technology
This file is part of FINT.
Licensed under the GNU General Public License v3.0;
you may not use this file except in compliance with the License.
*******************************************************/

#pragma once

#include <ros/ros.h>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>

#include "shape_classifier/cnn.hpp"

using namespace std;
using namespace Eigen;

extern double MAP_RES;
extern Matrix<double,3,2> MAP_LIM;
extern Matrix<double,3,2> LOCAL_EDF_LIM;

extern double PROB_HIT;
extern double PROB_MISS;
extern double CLAMP_MIN;
extern double CLAMP_MAX;
extern double MIN_OCCUPANCY;

extern double UPDATE_FREQ;

extern int DEPTH_MARGIN;
extern int NUM_PIXEL_SKIP;
extern double DEPTH_SCALE;
extern double MIN_DEPTH;
extern double MAX_DEPTH;
extern double MAX_RAY_LENGTH;

extern int ADAPTIVE_SPEED;

extern double DRONE_DIM;
extern double HOUGH_CIRCLE_THRESHOLD, HOUGH_RECTANGLE_THRESHOLD;
extern Vector3d TUNNEL_ENTRANCE_POS;
extern Vector3d TUNNEL_ENTRANCE_DIR;
extern double TUNNEL_DIM;
extern double TUNNEL_STEP_RES;
extern double CROSS_SECTION_STEP_RES;
extern double TUNNEL_WAY_PT_MIN_INTERVAL;
extern double GRAD_MAX_RES;
extern double PLAN_RANGE;
extern double FLIGHT_SPEED;
extern double VIRTUAL_FLIGHT_PROGRESS_SPEED;
extern double MAX_SPEED, MAX_ACC;
extern double MAX_ANGULAR_SPEED;
extern double MAX_YAW_DIR_CURVATURE_RATIO;
extern double YAW_AHEAD_LENGTH;

extern double MAX_YAW_CHANGE_OVER_DISTANCE;
extern double MAX_YAW_CENTER_LINE_DIR_DIFF;

extern double VERT_SECTION_COS_THRESHOLD;

extern vector<pair<Eigen::MatrixXf, Eigen::VectorXf>> CIRCLE_LINEAR_LAYERS;
extern vector<pair<Eigen::MatrixXf, Eigen::VectorXf>> RECT_LINEAR_LAYERS;

extern nn SHAPE_CLASSIFIER_NET;

struct camera_module_info{
    double fx_;
    double fy_;
    double cx_;
    double cy_;

    double depth_fx_;
    double depth_fy_;
    double depth_cx_;
    double depth_cy_;

    Eigen::Matrix4d Tic_;
    int img_rows_;
    int img_cols_;

    Eigen::Matrix3d cam_R_depth_;
    Eigen::Vector3d cam_t_depth_;

    std::string depth_topic_;
    std::string cam_pose_topic_;
};
extern std::vector<camera_module_info> CAM_INFO_VEC;
extern double OPTICAL_FLOW_CAL_RES;
extern double MAX_RAYCAST_LENGTH;

extern int QUAD_ALGORITHM_ID;
extern int NON_QUAD_ALGORITHM_ID;
extern int BSPLINE_DEGREE;

extern double W_DISTURBANCE;
extern double W_VISION;
extern double W_HEURISTIC;

extern double W_SMOOTH_3d;
extern double W_SMOOTH_1d_JERK;
extern double W_SMOOTH_1d_ACC;
extern double W_SMOOTH_YAW;
extern double W_INTERVAL;
extern double W_DIST;
extern double W_FEASI;
extern double W_START;
extern double W_END;
extern double W_END_HARD;
extern double W_GUIDE;
extern double W_WAYPT;
extern double W_TIME;
extern double W_YAW_WAYPT;

extern double DISTANCE_COST_ORIGIN;

extern int MAX_ITERATION_NUM1;
extern int MAX_ITERATION_NUM2;
extern int MAX_ITERATION_NUM3;
extern int MAX_ITERATION_NUM4;
extern double MAX_ITERATION_TIME1;
extern double MAX_ITERATION_TIME2;
extern double MAX_ITERATION_TIME3;
extern double MAX_ITERATION_TIME4;
extern double REPLAN_FREQ;
extern int USE_EXACT_TIME_SYNC;
extern double TIME_COMMIT;

void readParametersNet(std::string config_file, vector<pair<Eigen::MatrixXf, Eigen::VectorXf>>& linear_layers);
nn readParametersCNN(std::string config_file);
void readParameters(std::string config_file);
