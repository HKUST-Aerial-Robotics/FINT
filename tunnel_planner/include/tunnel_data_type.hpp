/*******************************************************
Copyright (C) 2025, Aerial Robotics Group, Hong Kong University of Science and Technology
This file is part of FINT.
Licensed under the GNU General Public License v3.0;
you may not use this file except in compliance with the License.
*******************************************************/
#pragma once
#include <vector>
#include <Eigen/Eigen>

using namespace std;
using namespace Eigen;

namespace tunnel_planner{

enum plan_status{
    ACCEPT = 0,
    SHORT_TRAJECTORY = 1,
    NO_EXTENSION = 2,
    REACH_EXIT = 3,
    UNKNOWN_FAILURE = 4
};

enum yaw_stragety{
    CONSTANT = -1,
    TANGENT = 0,
    CONSTANT_PT = 1,
    CONSTANT_TRAJ_DIST = 2,
    CONSTANT_CHORD_DIST = 3,
    PLAN = 4
};

enum corridor_status{
    NORMAL = 0,
    CORRIDOR_REACH_EXIT = 1,
    NO_CORRIDOR = 2
};

enum tunnel_shape{
    UNKNOWN = 0,
    CIRCLE = 1,
    RECTANGLE = 2,
    IRREGULAR = 3,
    FREE_SHAPE = 4,
    OUTSIDE = -1,
    BEFORE = -2
};

struct cross_section{
    bool is_predict = false;
    int cross_section_shape_ = UNKNOWN;
    Vector3d center_;
    Matrix3d w_R_cs;
    // h,w for rectangle; r for circle
    vector<double> cross_section_data_;

    double curve_length_;
    double curvature_;

    double yaw_;
    double yaw_dot_;
};

struct vert_section{
    double entry_t;
    double entry_length;
    double entry_yaw;

    double exit_t;
    double exit_length;
    double exit_yaw;
};

};