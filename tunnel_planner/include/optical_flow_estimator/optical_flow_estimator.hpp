/*******************************************************
Copyright (C) 2025, Aerial Robotics Group, Hong Kong University of Science and Technology
This file is part of FINT.
Licensed under the GNU General Public License v3.0;
you may not use this file except in compliance with the License.
*******************************************************/

#pragma once

#include <Eigen/Eigen>
#include "tunnel_data_type.hpp"
#include "edf_map/edf_map_generator.h"
#include "raycast/raycast.hpp"

#include <visualization_msgs/MarkerArray.h>

using namespace std;
using namespace Eigen;

namespace tunnel_planner{

class optical_flow_estimator{
public:

    enum intersection_status{
        INTERSECT_TUNNEL = 0,
        NO_ENTRY = 1,
        PASS_THROUGH = 2,
        REVERSE_DIR = -1,
        INTERSECTION_ERROR = -2
    };

    struct camera_wise_data{
        camera_wise_data(const camera_module_info& cam_param, double cal_res): 
            fx_(cam_param.fx_), fy_(cam_param.fy_), cx_(cam_param.cx_), cy_(cam_param.cy_), 
            ric_(cam_param.Tic_.topLeftCorner(3,3)), tic_(cam_param.Tic_.topRightCorner(3,1)),
            img_rows_(cam_param.img_rows_), img_cols_(cam_param.img_cols_)
        { 
            cal_rows_ = static_cast<int>(ceil(1.0 * img_rows_ / cal_res));
            cal_cols_ = static_cast<int>(ceil(1.0 * img_cols_ / cal_res));

            cal_row_idx_.resize(cal_rows_);
            cal_col_idx_.resize(cal_cols_);

            double res_row = 1.0 * img_rows_ / cal_rows_;
            double res_col = 1.0 * img_cols_ / cal_cols_;

            cal_row_idx_ = VectorXd::LinSpaced(cal_rows_, 0.5 * res_row, img_rows_ - 0.5 * res_row);
            cal_col_idx_ = VectorXd::LinSpaced(cal_cols_, 0.5 * res_col, img_cols_ - 0.5 * res_col);

            inv_proj_dir_mat_.resize(cal_rows_, cal_cols_);

            for(int row_idx = 0; row_idx < cal_rows_; row_idx++){
                for(int col_idx = 0; col_idx < cal_cols_; col_idx++){
                    Vector3d cam_plane_inv_proj_pt = inv_projection(cal_col_idx_(col_idx), cal_row_idx_(row_idx), 1.0);
                    inv_proj_dir_mat_(row_idx, col_idx) = cam_plane_inv_proj_pt.normalized();
                }
            }

            cross_section_dist_vector_.assign(30, MatrixXd::Zero(cal_rows_, cal_cols_));
        }

        Vector3d inv_projection(const double& u, const double& v, const double& depth = 1.0){
            Vector3d pt;
            pt(0) = (u - cx_) * depth / fx_;
            pt(1) = (v - cy_) * depth / fy_;
            pt(2) = depth;
            return pt;
        }
        double fx_, fy_, cx_, cy_;
        Matrix3d ric_;
        Vector3d tic_;
        int img_rows_;
        int img_cols_;
        int cal_rows_, cal_cols_;
        VectorXd cal_row_idx_, cal_col_idx_;
        Matrix<Vector3d, Dynamic, Dynamic> inv_proj_dir_mat_;
        vector<MatrixXd> cross_section_dist_vector_;
    };
    optical_flow_estimator(ros::NodeHandle& n, const vector<camera_module_info>& camera_parameter_vec, shared_ptr<edf_map_generator> edf_map_generator_ptr,const shared_ptr<vector<cross_section>>& forward_corridor, const shared_ptr<vector<cross_section>>& backward_corridor, const double max_raycast_length, const int cal_res = 80);

    void set_forward_cross_sections();

    void cal_depth();

    double cal_total_mean_optical_flow(int cross_section_idx, double v, double yaw_dot = 0.0);

    double cal_mean_optical_flow(int cam_id, int cross_section_idx, double v, double yaw_dot);

    double cal_mean_optical_flow(int cam_id, int cross_section_idx, Vector3d& v, Vector3d& omega);


private:
    int intersect_tunnel(const Vector3d& ray_cast_start_pt, const Vector3d& ray_cast_dir, const cross_section& cross_section, double cross_section_length, bool forward_dir, double& dist);

    double intersect_forward_tunnel(const Vector3d& ray_cast_start_pt, const Vector3d& ray_cast_dir);

    double intersect_backward_tunnel(const Vector3d& ray_cast_start_pt, const Vector3d& ray_cast_dir);

    void pub_raycast_result();

    double res_;
    int num_cam_;
    vector<camera_wise_data> cam_data_vec_;
    vector<double> cross_section_yaw;
    // depth at cross sections
    int num_cross_section_;
    const shared_ptr<vector<cross_section>> forward_cross_sections_ptr_;
    const shared_ptr<vector<cross_section>> backward_cross_sections_ptr_;

    shared_ptr<edf_map_generator> edf_map_generator_ptr_;
    RayCaster rc_;
    double max_raycast_length_;


    ros::NodeHandle nh_;
    ros::Publisher ray_publisher_;
};

}