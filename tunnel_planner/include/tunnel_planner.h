/*******************************************************
Copyright (C) 2025, Aerial Robotics Group, Hong Kong University of Science and Technology
This file is part of FINT.
Licensed under the GNU General Public License v3.0;
you may not use this file except in compliance with the License.
*******************************************************/

#pragma once

#include <stdlib.h>
#include <time.h>
#include <memory>

#include <cv_bridge/cv_bridge.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <sensor_msgs/PointCloud.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <std_msgs/Int16.h>
#include <geometry_msgs/PoseArray.h>
#include <visualization_msgs/MarkerArray.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>

#include "edf_map/edf_map_generator.h"
#include "kinodynamic_astar.h"
#include "disturbance_estimator/regression_net.hpp"
#include "non_uniform_bspline.h"
#include "bspline_optimizer.h"
#include "tunnel_planner/Bspline_with_retiming.h"

#include "tunnel_data_type.hpp"

#include "hough/hough_circle.hpp"
#include "hough/hough_rectangle.hpp"

namespace tunnel_planner{

class tunnel_planner{

public:
    struct traj_data
    {
        int traj_id_;
        double duration_;
        ros::Time start_time_;
        Eigen::Vector3d start_pos_;
        double start_travel_distance_;
        vector<cross_section> corridor_;
        NonUniformBspline position_traj_, velocity_traj_, acceleration_traj_, yaw_traj_, yaw_dot_traj_;
        NonUniformBspline position_traj_1d_, velocity_traj_1d_, acceleration_traj_1d_;
        double predict_start_length_;
        bool traj_valid_;
    };

    tunnel_planner(ros::NodeHandle& n);

    void replan();

private:

    void start_replan_thread();
    void replan_loop();

    bool get_position_cmd_t_new(ros::Time t, Vector3d& pos, Vector3d& vel, Vector3d& acc, double& yaw, double& yaw_dot, double& curve_length, Vector3d& curvature);

    void latest_odom_callback(const nav_msgs::OdometryConstPtr& odom);
    // void plan_trigger_callback(const geometry_msgs::PoseStamped::ConstPtr &trigger);
    void tunnel_entrance_callback(const geometry_msgs::PoseStamped::ConstPtr &entrance_pose);

    void map_odom_callback(const sensor_msgs::PointCloudConstPtr& pcd, const sensor_msgs::PointCloudConstPtr& pcd_free, const nav_msgs::OdometryConstPtr& odom);

    void map_free_callback(const sensor_msgs::PointCloudConstPtr& pcd_free);

    void before_tunnel_plan_new(const Vector3d& start_pos);

    void tunnel_entrance_plan_new(const double plan_step, const double range, const Vector3d& start_pos, const Vector3d& start_vel, const Vector3d& start_acc, const double& start_yaw, const double& start_yaw_dot, const double& start_curve_length, const Vector3d& start_curvature);

    void in_tunnel_plan(const Vector3d& start_pos, const Vector3d& start_vel, const Vector3d& start_acc, const double& start_yaw, const double& start_yaw_dot = 0.0, const double& start_curve_length = 0.0, const Vector3d& start_curvature = Vector3d::Zero());

    int tunnel_plan_new(const double plan_step, const double range, const Vector3d& start_pos, const Vector3d& start_vel, const Vector3d& start_acc, const double& start_yaw, const double& start_yaw_dot, const double& start_curve_length, const Vector3d& start_curvature);

    NonUniformBspline extract_tunnel_center_line(double ts, const Vector3d& start_curvature);

    bool detect_corridor(double& ts);

    NonUniformBspline plan_tunnel_center_yaw(double ts, const double& start_yaw, const double& start_yaw_dot = 0.0, const bool change_yaw = true);

    NonUniformBspline traj_opt_1D(double& ts, const vector<double>& way_pts, const double& start_v, const double& start_a, const double& end_v);

    bool get_inter_cross_section_curve_length_vertical_cross_section_range(NonUniformBspline& center_line, double& predict_start_t);

    int search_corridor_path_hough(const cross_section& start_cross_section, const double plan_step, double &range, const double max_edf_tol, const double min_dir_tol);

    cross_section detect_cross_section_shape(const cv::Mat &cross_section_img, const Matrix3d& w_R_cs, const Vector3d& w_t_cs, const cross_section &prev_cross_section, bool cross_section_exist);

    bool detect_circle(const cv::Mat &cross_section_img, const Matrix3d& w_R_cs, cross_section &cross_section_result, int& outlier_cnt);

    bool detect_rectangle(const cv::Mat &cross_section_img, const Matrix3d& w_R_cs, cross_section &cross_section_result, int& outlier_cnt);

    int classify_shape(const cv::Mat &cross_section_img);

    double find_max_edf_in_plane(Vector3d& pt_in_plane, const Vector3d& plane_dir, double step, const double max_res, const double max_edf_value = numeric_limits<double>::max());

    Vector3d find_tunnel_dir(const Vector3d& max_edf_pt, const double max_edf_value, const Vector3d& init_tunnel_dir);

    double sphere_descend(Vector3d& pt_on_sphere, const Vector3d& sphere_center, const double radius, double step, const double max_res);
    
    Vector3d plane_dir_fitting(vector<Vector3d>& pts);

    Matrix3d cal_w_R_plane(const Vector3d &dir);

    bool construct_cross_section(const Vector3d &pt, const Vector3d &dir, Matrix3d& w_R_plane, cv::Mat& cross_section_mat);

    double calc_des_yaw(const double& last_yaw, const double& cur_yaw, const double& max_yaw_change);
    double calc_des_yaw(const double& last_yaw, const double& tunnel_dir_yaw, const double& cur_yaw, const double& max_yaw_change, const double& max_yaw_center_line_dir_diff = M_PI_2);

    void pub_corridor(vector<tuple<Vector3d, double, Vector3d>>& plan_corridor);
    void pub_init_corridor();
    void pub_corridor();
    void pub_past_corridor();

    void pub_traj_vis(NonUniformBspline &bspline);

    void pub_traj_init_vis(NonUniformBspline &bspline);

    ros::NodeHandle nh_;

    typedef message_filters::sync_policies::ExactTime<sensor_msgs::PointCloud, sensor_msgs::PointCloud, nav_msgs::Odometry> SyncPolicyMapOdomExact;
    typedef unique_ptr<message_filters::Synchronizer<SyncPolicyMapOdomExact>> SynchronizerMapOdomExact;
    SynchronizerMapOdomExact sync_map_odom_exact_;

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud, sensor_msgs::PointCloud, nav_msgs::Odometry> SyncPolicyMapOdomApproximate;
    typedef unique_ptr<message_filters::Synchronizer<SyncPolicyMapOdomApproximate>> SynchronizerMapOdomApproximate;
    SynchronizerMapOdomApproximate sync_map_odom_approximate_;

    message_filters::Subscriber<sensor_msgs::PointCloud> map_sub_;
    message_filters::Subscriber<sensor_msgs::PointCloud> map_free_sub_;
    message_filters::Subscriber<nav_msgs::Odometry> odom_sub_;


    ros::Subscriber latest_odom_sub_, tunnel_entrance_sub_;//, map_free_sub_, plan_trigger_sub_;

    ros::Publisher edf_pub_, map_reset_pub_, traj_full_pub_, corridor_pub_, corridor_center_pub_, corridor_center_init_pub_, corridor_center_path_pub_, traj_path_init_vis_pub_, traj_path_vis_pub_, traj_state_pub_;
    ros::Publisher past_corridor_pub_, past_corridor_center_pub_, past_corridor_center_path_pub_;
    ros::Publisher cross_section_img_pub_;

    unique_ptr<thread> replan_thread_ptr_;

    double drone_dim_, drone_radius_;
    int drone_radius_pixel_;
    
    double tunnel_dim_;
    int tunnel_dim_pixel_;
    int half_tunnel_dim_pixel_;

    Vector3d tunnel_entrance_pos_;
    Vector3d tunnel_entrance_dir_;

    Vector3d tunnel_exit_pos_;

    unique_ptr<HoughCircle> hough_circle_detector_;
    unique_ptr<HoughRectangle> hough_rectangle_detector_;
    double hough_circle_threshold_, hough_rectangle_threshold_;

    double tunnel_step_res_;
    double cross_section_step_res_;
    double tunnel_way_pt_min_interval_;
    double grad_max_res_;
    double plan_range_;
    double flight_speed_;
    double virtual_flight_progress_speed_;
    double max_speed_;
    double max_acc_;

    bool adaptive_speed_;

    double max_yaw_dir_curvature_ratio_;
    double yaw_ahead_length_;

    double max_yaw_change_over_distance_;
    double max_yaw_center_line_dir_diff_;

    double cmd_offset_z_;

    shared_ptr<NonUniformBspline> tunnel_center_line_;
    shared_ptr<NonUniformBspline> tunnel_center_vel_; 
    shared_ptr<NonUniformBspline> tunnel_center_acc_;

    shared_ptr<NonUniformBspline> tunnel_center_yaw_;
    shared_ptr<NonUniformBspline> tunnel_center_yaw_dot_;

    shared_ptr<vector<cross_section>> plan_corridor_;
    shared_ptr<vector<cross_section>> past_corridor_;

    shared_ptr<vector<vert_section>> vert_sections_;

    shared_ptr<regression_net> circle_net_, rect_net_;

    shared_ptr<nn> shape_classifier_net_;
    Vector3i shape_classifier_input_dim_;

    shared_ptr<optical_flow_estimator> optical_flow_estimator_;

    unique_ptr<optimizer_1d> bspline_optimizer_1d_;

    unique_ptr<BsplineOptimizer> bspline_optimizer_;


    nav_msgs::Odometry latest_odom_;

    shared_ptr<edf_map_generator> edf_map_generator_ptr_;   

    ros::Time last_plan_time_;

    traj_data last_traj_data_;


    int traj_id_;
    int traj_state_;
    enum traj_state{HOVER, BEFORE_TUNNEL, TUNNEL_ENTRANCE, IN_TUNNEL, AFTER_TUNNEL};
    bool in_vertical_section_;
    bool plan_fail_;
    int detect_tunnel_cnt_;
    bool use_bspline_;

    double traj_end_time_;

    ros::Duration time_commit_;
    ros::Time plan_odom_time_;

    unsigned int cross_section_cnt_ = 0;
};
}