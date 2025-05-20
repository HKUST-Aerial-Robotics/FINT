/*******************************************************
Copyright (C) 2025, Aerial Robotics Group, Hong Kong University of Science and Technology
This file is part of FINT.
Licensed under the GNU General Public License v3.0;
you may not use this file except in compliance with the License.
*******************************************************/

#ifndef _OCC_MAP_FUSION_H
#define _OCC_MAP_FUSION_H

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <cv_bridge/cv_bridge.h>
#include <iostream>
#include <memory>
#include <queue>
#include <mutex>
#include <thread>
#include <chrono>

#include <sensor_msgs/Image.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/PointCloud.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>

#include <edf_map/edf_voxel.h>
#include <edf_map/voxel_map.h>
#include <raycast/raycast.hpp>
#include <parameters.h>

#define logit(x) (log((x) / (1 - (x))))

using namespace std;
using namespace Eigen;

namespace tunnel_planner{

class occ_map_fusion{

public:

    struct project_param{
        int depth_margin;
        int num_pixel_skip;
        double depth_scale;
        double min_depth;
        double max_depth;
        double max_ray_length;
    };

    struct fusion_param{
        int prob_hit_log;
        int prob_miss_log;
        int clamp_min_log;
        int clamp_max_log;
        int min_occupancy_log;
    };

    virtual void Init(ros::NodeHandle & n, shared_ptr<voxel_map<edf_voxel>> map_ptr);

    void start_mapping(){
        start_mapping_ = true;
    }

    void stop_mapping(){
        start_mapping_ = false;
    }

protected:

    typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, geometry_msgs::PoseStamped> SyncPolicyImagePoseExact;
    typedef unique_ptr<message_filters::Synchronizer<SyncPolicyImagePoseExact>> SynchronizerImagePoseExact;

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, geometry_msgs::PoseStamped> SyncPolicyImagePoseApproximate;
    typedef unique_ptr<message_filters::Synchronizer<SyncPolicyImagePoseApproximate>> SynchronizerImagePoseApproximate;

    struct depth_pose_sub_ptr{
        message_filters::Subscriber<sensor_msgs::Image>* depth_sub_ptr_;
        message_filters::Subscriber<geometry_msgs::PoseStamped>* cam_pose_sub_ptr_;
        SynchronizerImagePoseExact sync_image_pose_exact_;
        SynchronizerImagePoseApproximate sync_image_pose_approximate_;
    };

    class camera_module_info_with_sub{
        public:

            camera_module_info_with_sub(const camera_module_info cam_module, occ_map_fusion* fusion_ptr): module_info_(cam_module), fusion_ptr_(fusion_ptr), update_flag_(false){}

            depth_pose_sub_ptr depth_pose_sub_ptr_;

            camera_module_info module_info_;

            unsigned int unique_id_;

            occ_map_fusion* fusion_ptr_;
        
            void depth_pose_callback(const sensor_msgs::ImageConstPtr& img, const geometry_msgs::PoseStampedConstPtr& pose);

            void fectch_data_for_update();

            ros::Time pose_time_, pose_time_input_;
            cv::Mat depth_img_, depth_img_input_;
            Vector3d cam_t_, cam_t_input_;
            Matrix3d cam_R_, cam_R_input_;

            bool process_flag_, update_flag_;

            RayCaster raycaster_;

            mutex m_data_;
    };

    

    vector<shared_ptr<camera_module_info_with_sub>> camera_modules_;

    ros::Publisher map_pub_, map2_pub_, free_map_pub_;

    ros::Timer update_timer_;

    void set_modules();
    void set_param();
    void register_sub(ros::NodeHandle &n);
    void register_pub(ros::NodeHandle &n);
    void start_process_thread();
    void process_loop();
    void pub_loop();

    void update_occ_map();

    void update_occ_map_3d(const shared_ptr<camera_module_info_with_sub>& cam_module);


    void project_depth(pcl::PointCloud<Vector3d>& pcd_world, Vector3d& min_bound, Vector3d& max_bound, const camera_module_info_with_sub& cam_module);
    void raycast_process(const pcl::PointCloud<Vector3d>& pcd_world, const Vector3d& min_bound, const Vector3d& max_bound, camera_module_info_with_sub& cam_module);
    void update_voxel_occ(const unsigned int vox_idx, const bool hit);

    void pub_occ_map();

    ros::Time map_time_;

    Vector3d latest_cam_t_;

    project_param project_param_;
    fusion_param fusion_param_;

    shared_ptr<voxel_map<edf_voxel>> map_ptr_;

    pcl::PointCloud<pcl::PointXYZ> cloud_;

    unsigned int last_pcd_size_ = 0;

    atomic<bool> publish_pcd_;
    atomic<bool> start_mapping_;

    unique_ptr<thread> updade_thread_ptr_;
    unique_ptr<thread> pub_thread_ptr_;

    friend class camera_module_info_with_sub;
};

}

#endif