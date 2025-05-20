/*******************************************************
Copyright (C) 2025, Aerial Robotics Group, Hong Kong University of Science and Technology
This file is part of FINT.
Licensed under the GNU General Public License v3.0;
you may not use this file except in compliance with the License.
*******************************************************/

#ifndef _EDF_MAP_FUSION_H
#define _EDF_MAP_FUSION_H

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <iostream>
#include <memory>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>


#include <edf_map/edf_voxel.h>
#include <edf_map/voxel_map.h>
#include <edf_map/occ_map_fusion.h>
#include <parameters.h>

#include <nav_msgs/Odometry.h>

namespace tunnel_planner{

class edf_map_generator{

public:
    edf_map_generator(ros::NodeHandle & n, double& map_res, Matrix<double,3,2>& map_lim);

    shared_ptr<voxel_map<edf_voxel>> get_edf_map_ptr(){ return edf_map_ptr_;}

    void set_latest_odom(const nav_msgs::OdometryConstPtr& odom_ptr){
        odom_mutex_.lock();
        latest_odom_ = *odom_ptr;
        odom_mutex_.unlock();
    }

    void start_process_thread();
    void process_loop();
   
    void cal_update_range(Vector3i& edf_min_coord, Vector3i& edf_max_coord);
    void reset_edf_map();
    void reset_edf_map(const Vector3i& edf_min_coord, const Vector3i& edf_max_coord);
    void set_edf_map_free(const Vector3i& edf_min_coord, const Vector3i& edf_max_coord);
    void update_edf_map(const Vector3i& edf_min_coord, const Vector3i& edf_max_coord);
    void publish_edf(const Vector3i& edf_min_coord, const Vector3i& edf_max_coord, const Vector3d& body_t);

    void get_edf_range(Vector3d& edf_min_pos, Vector3d& edf_max_pos){
        edf_min_pos = edf_min_pos_;
        edf_max_pos = edf_max_pos_;
    }

    double get_dist(const Vector3d& pos);
    double get_dist_grad(const Vector3d& pos, Vector3d& grad);

    bool get_occ(const Vector3d& pos){
        if(!edf_map_ptr_->in_map(pos)){
            return false;
        }
        
        return edf_map_ptr_->map_data[edf_map_ptr_->pos2idx(pos)].type == edf_voxel::OCC;
    }

    int get_type(const Vector3d& pos){
        if(!edf_map_ptr_->in_map(pos)){
            return edf_voxel::UNKNOWN;
        }
        
        return edf_map_ptr_->map_data[edf_map_ptr_->pos2idx(pos)].type;
    }

    void wait_for_edf_available(const double& wait_time){

        edf_flag_mutex_.lock();

        while(true){
            if(accessing_edf_){
                edf_flag_mutex_.unlock();
                this_thread::sleep_for(std::chrono::duration<double>(wait_time));
                edf_flag_mutex_.lock();
            }
            else{
                accessing_edf_ = true;
                break;
            }
        }

        edf_flag_mutex_.unlock();       
    }

    void release_edf_resource(){
        accessing_edf_ = false;
    }

    void start_edf(){
        omf_.start_mapping();
        start_edf_ = true;
    }

    void stop_edf(){
        start_edf_ = false;
        omf_.stop_mapping();
    }

private:

    template <typename F_get_val, typename F_set_val>
    void fill_edf(F_get_val f_get_val, F_set_val f_set_val, int start, int end, int dim);

    std::vector<double> tmp_buffer1_, tmp_buffer2_, tmp_buffer3_;

    std::vector<unsigned int> update_idx_vec_;

    std::shared_ptr<voxel_map<edf_voxel>> edf_map_ptr_;

    Eigen::Vector3d local_range_min_;
    Eigen::Vector3d local_range_max_;

    Eigen::Vector3d edf_min_pos_;
    Eigen::Vector3d edf_max_pos_;

    bool accessing_edf_;
    std::mutex edf_flag_mutex_;

    occ_map_fusion omf_;
    nav_msgs::Odometry latest_odom_;

    std::mutex odom_mutex_;

    std::unique_ptr<std::thread> updade_thread_ptr_;

    atomic<bool> start_edf_;

    ros::Publisher edf_pub_;

};

}
#endif
