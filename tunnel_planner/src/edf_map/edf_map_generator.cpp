/*******************************************************
Copyright (C) 2025, Aerial Robotics Group, Hong Kong University of Science and Technology
This file is part of FINT.
Licensed under the GNU General Public License v3.0;
you may not use this file except in compliance with the License.
*******************************************************/

#include <edf_map/edf_map_generator.h>

namespace tunnel_planner{

edf_map_generator::edf_map_generator(ros::NodeHandle & n, double& map_res, Matrix<double,3,2>& map_lim){

    
    edf_map_ptr_ = make_shared<voxel_map<edf_voxel>>(map_res, map_lim);

    local_range_min_ = LOCAL_EDF_LIM.col(0);
    local_range_max_ = LOCAL_EDF_LIM.col(1);

    tmp_buffer1_.assign(edf_map_ptr_->map_data.size(), 0.0);
    tmp_buffer2_.assign(edf_map_ptr_->map_data.size(), 0.0);

    tmp_buffer3_.assign(edf_map_ptr_->map_data.size(), 0.0);
    update_idx_vec_.reserve(edf_map_ptr_->map_data.size());

    edf_pub_ = n.advertise<sensor_msgs::PointCloud2>("edf", 10);

    start_edf_ = false;
    accessing_edf_ = false;

    omf_.Init(n, edf_map_ptr_);

    start_process_thread();
}

void edf_map_generator::start_process_thread(){
    updade_thread_ptr_ = make_unique<thread>(&edf_map_generator::process_loop, this);
}

void edf_map_generator::process_loop(){

    const std::chrono::duration<double> max_update_interval(1.0 / UPDATE_FREQ);

    std::chrono::time_point<std::chrono::system_clock> start_time;
    std::chrono::duration<double> process_time;

    Eigen::Vector3i edf_min_coord;
    Eigen::Vector3i edf_max_coord;

    while(true){
        start_time = std::chrono::system_clock::now();

        if(start_edf_){
            cal_update_range(edf_min_coord, edf_max_coord);
            update_edf_map(edf_min_coord, edf_max_coord);

            edf_min_pos_ = edf_map_ptr_->coord2pos(edf_min_coord);
            edf_max_pos_ = edf_map_ptr_->coord2pos(edf_max_coord);

            publish_edf(edf_min_coord, edf_max_coord, Vector3d(latest_odom_.pose.pose.position.x, latest_odom_.pose.pose.position.y, latest_odom_.pose.pose.position.z));
        }
        process_time = std::chrono::system_clock::now() - start_time;

        if(process_time < max_update_interval){
            this_thread::sleep_for(max_update_interval - process_time);
        }
        else{
            this_thread::sleep_for(std::chrono::duration<double>(1e-3));
        }

    }
}


void edf_map_generator::cal_update_range(Vector3i& edf_min_coord, Vector3i& edf_max_coord){

    odom_mutex_.lock();

    Eigen::Vector3d cur_pos(latest_odom_.pose.pose.position.x, latest_odom_.pose.pose.position.y,latest_odom_.pose.pose.position.z);
    Eigen::Vector3d cur_vel(latest_odom_.twist.twist.linear.x, latest_odom_.twist.twist.linear.y,latest_odom_.twist.twist.linear.z);
    double cur_yaw = atan2(2 * (latest_odom_.pose.pose.orientation.w * latest_odom_.pose.pose.orientation.z + latest_odom_.pose.pose.orientation.x * latest_odom_.pose.pose.orientation.y), 1 - 2 * (latest_odom_.pose.pose.orientation.y * latest_odom_.pose.pose.orientation.y + latest_odom_.pose.pose.orientation.z * latest_odom_.pose.pose.orientation.z));

    odom_mutex_.unlock();

    Eigen::Vector3d margin(0.1, 0.1, 0.1);
    Eigen::Vector3d edf_min_pos;
    Eigen::Vector3d edf_max_pos;


    if(cur_vel.squaredNorm() > 0.04){
        // cal range according to vel dir
        Vector3d x_dir = cur_vel.normalized();
        Vector3d y_dir = Vector3d::UnitY();
        Vector3d z_dir = Vector3d::UnitZ();

        // FLU coord
        // if X ~ up/down, FY? coord

        if(abs(x_dir.z()) > 0.99){
            // X ~ approx z dir
            z_dir = (x_dir.cross(y_dir)).normalized();
            y_dir = (z_dir.cross(x_dir)).normalized();
            
        }
        else{

            y_dir = (Vector3d::UnitZ().cross(x_dir)).normalized();
            z_dir = (x_dir.cross(y_dir)).normalized();
        }

        Eigen::Matrix3d w_R_local;

        w_R_local.col(0) = x_dir;
        w_R_local.col(1) = y_dir;
        w_R_local.col(2) = z_dir;

        Eigen::Matrix<double, 3, 8> corners;
        corners.col(0) = w_R_local * Vector3d(local_range_min_.x(), local_range_min_.y(), local_range_min_.z());
        corners.col(1) = w_R_local * Vector3d(local_range_min_.x(), local_range_min_.y(), local_range_max_.z());
        corners.col(2) = w_R_local * Vector3d(local_range_min_.x(), local_range_max_.y(), local_range_min_.z());
        corners.col(3) = w_R_local * Vector3d(local_range_min_.x(), local_range_max_.y(), local_range_max_.z());
        corners.col(4) = w_R_local * Vector3d(local_range_max_.x(), local_range_min_.y(), local_range_min_.z());
        corners.col(5) = w_R_local * Vector3d(local_range_max_.x(), local_range_min_.y(), local_range_max_.z());
        corners.col(6) = w_R_local * Vector3d(local_range_max_.x(), local_range_max_.y(), local_range_min_.z());
        corners.col(7) = w_R_local * Vector3d(local_range_max_.x(), local_range_max_.y(), local_range_max_.z());

        Eigen::Vector3d local_min = corners.rowwise().minCoeff();
        Eigen::Vector3d local_max = corners.rowwise().maxCoeff();

        edf_min_pos = cur_pos + local_min - margin;
        edf_max_pos = cur_pos + local_max + margin;

    }
    else{
        // cal range according to yaw dir
        Eigen::Matrix2d xy_R;
        xy_R << cos(cur_yaw), -sin(cur_yaw), sin(cur_yaw), cos(cur_yaw);

        Eigen::Matrix<double, 2, 4> corners;
        corners.col(0) = xy_R * local_range_min_.topRows(2);
        corners.col(1) = xy_R * local_range_max_.topRows(2);
        corners.col(2) = xy_R * Vector2d(local_range_min_.x(), local_range_max_.y());
        corners.col(3) = xy_R * Vector2d(local_range_max_.x(), local_range_min_.y());

        Eigen::Vector2d local_min_xy = corners.rowwise().minCoeff();
        Eigen::Vector2d local_max_xy = corners.rowwise().maxCoeff();

        edf_min_pos = Vector3d(cur_pos.x() + local_min_xy.x(), cur_pos.y() + local_min_xy.y(), cur_pos.z() + local_range_min_.z()) - margin;
        edf_max_pos = Vector3d(cur_pos.x() + local_max_xy.x(), cur_pos.y() + local_max_xy.y(), cur_pos.z() + local_range_max_.z()) + margin;
    }    

    double half_map_res = 0.5 * edf_map_ptr_->map_res;
    edf_min_pos.x() = max(edf_map_ptr_->xmin + half_map_res, edf_min_pos.x());
    edf_min_pos.y() = max(edf_map_ptr_->ymin + half_map_res, edf_min_pos.y());
    edf_min_pos.z() = max(edf_map_ptr_->zmin + half_map_res, edf_min_pos.z());
    edf_min_coord = edf_map_ptr_->pos2coord(edf_min_pos);

    edf_max_pos.x() = min(edf_map_ptr_->xmax - half_map_res, edf_max_pos.x());
    edf_max_pos.y() = min(edf_map_ptr_->ymax - half_map_res, edf_max_pos.y());
    edf_max_pos.z() = min(edf_map_ptr_->zmax - half_map_res, edf_max_pos.z());
    edf_max_coord = edf_map_ptr_->pos2coord(edf_max_pos);

}

void edf_map_generator::reset_edf_map(){
    reset_edf_map(Vector3i::Zero(), edf_map_ptr_->map_size - Vector3i::Ones());
}

void edf_map_generator::reset_edf_map(const Vector3i& edf_min_coord, const Vector3i& edf_max_coord){
    for(int x = edf_min_coord.x(); x <= edf_max_coord.x(); x++)
        for(int y = edf_min_coord.y(); y <= edf_max_coord.y(); y++)
            for(int z = edf_min_coord.z(); z <= edf_max_coord.z(); z++){
                unsigned int idx = edf_map_ptr_->coord2idx(x,y,z);
                edf_map_ptr_->map_data[idx].type = edf_voxel::UNKNOWN;
                // edf_map_ptr_->map_data[idx].type = edf_voxel::FREE;
                edf_map_ptr_->map_data[idx].edf_value = 0.0;
            }
}

void edf_map_generator::set_edf_map_free(const Vector3i& edf_min_coord, const Vector3i& edf_max_coord){
    for(int x = edf_min_coord.x(); x <= edf_max_coord.x(); x++)
        for(int y = edf_min_coord.y(); y <= edf_max_coord.y(); y++)
            for(int z = edf_min_coord.z(); z <= edf_max_coord.z(); z++){
                unsigned int idx = edf_map_ptr_->coord2idx(x,y,z);
                edf_map_ptr_->map_data[idx].type = edf_voxel::FREE;
                edf_map_ptr_->map_data[idx].edf_value = 0.0;
            }
}

void edf_map_generator::update_edf_map(const Vector3i& edf_min_coord, const Vector3i& edf_max_coord){
    // ROS_WARN("update edf");
    
    ros::Time t0 = ros::Time::now();
    update_idx_vec_.clear();
    /* ========== compute positive DT ========== */

    for (int x = edf_min_coord.x(); x <= edf_max_coord.x(); x++) {
        for (int y = edf_min_coord.y(); y <= edf_max_coord.y(); y++) {
        fill_edf(
            [&](int z) {
                return edf_map_ptr_->map_data[edf_map_ptr_->coord2idx(x,y,z)].type == edf_voxel::OCC ?
                    0 :
                    std::numeric_limits<double>::max();
            },
            [&](int z, double val) {unsigned int idx = edf_map_ptr_->coord2idx(x, y, z); update_idx_vec_.emplace_back(idx); tmp_buffer1_[idx] = val; }, edf_min_coord.z(),
            edf_max_coord.z(), 2);
        }
    }

    for (int x = edf_min_coord.x(); x <= edf_max_coord.x(); x++) {
        for (int z = edf_min_coord.z(); z <= edf_max_coord.z(); z++) {
        fill_edf([&](int y) { return tmp_buffer1_[edf_map_ptr_->coord2idx(x, y, z)]; },
                [&](int y, double val) { tmp_buffer2_[edf_map_ptr_->coord2idx(x, y, z)] = val; }, edf_min_coord.y(),
                edf_max_coord.y(), 1);
        }
    }


    for (int y = edf_min_coord.y(); y <= edf_max_coord.y(); y++) {
        for (int z = edf_min_coord.z(); z <= edf_max_coord.z(); z++) {
        fill_edf([&](int x) { return tmp_buffer2_[edf_map_ptr_->coord2idx(x, y, z)]; },
                [&](int x, double val) {
                    tmp_buffer3_[edf_map_ptr_->coord2idx(x, y, z)] = edf_map_ptr_->map_res * std::sqrt(val);
                },
                edf_min_coord.x(), edf_max_coord.x(), 0);
        }
    }

    t0 = ros::Time::now();

    wait_for_edf_available(5e-4);

    
    for (auto& idx: update_idx_vec_){
        edf_map_ptr_->map_data[idx].edf_value = tmp_buffer3_[idx];
    }
    release_edf_resource();

}

template <typename F_get_val, typename F_set_val>
void edf_map_generator::fill_edf(F_get_val f_get_val, F_set_val f_set_val, int start, int end, int dim) {
  int v[edf_map_ptr_->map_size(dim)];
  double z[edf_map_ptr_->map_size(dim) + 1];

  int k = start;
  v[start] = start;
  z[start] = -std::numeric_limits<double>::max();
  z[start + 1] = std::numeric_limits<double>::max();

  for (int q = start + 1; q <= end; q++) {
    k++;
    double s;

    do {
      k--;
      s = ((f_get_val(q) + q * q) - (f_get_val(v[k]) + v[k] * v[k])) / (2 * q - 2 * v[k]);
    } while (s <= z[k]);

    k++;

    v[k] = q;
    z[k] = s;
    z[k + 1] = std::numeric_limits<double>::max();
  }

  k = start;

  for (int q = start; q <= end; q++) {
    while (z[k + 1] < q) k++;
    double val = (q - v[k]) * (q - v[k]) + f_get_val(v[k]);
    f_set_val(q, val);
  }
}


void edf_map_generator::publish_edf(const Vector3i& edf_min_coord, const Vector3i& edf_max_coord, const Vector3d& body_t) {
  double dist;
  pcl::PointCloud<pcl::PointXYZI> cloud;
  pcl::PointXYZI pt;

  if(!(edf_map_ptr_->in_map(body_t))){
    return;
  }

  Vector3i body_coord = edf_map_ptr_->pos2coord(body_t);

  const double min_dist = 0.0;
  const double max_dist = 1.0;

  for (int x = edf_min_coord(0); x <= edf_max_coord(0); ++x)
    for (int y = edf_min_coord(1); y <= edf_max_coord(1); ++y) {

      Vector3d pos = edf_map_ptr_->coord2pos(Vector3i(x,y,0));
      

      dist = edf_map_ptr_->map_data[edf_map_ptr_->coord2idx(x,y,body_coord.z())].edf_value;
      dist = min(dist, max_dist);
      dist = max(dist, min_dist);

      pt.x = pos(0);
      pt.y = pos(1);
      pt.z = body_t.z();
      pt.intensity = (dist - min_dist) / (max_dist - min_dist);
      cloud.push_back(pt);
    }

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = "world";
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);

  edf_pub_.publish(cloud_msg);

}

double edf_map_generator::get_dist(const Vector3d& pos){
    if(!edf_map_ptr_->in_map(pos)){
        return 0.0;
    }

    Vector3i coord_m = edf_map_ptr_->pos2coord(pos - 0.5 * edf_map_ptr_->map_res * Vector3d::Ones());

    Vector3d pos_m = edf_map_ptr_->coord2pos(coord_m);

    Vector3d diff = (pos - pos_m) * edf_map_ptr_->inv_map_res;

    double edf_values[2][2][2];
    for (int x = 0; x < 2; x++) {
        for (int y = 0; y < 2; y++) {
            for (int z = 0; z < 2; z++) {
                Vector3i current_idx = coord_m + Vector3i(x, y, z);
                edf_values[x][y][z] = edf_map_ptr_->map_data[edf_map_ptr_->coord2idx(current_idx)].edf_value;
            }
        }
    }

    double v00 = (1 - diff(0)) * edf_values[0][0][0] + diff(0) * edf_values[1][0][0];
    double v01 = (1 - diff(0)) * edf_values[0][0][1] + diff(0) * edf_values[1][0][1];
    double v10 = (1 - diff(0)) * edf_values[0][1][0] + diff(0) * edf_values[1][1][0];
    double v11 = (1 - diff(0)) * edf_values[0][1][1] + diff(0) * edf_values[1][1][1];
    double v0 = (1 - diff(1)) * v00 + diff(1) * v10;
    double v1 = (1 - diff(1)) * v01 + diff(1) * v11;

    return (1 - diff(2)) * v0 + diff(2)* v1;
}

double edf_map_generator::get_dist_grad(const Vector3d& pos, Vector3d& grad){

    Vector3i coord_m = edf_map_ptr_->pos2coord(pos - 0.5 * edf_map_ptr_->map_res * Vector3d::Ones());

    Vector3d pos_m = edf_map_ptr_->coord2pos(coord_m);
    Vector3d pos_check = pos_m + edf_map_ptr_->map_res * Vector3d::Ones();

    if (!edf_map_ptr_->in_map(pos_check) || !edf_map_ptr_->in_map(pos_m))
    {
        ROS_DEBUG_STREAM("edf out of map at "<< pos.transpose());
        grad.setZero();
        return 0.0;
    }

    Vector3d diff = (pos - pos_m) * edf_map_ptr_->inv_map_res;

    double edf_values[2][2][2];
    for (int x = 0; x < 2; x++) {
        for (int y = 0; y < 2; y++) {
            for (int z = 0; z < 2; z++) {
                Vector3i current_idx = coord_m + Vector3i(x, y, z);
                edf_values[x][y][z] = edf_map_ptr_->map_data[edf_map_ptr_->coord2idx(current_idx)].edf_value;
            }
        }
    }

    double v00 = (1 - diff(0)) * edf_values[0][0][0] + diff(0) * edf_values[1][0][0];
    double v01 = (1 - diff(0)) * edf_values[0][0][1] + diff(0) * edf_values[1][0][1];
    double v10 = (1 - diff(0)) * edf_values[0][1][0] + diff(0) * edf_values[1][1][0];
    double v11 = (1 - diff(0)) * edf_values[0][1][1] + diff(0) * edf_values[1][1][1];
    double v0 = (1 - diff(1)) * v00 + diff(1) * v10;
    double v1 = (1 - diff(1)) * v01 + diff(1) * v11;
    

    grad.z() = (v1 - v0) * edf_map_ptr_->inv_map_res;
    grad.y() = ((1 - diff(2)) * (v10 - v00) + diff(2) * (v11 - v01)) * edf_map_ptr_->inv_map_res;

    grad.x() = ((1 - diff(2)) * (1 - diff(1)) * (edf_values[1][0][0] - edf_values[0][0][0])
            + (1 - diff(2)) * diff(1) * (edf_values[1][1][0] - edf_values[0][1][0])
            + diff(2) * (1 - diff(1)) * (edf_values[1][0][1] - edf_values[0][0][1])
            + diff(2) * diff(1) * (edf_values[1][1][1] - edf_values[0][1][1]))
            * edf_map_ptr_->inv_map_res;

    return (1 - diff(2)) * v0 + diff(2)* v1;

}

}
