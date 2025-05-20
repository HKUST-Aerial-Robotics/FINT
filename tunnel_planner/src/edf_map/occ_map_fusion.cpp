/*******************************************************
Copyright (C) 2025, Aerial Robotics Group, Hong Kong University of Science and Technology
This file is part of FINT.
Licensed under the GNU General Public License v3.0;
you may not use this file except in compliance with the License.
*******************************************************/

#include <edf_map/occ_map_fusion.h>

namespace tunnel_planner{

void occ_map_fusion::Init(ros::NodeHandle& n, shared_ptr<voxel_map<edf_voxel>> map_ptr){

    // std::string config_file;
    // n.getParam("config_file", config_file);
    // std::cout << "config file:\n" << config_file << '\n';

    // readParameters(config_file);

    // update_flag_ = -1;

    // map_ptr_ = make_shared<voxel_map<edf_voxel>>(MAP_RES, MAP_LIM);

    map_ptr_ = map_ptr;

    publish_pcd_ = false;

    cloud_.reserve(map_ptr_->map_data.size());

    set_modules();
    set_param();
    register_sub(n);
    register_pub(n);

    start_process_thread();

}

void occ_map_fusion::set_modules(){
    for(auto& cam_module : CAM_INFO_VEC){
        camera_modules_.emplace_back(make_unique<camera_module_info_with_sub>(cam_module, this));
    }

    for(unsigned int i = 0; i < camera_modules_.size(); i++){
        camera_modules_[i]->unique_id_ = i;
        
        cout<<"cam "<<i<<":\ncam_t_depth:\n"<<camera_modules_[i]->module_info_.cam_t_depth_.transpose()<<endl<<"cam_R_depth:\n"<<camera_modules_[i]->module_info_.cam_R_depth_<<endl;
    }
}

void occ_map_fusion::set_param(){
    project_param_.depth_margin = DEPTH_MARGIN;
    project_param_.num_pixel_skip = NUM_PIXEL_SKIP;
    project_param_.depth_scale = DEPTH_SCALE;
    project_param_.min_depth = MIN_DEPTH;
    project_param_.max_depth = MAX_DEPTH;
    project_param_.max_ray_length = MAX_RAY_LENGTH;


    double prob_hit_log      = logit(PROB_HIT);
    double prob_miss_log     = logit(PROB_MISS);
    double clamp_min_log     = logit(CLAMP_MIN);
    double clamp_max_log     = logit(CLAMP_MAX);
    double min_occupancy_log = logit(MIN_OCCUPANCY);

    const double prob_scale = 1e8; 

    fusion_param_.prob_hit_log      = static_cast<int>(round(prob_hit_log      * prob_scale));
    fusion_param_.prob_miss_log     = static_cast<int>(round(prob_miss_log     * prob_scale));
    fusion_param_.clamp_min_log     = static_cast<int>(round(clamp_min_log     * prob_scale));
    fusion_param_.clamp_max_log     = static_cast<int>(round(clamp_max_log     * prob_scale));
    fusion_param_.min_occupancy_log = static_cast<int>(round(min_occupancy_log * prob_scale));

    cout<<"prob_hit_log: "<<fusion_param_.prob_hit_log<<endl;
    cout<<"prob_miss_log: "<<fusion_param_.prob_miss_log<<endl;
    cout<<"clamp_min_log: "<<fusion_param_.clamp_min_log<<endl;
    cout<<"clamp_max_log: "<<fusion_param_.clamp_max_log<<endl;
    cout<<"min_occupancy_log: "<<fusion_param_.min_occupancy_log<<endl;

    for(auto& cam_module :camera_modules_){
        cam_module->raycaster_.setParams(MAP_RES, MAP_LIM.leftCols(1));
    }

    start_mapping_ = false;
}

void occ_map_fusion::register_sub(ros::NodeHandle &n){


    if(USE_EXACT_TIME_SYNC){

        for (unsigned int i = 0; i < camera_modules_.size(); i++){
        
            camera_modules_[i]->depth_pose_sub_ptr_.depth_sub_ptr_ = new message_filters::Subscriber<sensor_msgs::Image> (n, camera_modules_[i]->module_info_.depth_topic_, 10, ros::TransportHints().tcpNoDelay(true));
            camera_modules_[i]->depth_pose_sub_ptr_.cam_pose_sub_ptr_ = new message_filters::Subscriber<geometry_msgs::PoseStamped> (n, camera_modules_[i]->module_info_.cam_pose_topic_, 100, ros::TransportHints().tcpNoDelay(true));
            
            camera_modules_[i]->depth_pose_sub_ptr_.sync_image_pose_exact_.reset(new message_filters::Synchronizer<SyncPolicyImagePoseExact>(SyncPolicyImagePoseExact(100), *(camera_modules_[i]->depth_pose_sub_ptr_.depth_sub_ptr_), *(camera_modules_[i]->depth_pose_sub_ptr_.cam_pose_sub_ptr_)));
            camera_modules_[i]->depth_pose_sub_ptr_.sync_image_pose_exact_->registerCallback(boost::bind(&occ_map_fusion::camera_module_info_with_sub::depth_pose_callback, camera_modules_[i].get(), _1, _2));
            // camera_modules_[i].depth_pose_sub_ptr_.sync_image_pose_ptr_ = new message_filters::TimeSynchronizer<sensor_msgs::Image, geometry_msgs::PoseStamped> (*camera_modules_[i].depth_pose_sub_ptr_.depth_sub_ptr_, *camera_modules_[i].depth_pose_sub_ptr_.cam_pose_sub_ptr_, 1000);
            // camera_modules_[i].depth_pose_sub_ptr_.sync_image_pose_ptr_ ->registerCallback(boost::bind(&occ_map_fusion::camera_module_info_with_sub::depth_pose_callback, &(this->camera_modules_[i]), _1, _2));
            
            camera_modules_[i]->depth_pose_sub_ptr_.sync_image_pose_approximate_.release();
        }

    }else{

        for (unsigned int i = 0; i < camera_modules_.size(); i++){
        
            camera_modules_[i]->depth_pose_sub_ptr_.depth_sub_ptr_ = new message_filters::Subscriber<sensor_msgs::Image> (n, camera_modules_[i]->module_info_.depth_topic_, 10, ros::TransportHints().tcpNoDelay(true));
            camera_modules_[i]->depth_pose_sub_ptr_.cam_pose_sub_ptr_ = new message_filters::Subscriber<geometry_msgs::PoseStamped> (n, camera_modules_[i]->module_info_.cam_pose_topic_, 1000, ros::TransportHints().tcpNoDelay(true));
            
            camera_modules_[i]->depth_pose_sub_ptr_.sync_image_pose_approximate_.reset(new message_filters::Synchronizer<SyncPolicyImagePoseApproximate>(SyncPolicyImagePoseApproximate(100), *camera_modules_[i]->depth_pose_sub_ptr_.depth_sub_ptr_, *camera_modules_[i]->depth_pose_sub_ptr_.cam_pose_sub_ptr_));
            camera_modules_[i]->depth_pose_sub_ptr_.sync_image_pose_approximate_->registerCallback(boost::bind(&occ_map_fusion::camera_module_info_with_sub::depth_pose_callback, camera_modules_[i].get(), _1, _2));
            // camera_modules_[i].depth_pose_sub_ptr_.sync_image_pose_ptr_ = new message_filters::TimeSynchronizer<sensor_msgs::Image, geometry_msgs::PoseStamped> (*camera_modules_[i].depth_pose_sub_ptr_.depth_sub_ptr_, *camera_modules_[i].depth_pose_sub_ptr_.cam_pose_sub_ptr_, 1000);
            // camera_modules_[i].depth_pose_sub_ptr_.sync_image_pose_ptr_ ->registerCallback(boost::bind(&occ_map_fusion::camera_module_info_with_sub::depth_pose_callback, &(this->camera_modules_[i]), _1, _2));
            
            camera_modules_[i]->depth_pose_sub_ptr_.sync_image_pose_exact_.release();
        }

    }


    // update_timer_ = n.createTimer(ros::Duration(1.0 / UPDATE_FREQ), &occ_map_fusion::update_occ_map, this);
}


void occ_map_fusion::register_pub(ros::NodeHandle &n){
    map_pub_ = n.advertise<sensor_msgs::PointCloud>("occ_map", 10);
    map2_pub_ = n.advertise<sensor_msgs::PointCloud2>("occ_map2", 10);
    free_map_pub_ = n.advertise<sensor_msgs::PointCloud>("free_map", 10);
}

void occ_map_fusion::start_process_thread(){

    updade_thread_ptr_ = make_unique<thread>(&occ_map_fusion::process_loop, this);
    pub_thread_ptr_ = make_unique<thread>(&occ_map_fusion::pub_loop, this);

}

void occ_map_fusion::process_loop(){

    const std::chrono::duration<double> max_update_interval(1.0 / UPDATE_FREQ);

    std::chrono::time_point<std::chrono::system_clock> start_time;
    std::chrono::duration<double> process_time;

    while(true){
        start_time = std::chrono::system_clock::now();
        if(start_mapping_){
            update_occ_map();
        }
        process_time = std::chrono::system_clock::now() - start_time;

        // cout<<"process_time: "<<process_time.count()<<endl;

        if(process_time < max_update_interval){
            this_thread::sleep_for(max_update_interval - process_time);
        }

    }
}

void occ_map_fusion::pub_loop(){

    const std::chrono::duration<double> max_update_interval(1.0 / 10.0);

    std::chrono::time_point<std::chrono::system_clock> start_time;
    std::chrono::duration<double> pub_time;

    while(true){
        start_time = std::chrono::system_clock::now();

        if(start_mapping_ && publish_pcd_){
            publish_pcd_ = false;
            pub_occ_map();
        }

        pub_time = std::chrono::system_clock::now() - start_time;

        // cout<<"pub_time: "<<pub_time.count()<<endl;

        if(pub_time < max_update_interval){
            this_thread::sleep_for(max_update_interval - pub_time);
        }

    }
}

void occ_map_fusion::camera_module_info_with_sub::depth_pose_callback(const sensor_msgs::ImageConstPtr& img, const geometry_msgs::PoseStampedConstPtr& pose){
    Vector3d cam_t(pose->pose.position.x, pose->pose.position.y, pose->pose.position.z);

    if (!fusion_ptr_->map_ptr_->in_map(cam_t)) return;

    // this->m_update_.lock();

    this->cam_t_input_ = cam_t;
    this->cam_R_input_ = Quaterniond(pose->pose.orientation.w, pose->pose.orientation.x, pose->pose.orientation.y, pose->pose.orientation.z).toRotationMatrix();
    
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(img, img->encoding);

    if (img->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
        (cv_ptr->image).convertTo(cv_ptr->image, CV_16UC1, 1.0 / fusion_ptr_->project_param_.depth_scale);
    }
    // cv_ptr->image.copyTo(this->depth_img_);

    this->depth_img_input_ = cv_ptr->image;

    this->pose_time_input_ = pose->header.stamp;

    this->update_flag_ = true;

    fusion_ptr_->latest_cam_t_ = cam_t;

    // this->m_update_.unlock();

    // fusion_ptr_->m_update_.lock();

    // fusion_ptr_->cam_t_ = cam_t;
    // fusion_ptr_->cam_R_ = Quaterniond(pose->pose.orientation.w, pose->pose.orientation.x, pose->pose.orientation.y, pose->pose.orientation.z).toRotationMatrix();

    // /* get depth image */
    // cv_bridge::CvImagePtr cv_ptr;
    // cv_ptr = cv_bridge::toCvCopy(img, img->encoding);

    // if (img->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
    //     (cv_ptr->image).convertTo(cv_ptr->image, CV_16UC1, 1.0 / fusion_ptr_->project_param_.depth_scale);
    // }
    // cv_ptr->image.copyTo(fusion_ptr_->depth_img_);

    // fusion_ptr_->map_time_ = pose->header.stamp;

    // fusion_ptr_->update_flag_ = unique_id_;

    // fusion_ptr_->m_update_.unlock();
}


void occ_map_fusion::camera_module_info_with_sub::fectch_data_for_update(){
    pose_time_ = pose_time_input_;
    depth_img_ = depth_img_input_.clone();
    cam_t_ = cam_t_input_;
    cam_R_ = cam_R_input_;
    process_flag_ = update_flag_;
    update_flag_ = false;
}

void occ_map_fusion::update_occ_map(){

    // ROS_WARN("update occ");
    // m_update_.lock();

    ros::Time t0 = ros::Time::now();

    for(auto& cam_module_ptr : camera_modules_){
        cam_module_ptr->m_data_.lock();
    
        if(cam_module_ptr->update_flag_){
            // ROS_WARN("update cam %d", update_flag_);
            // update_occ_map_3d();
            cam_module_ptr->fectch_data_for_update();
            
        }

        cam_module_ptr->m_data_.unlock();
    }


    vector<thread> update_occ_map_thread_vec;

    for(auto& cam_module_ptr : camera_modules_){

        if(cam_module_ptr->process_flag_){

            update_occ_map_thread_vec.emplace_back(&occ_map_fusion::update_occ_map_3d, this, cam_module_ptr);
            // update_occ_map_3d(cam_module_ptr);
            map_time_ = max(cam_module_ptr->pose_time_, map_time_);

            cam_module_ptr->process_flag_ = false;
            publish_pcd_ = true;

        }
    }

    for(auto& update_map_thread : update_occ_map_thread_vec){
        update_map_thread.join();
    }

    // ros::Time t1 = ros::Time::now();
    // cout<<"update occ time: "<<(t1-t0).toSec()<<endl;


    // if(publish_pcd_){
    //     pub_occ_map();
    // }

    // ros::Time t1 = ros::Time::now();
    // cout<<"pub time: "<<(t1-t0).toSec()<<endl;
    // m_update_.unlock();

}

// void occ_map_fusion::reset_trigger_callback(const geometry_msgs::PoseStamped::ConstPtr &trigger)
// {
//     ROS_WARN("reset trigger");
//     for(auto& i : map_ptr_->map_data){
//         i.type = i.UNKNOWN;
//         i.occ_value = 0.0;
//     }
//     // map_ptr_.reset(new voxel_map<edf_voxel>(MAP_RES, MAP_LIM));
//     ROS_WARN("reset finish");

// }


// void occ_map_fusion::update_occ_map_3d(){
//     pcl::PointCloud<Vector3d> pcd_world;
//     pcd_world.clear();

//     Vector3d min_bound = cam_t_, max_bound = cam_t_;

//     ros::Time t0 = ros::Time::now();
//     project_depth(pcd_world, min_bound, max_bound);
//     ros::Time t1 = ros::Time::now();
//     // cout<<"project time: "<<(t1-t0).toSec()<<endl;

//     t0 = ros::Time::now();
//     raycast_process(pcd_world, min_bound, max_bound);
//     t1 = ros::Time::now();
//     // cout<<"cast time: "<<(t1-t0).toSec()<<endl;
// }

void occ_map_fusion::update_occ_map_3d(const shared_ptr<camera_module_info_with_sub>& cam_module){
    pcl::PointCloud<Vector3d> pcd_world;
    pcd_world.clear();

    
    // if(cam_module.process_flag_){
    
    Vector3d min_bound = cam_module->cam_t_, max_bound = cam_module->cam_t_;

    // ros::Time t0 = ros::Time::now();
    project_depth(pcd_world, min_bound, max_bound, *cam_module);
    // ros::Time t1 = ros::Time::now();
    // cout<<"project time: "<<(t1-t0).toSec()<<endl;

    // t0 = ros::Time::now();
    raycast_process(pcd_world, min_bound, max_bound, *cam_module);
    // t1 = ros::Time::now();

    // cout<<"update occ map: "<<min_bound.transpose()<<" to "<<max_bound.transpose()<<endl;

    // }

    // cout<<"cast time: "<<(t1-t0).toSec()<<endl;
}


void occ_map_fusion::project_depth(pcl::PointCloud<Vector3d>& pcd_world, Vector3d& min_bound, Vector3d& max_bound, const camera_module_info_with_sub& cam_module){

    // camera_module_info& cam_module = camera_modules_[update_flag_].module_info_;

    const cv::Mat& depth_img = cam_module.depth_img_;

    const double inv_fx = 1.0 / cam_module.module_info_.depth_fx_; 
    const double inv_fy = 1.0 / cam_module.module_info_.depth_fy_;
    const double& cx = cam_module.module_info_.depth_cx_;
    const double& cy= cam_module.module_info_.depth_cy_;

    const Matrix3d& cam_R_depth = cam_module.module_info_.cam_R_depth_;
    const Vector3d& cam_t_depth = cam_module.module_info_.cam_t_depth_;

    const Matrix3d& cam_R = cam_module.cam_R_;
    const Vector3d& cam_t = cam_module.cam_t_;

    int cols = depth_img.cols;
    int rows = depth_img.rows;
    double depth = 0.0;
    uint16_t raw_depth;

    Vector3d pt, pt_world;

    for (int v = project_param_.depth_margin; v < rows - project_param_.depth_margin; v += project_param_.num_pixel_skip) {

        for (int u = project_param_.depth_margin; u < cols - project_param_.depth_margin; u += project_param_.num_pixel_skip) {
            
            raw_depth = depth_img.at<uint16_t>(v,u);
            depth = raw_depth * project_param_.depth_scale;

            // cout<<"depth: "<<depth<<endl;

            if (isnan(depth)){
                // ROS_ERROR("nan depth");
                continue;
            }

            // if(raw_depth == 0 || depth > project_param_.max_depth){
            // if(depth > project_param_.max_depth || depth == 0.0){
            if(depth > project_param_.max_depth){
                depth = project_param_.max_depth + project_param_.depth_scale;
            }
            
            if (depth < project_param_.min_depth){
                continue;
            }


            // project to world frame
            pt(0) = (u - cx) * depth * inv_fx;
            pt(1) = (v - cy) * depth * inv_fy;
            pt(2) = depth;

            pt_world = cam_R * (cam_R_depth * pt + cam_t_depth) + cam_t;

            pcd_world.emplace_back(pt_world);

            min_bound.x() = min(min_bound.x(),pt_world.x());
            min_bound.y() = min(min_bound.y(),pt_world.y());
            min_bound.z() = min(min_bound.z(),pt_world.z());

            max_bound.x() = max(max_bound.x(),pt_world.x());
            max_bound.y() = max(max_bound.y(),pt_world.y());
            max_bound.z() = max(max_bound.z(),pt_world.z());

        }
    }

    min_bound.x() = max(min_bound.x(),map_ptr_->xmin);
    min_bound.y() = max(min_bound.y(),map_ptr_->ymin);
    min_bound.z() = max(min_bound.z(),map_ptr_->zmin);

    max_bound.x() = min(max_bound.x(),map_ptr_->xmax);
    max_bound.y() = min(max_bound.y(),map_ptr_->ymax);
    max_bound.z() = min(max_bound.z(),map_ptr_->zmax);
}


void occ_map_fusion::raycast_process(const pcl::PointCloud<Vector3d>& pcd_world, const Vector3d& min_bound, const Vector3d& max_bound, camera_module_info_with_sub& cam_module){

    if(pcd_world.points.size() == 0) return;

    const Vector3d& cam_t = cam_module.cam_t_;

    Vector3i min_coord = map_ptr_->pos2coord(min_bound);
    Vector3i max_coord = map_ptr_->pos2coord(max_bound);
    Vector3i local_size = max_coord - min_coord + Vector3i::Ones();
    unsigned int vox_idx = 0, local_vox_idx = 0;
    Vector3i vox_coord(0,0,0), local_vox_coord(0,0,0);

    vector<uint8_t> ray_cast_flag(local_size.prod(),0);
    //bit 0 ray end
    //bit 1 ray traverse

    Vector3d ray_pt;

    double ray_length = 0.0;
    bool occ2set = false;

    for(auto pt_world : pcd_world.points){
        occ2set = true;
        ray_length = (pt_world - cam_t).norm();

        if(ray_length > project_param_.max_ray_length){
            pt_world = (pt_world - cam_t) / ray_length * project_param_.max_ray_length + cam_t;
            occ2set = false;
        }
        if(!map_ptr_->in_map(pt_world)){
            pt_world = map_ptr_->clamp_point_at_boundary(pt_world,cam_t);
            occ2set = false;
        }

        vox_coord = map_ptr_->pos2coord(pt_world);
        local_vox_coord = vox_coord - min_coord;

        vox_idx = map_ptr_->coord2idx(vox_coord);

        local_vox_idx = local_vox_coord.dot(Vector3i(local_size.y()*local_size.z(), local_size.z(), 1));

        // ray end finish

        // ensure ray end in the same voxel only cast once
        if(ray_cast_flag[local_vox_idx] & uint8_t(1)){
            continue;
        }else{
            ray_cast_flag[local_vox_idx] |= uint8_t(1);
            update_voxel_occ(vox_idx, occ2set);
        }


        // raycast start
        cam_module.raycaster_.input(pt_world, cam_t);

        while (cam_module.raycaster_.nextId(vox_coord)){

            local_vox_coord = vox_coord - min_coord;
            vox_idx = map_ptr_->coord2idx(vox_coord);
            local_vox_idx = local_vox_coord.dot(Vector3i(local_size.y()*local_size.z(), local_size.z(), 1));

            if(local_vox_idx < 0 || local_vox_idx > ray_cast_flag.size()){
                break;
            }

            if(ray_cast_flag[local_vox_idx] == uint8_t(0)){
                update_voxel_occ(vox_idx, false);
            }
            
            ray_cast_flag[local_vox_idx] |= uint8_t(2);


        }

    }
    
}


void occ_map_fusion::update_voxel_occ(const unsigned int vox_idx, const bool hit){


    int log_odds_update = hit ? fusion_param_.prob_hit_log : fusion_param_.prob_miss_log;

    auto& vox_data = map_ptr_->map_data[vox_idx];

    if(vox_data.type == edf_voxel::UNKNOWN){
        vox_data.occ_value = fusion_param_.min_occupancy_log;
        // ROS_WARN("update unknown");
    }

    vox_data.occ_value = min(max(vox_data.occ_value + log_odds_update, fusion_param_.clamp_min_log), fusion_param_.clamp_max_log);

    vox_data.type = vox_data.occ_value > fusion_param_.min_occupancy_log ? edf_voxel::OCC : edf_voxel::FREE;
}



void occ_map_fusion::pub_occ_map(){

    pcl::PointXYZ pt;
    
    cloud_.clear();

    Vector3d pos;
    // Vector3d pos_cam_diff;
    // const double squared_max_length = project_param_.max_ray_length * project_param_.max_ray_length;

    // ros::Time t0 = ros::Time::now();

    // Vector3d min_pos = latest_cam_t_ - project_param_.max_ray_length * Vector3d::Ones();
    // min_pos.x() = max(map_ptr_->xmin, min_pos.x());
    // min_pos.y() = max(map_ptr_->ymin, min_pos.y());
    // min_pos.z() = max(map_ptr_->zmin, min_pos.z());
    // Vector3i min_coord = map_ptr_->pos2coord(min_pos);

    // Vector3d max_pos = latest_cam_t_ + project_param_.max_ray_length * Vector3d::Ones();
    // max_pos.x() = min(map_ptr_->xmax, max_pos.x());
    // max_pos.y() = min(map_ptr_->ymax, max_pos.y());
    // max_pos.z() = min(map_ptr_->zmax, max_pos.z());
    // Vector3i max_coord = map_ptr_->pos2coord(max_pos);

    // Vector3i local_coord_size = max_coord - min_coord + Vector3i::Ones();

    // Vector3i global_coord;

    // sensor_msgs::PointCloud pcd_free;
    // geometry_msgs::Point32 pt_32;

    // for(int local_x = 0; local_x < local_coord_size.x(); local_x++){
    //     for(int local_y = 0; local_y < local_coord_size.y(); local_y++){
    //         for(int local_z = 0; local_z < local_coord_size.z(); local_z++){
    //             global_coord.x() = local_x + min_coord.x();
    //             global_coord.y() = local_y + min_coord.y();
    //             global_coord.z() = local_z + min_coord.z();

    //             int voxel_type = map_ptr_->map_data[map_ptr_->coord2idx(global_coord)].type;

    //             // if(voxel_type == edf_voxel::OCC){
    //             //     pos = map_ptr_->coord2pos(global_coord);
    //             //     pt.x = pt_32.x = pos.x();
    //             //     pt.y = pt_32.y = pos.y();
    //             //     pt.z = pt_32.z = pos.z();
    //             //     cloud_.emplace_back(pt);
    //             //     pcd_.points.emplace_back(pt_32);
    //             // }
    //             // else 
    //             if(voxel_type == edf_voxel::FREE){
    //                 pos = map_ptr_->coord2pos(global_coord);
    //                 pt_32.x = pos.x();
    //                 pt_32.y = pos.y();
    //                 pt_32.z = pos.z();
    //                 pcd_free.points.emplace_back(pt_32);
    //             }
    //         }
    //     }
    // }


    for(auto map_data_ptr = map_ptr_->map_data.begin(); map_data_ptr != map_ptr_->map_data.end(); map_data_ptr++){
        int idx = map_data_ptr - map_ptr_->map_data.begin();
        if( map_data_ptr->type == edf_voxel::OCC){
            pos = map_ptr_->idx2pos(idx);
            pt.x = pos.x();
            pt.y = pos.y();
            pt.z = pos.z();
            cloud_.emplace_back(pt);
        }
    }


    last_pcd_size_ = cloud_.size();

    

    cloud_.width = cloud_.points.size();
    cloud_.height = 1;
    cloud_.is_dense = true;
    cloud_.header.frame_id = "world";
    sensor_msgs::PointCloud2 cloud_msg;

    pcl::toROSMsg(cloud_, cloud_msg);

    cloud_msg.header.stamp = map_time_;

    map2_pub_.publish(cloud_msg);

    // pcd_free.header = cloud_msg.header;
    // free_map_pub_.publish(pcd_free);
}

}