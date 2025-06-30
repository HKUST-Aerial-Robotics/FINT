#include "marker_detector.hpp"


void marker_detector::read_param(std::string config_file){

    FILE *fh = fopen(config_file.c_str(),"r");
    if(fh == NULL){
        ROS_WARN("config_file dosen't exist; wrong config_file path");
        ROS_BREAK();
        return;          
    }
    fclose(fh);

    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    int pn = config_file.find_last_of('/');
    std::string configPath = config_file.substr(0, pn);

    cv::FileNode markers = fsSettings["markers"];

    marker_size_ = markers["marker_size"];
    required_sample_ = markers["required_sample"];
    output_min_sample_ = markers["output_min_sample"];
    pos_tol_ = markers["pos_tol"];
    inlayer_tol_ = markers["inlayer_tol"];

    cv::FileNode marker_ids = markers["ids"];
    for(cv::FileNodeIterator it = marker_ids.begin(); it != marker_ids.end(); it++){
        marker_id_.emplace_back((*it)["id"]);
    }


    cv::FileNode cam_module_node = fsSettings["cam_module"];
    int num_cam_module = cam_module_node["num"];
    printf("camera module number %d\n", num_cam_module);

    // cam_modules_.assign(num_cam_module, this);

    cv::FileNode cam_modules_node = cam_module_node["modules"];
    int cur_cam_module = 0;
    for(cv::FileNodeIterator it = cam_modules_node.begin(); it != cam_modules_node.end(); it++, cur_cam_module++){
        
        if(cur_cam_module >= num_cam_module) break;

        cam_modules_.emplace_back(this);
         
        (*it)["image0_topic"] >> cam_modules_[cur_cam_module].img0_topic_;
        (*it)["image1_topic"] >> cam_modules_[cur_cam_module].img1_topic_;
        (*it)["latest_cam_pose_topic"] >> cam_modules_[cur_cam_module].cam_pose_topic_;
        
        
        (*it)["cam_fx"] >> cam_modules_[cur_cam_module].K_.at<double>(0,0);
        (*it)["cam_fy"] >> cam_modules_[cur_cam_module].K_.at<double>(1,1);
        (*it)["cam_cx"] >> cam_modules_[cur_cam_module].K_.at<double>(0,2);
        (*it)["cam_cy"] >> cam_modules_[cur_cam_module].K_.at<double>(1,2);

        (*it)["cam_k1"] >> cam_modules_[cur_cam_module].D_.at<double>(0,0);
        (*it)["cam_k2"] >> cam_modules_[cur_cam_module].D_.at<double>(1,0);
        (*it)["cam_p1"] >> cam_modules_[cur_cam_module].D_.at<double>(2,0);
        (*it)["cam_p2"] >> cam_modules_[cur_cam_module].D_.at<double>(3,0);
        
    }    


    marker_pos_buffer_.resize(marker_id_.size());
    mean_marker_pos_.resize(3, marker_id_.size());

    register_sub();
}

void marker_detector::register_sub(){

    for(int i = 0; i < cam_modules_.size(); i++){
        cam_module& cam_module_temp = cam_modules_[i];

        cam_module_temp.sub_color_ptr_.reset(new message_filters::Subscriber<sensor_msgs::Image>(nh_, cam_module_temp.img0_topic_, 10, ros::TransportHints().tcpNoDelay(true)));
        cam_module_temp.sub_depth_ptr_.reset(new message_filters::Subscriber<sensor_msgs::Image>(nh_, cam_module_temp.img1_topic_, 10, ros::TransportHints().tcpNoDelay(true)));
        cam_module_temp.sub_cam_pose_ptr_.reset(new message_filters::Subscriber<geometry_msgs::PoseStamped>(nh_, cam_module_temp.cam_pose_topic_, 100, ros::TransportHints().tcpNoDelay(true)));        
        cam_module_temp.sync_color_depth_pose_ptr_.reset(new message_filters::Synchronizer<ColorDepthPoseSyncPolicy>(ColorDepthPoseSyncPolicy(100), *cam_module_temp.sub_color_ptr_, *cam_module_temp.sub_depth_ptr_, *cam_module_temp.sub_cam_pose_ptr_));

        cam_module_temp.sync_color_depth_pose_ptr_->registerCallback(boost::bind(&cam_module::color_depth_pose_callback, &cam_module_temp,_1, _2, _3));
    }

    detect_trigger_sub_ = nh_.subscribe<geometry_msgs::PoseStamped>("detect_trigger", 10, &marker_detector::detect_trigger_callback, this);

}

Vector3d marker_detector::plane_dir_fitting(MatrixXd &pts_mat)
{
    Vector3d mean_pt = pts_mat.rowwise().mean();
    MatrixXd pts_mat_around_mean = pts_mat - mean_pt * RowVectorXd::Ones(pts_mat.cols());
    Eigen::JacobiSVD<Eigen::MatrixXd> svd = pts_mat_around_mean.jacobiSvd(ComputeFullU | ComputeThinV);
    return svd.matrixU().col(2).normalized();
}

bool marker_detector::get_mean_pos(){

    Vector3d marker_pos_mean_temp; 
    for(int mk_idx = 0; mk_idx < marker_id_.size(); mk_idx++){
        for(bool remove_marker = true; remove_marker;){

            if(marker_pos_buffer_[mk_idx].size() < output_min_sample_) return false;

            marker_pos_mean_temp.setZero();
            double max_error = 0.0;

            for(auto& mk_pos : marker_pos_buffer_[mk_idx]){
                marker_pos_mean_temp += mk_pos;
            }
            marker_pos_mean_temp /= marker_pos_buffer_[mk_idx].size();

            for(auto& mk_pos : marker_pos_buffer_[mk_idx]){
                max_error = max(max_error, (mk_pos - marker_pos_mean_temp).norm());
            }

            max_error *= inlayer_tol_;

            max_error = max(max_error, pos_tol_);  

            // cout<<"marker "<<marker_id_[mk_idx]<<" max_error: "<< max_error<<endl;   

            remove_marker = false;

            for(auto mk_pos_it = marker_pos_buffer_[mk_idx].begin(); mk_pos_it!= marker_pos_buffer_[mk_idx].end();){

                if((*mk_pos_it - marker_pos_mean_temp).norm() > max_error){
                    // ROS_ERROR("remove %d!", distance(marker_pos_buffer[mk_idx].begin(), mk_pos_it));
                    // cout<<"remove large error marker, pos: "<<(*mk_pos_it).transpose()<<endl;
                    mk_pos_it = marker_pos_buffer_[mk_idx].erase(mk_pos_it);
                    remove_marker = true;
                }else{
                    mk_pos_it++;
                }
            }

        }

        mean_marker_pos_.col(mk_idx) = marker_pos_mean_temp;
        cout<<"use marker num: "<< marker_pos_buffer_[mk_idx].size()<<endl;
        // cout<<"marker mean pos:\n"<< marker_pos_mean_temp.transpose()<<endl;
    }

    return true;
}

void marker_detector::publish_mean_pose(const Vector3d& cam_dir ,const ros::Time& stamp){
    detecting_ = false;
    for(int mk_idx = 0; mk_idx < marker_id_.size(); mk_idx++){
        ROS_WARN("marker id: %d", marker_id_[mk_idx]);
        std::cout<<"marker mean pos:\n"<<mean_marker_pos_.col(mk_idx).transpose()<<endl;
    }

    Vector3d center_pos = mean_marker_pos_.rowwise().mean();
    Vector3d plane_normal = plane_dir_fitting(mean_marker_pos_);
    if(plane_normal.dot(cam_dir) < 0){
        plane_normal = -plane_normal;
    }

    geometry_msgs::PoseStamped pose;
    pose.header.frame_id = "world";
    pose.header.stamp = stamp;
    pose.pose.position.x = center_pos.x(); 
    pose.pose.position.y = center_pos.y(); 
    pose.pose.position.z = center_pos.z();

    Matrix3d pose_ori = Matrix3d::Identity();
    pose_ori.col(0) = plane_normal.normalized();
    pose_ori.col(1) = pose_ori.col(2).cross(plane_normal).normalized();
    pose_ori.col(2) = pose_ori.col(0).cross(pose_ori.col(1)).normalized();
    Quaterniond pose_q(pose_ori);
    pose_q.normalize();

    pose.pose.orientation.w = pose_q.w();
    pose.pose.orientation.x = pose_q.x();
    pose.pose.orientation.y = pose_q.y();
    pose.pose.orientation.z = pose_q.z();

    mean_pose_pub_.publish(pose);

    cout<<"center pos:\n"<<center_pos.transpose()<<endl;
    cout<<"plane normal:\n"<<plane_normal.transpose()<<endl;
}


void marker_detector::cam_module::color_depth_pose_callback(const sensor_msgs::ImageConstPtr &color_img_ptr, const sensor_msgs::ImageConstPtr &depth_img_ptr, const geometry_msgs::PoseStampedConstPtr &cam_pose_ptr)
{
    if(!detector_ptr_->detecting_) return;

    cv_bridge::CvImagePtr cv_ptr_color = cv_bridge::toCvCopy(color_img_ptr, color_img_ptr->encoding);
    cv_bridge::CvImagePtr cv_ptr_depth = cv_bridge::toCvCopy(depth_img_ptr, depth_img_ptr->encoding);

    if (depth_img_ptr->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
        (cv_ptr_depth->image).convertTo(cv_ptr_depth->image, CV_16UC1, 0.001);
    }

    Mat color_img, depth_img;
    cv_ptr_color->image.copyTo(color_img);
    cv_ptr_depth->image.copyTo(depth_img);

    Vector3d t_wc(cam_pose_ptr->pose.position.x, cam_pose_ptr->pose.position.y, cam_pose_ptr->pose.position.z);
    Matrix3d R_wc = Quaterniond(cam_pose_ptr->pose.orientation.w, cam_pose_ptr->pose.orientation.x, cam_pose_ptr->pose.orientation.y, cam_pose_ptr->pose.orientation.z).toRotationMatrix();

    // Mat frame_undistort;
    // undistort(color_img, frame_undistort, K, D);

    vector< int > markerIds; 
    vector< vector<Point2f> > markerCorners, rejectedCandidates; 

    cv::Ptr<cv::aruco::Dictionary> dictionary=cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_250);
    

    cv::aruco::detectMarkers(color_img, dictionary, markerCorners, markerIds);//, parameters, rejectedCandidates);
    if(markerIds.size() == 4){
        cv::aruco::drawDetectedMarkers(color_img, markerCorners);
    }
    // cout<<"num markers: "<<markerIds.size()<<endl;
    
    vector<Vec3d> rvecs, tvecs;

    if(markerIds.size() > 0){
        cv::aruco::estimatePoseSingleMarkers(markerCorners, detector_ptr_->marker_size_, K_, D_, rvecs, tvecs);

        for(unsigned int i = 0; i < markerIds.size(); i++){
            // cv::aruco::drawAxis(frame_undistort, K, D, rvecs[i], tvecs[i], 0.1); 

            Vector3d t_cm; 

            
            Point2f marker_center = 0.25 * (markerCorners[i][0] + markerCorners[i][1] + markerCorners[i][2] + markerCorners[i][3]);
            Vector2i marker_center_coord (static_cast<int>(round(marker_center.x)), static_cast<int>(round(marker_center.y)));

            double depth = depth_img.at<uint16_t>(marker_center_coord.y(), marker_center_coord.x()) * 0.001;

            t_cm.x() = (marker_center_coord.x() - K_.at<double>(0,2)) / K_.at<double>(0,0) * depth;
            t_cm.y() = (marker_center_coord.y() - K_.at<double>(1,2)) / K_.at<double>(1,1) * depth;
            t_cm.z() = depth;
           

            Vector3d marker_pos = R_wc * t_cm + t_wc;
            // Matrix3d temp_marker_ori = R_ic * R_eigen;

            for(int mk_idx = 0; mk_idx < detector_ptr_->marker_id_.size(); mk_idx++){
                if(markerIds[i] == detector_ptr_->marker_id_[mk_idx]){
                    detector_ptr_->marker_pos_buffer_[mk_idx].emplace_back(marker_pos);
                    break;
                }
            }

            // ROS_WARN("marker id: %d", markerIds[i]);
            // std::cout<<"marker_pos:\n"<<marker_pos.transpose()<<endl;

        }

        bool marker_enough = true;
        for(int mk_idx = 0; mk_idx < detector_ptr_->marker_id_.size(); mk_idx++){
            if(detector_ptr_->marker_pos_buffer_[mk_idx].size() < detector_ptr_->required_sample_){
                marker_enough = false;
                break;
            }
        }

        // ROS_ERROR("marker enough? %d", marker_enough);
        // for(int mk_idx = 0; mk_idx < detector_ptr_->marker_id_.size(); mk_idx++){
        //     cout<<"marker "<<detector_ptr_->marker_id_[mk_idx]<<" buffer size: "<<detector_ptr_->marker_pos_buffer_[mk_idx].size()<<endl;
        // }
	

        if(marker_enough && detector_ptr_->get_mean_pos()){
            detector_ptr_->publish_mean_pose(R_wc.col(2), cam_pose_ptr->header.stamp);
        }

        
    }

    
}

void marker_detector::detect_trigger_callback(const geometry_msgs::PoseStamped::ConstPtr &trigger)
{
    ROS_WARN("rcv trigger");
    for(auto& buffer_list: marker_pos_buffer_){
        buffer_list.clear();
    }
    detecting_ = true;
}
