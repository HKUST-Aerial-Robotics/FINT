/*******************************************************
Copyright (C) 2025, Aerial Robotics Group, Hong Kong University of Science and Technology
This file is part of FINT.
Licensed under the GNU General Public License v3.0;
you may not use this file except in compliance with the License.
*******************************************************/
#include <tunnel_planner.h>

namespace tunnel_planner{

double fRand(double fMin, double fMax)
{
    srand (ros::Time::now().nsec);
    double f = (double)rand() / (RAND_MAX + 1.0);
    return fMin + f * (fMax - fMin);
}

tunnel_planner::tunnel_planner(ros::NodeHandle& n): nh_(n){

    latest_odom_sub_ = nh_.subscribe<nav_msgs::Odometry>("latest_odom", 1, &tunnel_planner::latest_odom_callback, this);
    // plan_trigger_sub_ = nh_.subscribe<geometry_msgs::PoseStamped>("plan_trigger", 10, &tunnel_planner::plan_trigger_callback, this);
    tunnel_entrance_sub_ = nh_.subscribe<geometry_msgs::PoseStamped>("tunnel_entrance", 10, &tunnel_planner::tunnel_entrance_callback, this);

    // map_free_sub_ = nh_.subscribe<sensor_msgs::PointCloud>("free_map", 10, &tunnel_planner::map_free_callback, this);

    if(USE_EXACT_TIME_SYNC){
        map_sub_.subscribe(nh_, "occ_map", 10);
        map_free_sub_.subscribe(nh_, "free_map", 10);
        odom_sub_.subscribe(nh_, "odom", 10);
        sync_map_odom_exact_.reset(new message_filters::Synchronizer<SyncPolicyMapOdomExact>(SyncPolicyMapOdomExact(100), map_sub_, map_free_sub_, odom_sub_));
        sync_map_odom_exact_->registerCallback(boost::bind(&tunnel_planner::map_odom_callback, this, _1, _2, _3));
    }else{
        map_sub_.subscribe(nh_, "occ_map", 10);
        map_free_sub_.subscribe(nh_, "free_map", 10);
        odom_sub_.subscribe(nh_, "odom", 100);
        sync_map_odom_approximate_.reset(new message_filters::Synchronizer<SyncPolicyMapOdomApproximate>(SyncPolicyMapOdomApproximate(100), map_sub_, map_free_sub_, odom_sub_));
        sync_map_odom_approximate_->registerCallback(boost::bind(&tunnel_planner::map_odom_callback, this, _1, _2, _3));
    }
    
    
    
    // replan_timer_ = nh_.createTimer(ros::Duration(1.0 / REPLAN_FREQ), &tunnel_planner::replan, this);
    map_reset_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("map_reset_trigger", 10);
    traj_full_pub_ = nh_.advertise<Bspline_with_retiming>("traj_full", 10);

    corridor_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("corridor", 10);
    corridor_center_pub_ = nh_.advertise<geometry_msgs::PoseArray>("corridor_center", 10);
    corridor_center_init_pub_ = nh_.advertise<geometry_msgs::PoseArray>("corridor_center_init", 10);
    corridor_center_path_pub_ = nh_.advertise<nav_msgs::Path>("corridor_center_path", 10);
    traj_path_init_vis_pub_ = nh_.advertise<nav_msgs::Path>("traj_path_init_vis", 10);
    traj_path_vis_pub_ = nh_.advertise<nav_msgs::Path>("traj_path_vis", 10);

    past_corridor_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("past_corridor", 10);
    past_corridor_center_pub_ = nh_.advertise<geometry_msgs::PoseArray>("past_corridor_center", 10);
    past_corridor_center_path_pub_ = nh_.advertise<nav_msgs::Path>("past_corridor_center_path", 10);

    cross_section_img_pub_ = nh_.advertise<sensor_msgs::Image>("cross_section_img", 10);

    traj_state_pub_ = nh_.advertise<std_msgs::Int16>("traj_state", 10);

    drone_dim_ = DRONE_DIM;
    drone_radius_ = 0.5 * DRONE_DIM;
    drone_radius_pixel_ = static_cast<int>(ceil(0.5 * drone_dim_ / MAP_RES));
    
    tunnel_dim_ = TUNNEL_DIM;
    tunnel_dim_pixel_ = static_cast<int>(ceil(tunnel_dim_ / MAP_RES));
    half_tunnel_dim_pixel_ = static_cast<int>(ceil(0.5 * tunnel_dim_ / MAP_RES));

    hough_circle_detector_.reset(new HoughCircle(0.2, drone_radius_pixel_, half_tunnel_dim_pixel_+2, MAP_RES, 0.04));
    hough_rectangle_detector_.reset(new HoughRectangle(30, 50, -90, 90, drone_radius_pixel_, half_tunnel_dim_pixel_+2, 1, MAP_RES, 0.04));

    hough_circle_threshold_ = HOUGH_CIRCLE_THRESHOLD;
    hough_rectangle_threshold_ = HOUGH_RECTANGLE_THRESHOLD;

    shape_classifier_net_ = make_shared<nn>(SHAPE_CLASSIFIER_NET);
    shape_classifier_input_dim_ = shape_classifier_net_->get_input_dim();

    tunnel_entrance_pos_ = TUNNEL_ENTRANCE_POS + 0.5 * TUNNEL_ENTRANCE_DIR;
    tunnel_entrance_dir_ = TUNNEL_ENTRANCE_DIR;
    
    tunnel_step_res_ = TUNNEL_STEP_RES;
    cross_section_step_res_ = CROSS_SECTION_STEP_RES;
    tunnel_way_pt_min_interval_ = TUNNEL_WAY_PT_MIN_INTERVAL;
    grad_max_res_ = GRAD_MAX_RES;
    plan_range_ = PLAN_RANGE;

    adaptive_speed_ = ADAPTIVE_SPEED;
    if(!adaptive_speed_){
        flight_speed_ = FLIGHT_SPEED;
    }
    virtual_flight_progress_speed_ = VIRTUAL_FLIGHT_PROGRESS_SPEED;

    time_commit_ = ros::Duration(TIME_COMMIT);
    
    max_speed_ = MAX_SPEED;
    max_acc_ = MAX_ACC;
    max_yaw_dir_curvature_ratio_ = MAX_YAW_DIR_CURVATURE_RATIO;
    yaw_ahead_length_ = YAW_AHEAD_LENGTH;

    max_yaw_change_over_distance_ = fabs(MAX_YAW_CHANGE_OVER_DISTANCE);
    max_yaw_center_line_dir_diff_ = fabs(MAX_YAW_CENTER_LINE_DIR_DIFF);

    cmd_offset_z_ = 0.0;

    tunnel_exit_pos_ = Vector3d(numeric_limits<double>::max(), numeric_limits<double>::max(), numeric_limits<double>::max());

    edf_map_generator_ptr_.reset(new edf_map_generator(nh_, MAP_RES, MAP_LIM));

    plan_corridor_.reset(new vector<cross_section>);
    past_corridor_.reset(new vector<cross_section>);

    vert_sections_.reset(new vector<vert_section>);

    tunnel_center_line_.reset(new NonUniformBspline);
    tunnel_center_vel_.reset(new NonUniformBspline);
    tunnel_center_acc_.reset(new NonUniformBspline);
    tunnel_center_yaw_.reset(new NonUniformBspline);
    tunnel_center_yaw_dot_.reset(new NonUniformBspline);

    double w_disturbance = W_DISTURBANCE;
    if(!CIRCLE_LINEAR_LAYERS.empty() && !RECT_LINEAR_LAYERS.empty()){
        circle_net_.reset(new regression_net(CIRCLE_LINEAR_LAYERS));
        rect_net_.reset(new regression_net(RECT_LINEAR_LAYERS));
    }
    else{
        w_disturbance = 0.0;
    }

    double w_vision = W_VISION;
    optical_flow_estimator_.reset(new optical_flow_estimator(nh_, CAM_INFO_VEC, edf_map_generator_ptr_, plan_corridor_, past_corridor_, MAX_RAYCAST_LENGTH, OPTICAL_FLOW_CAL_RES));

    bspline_optimizer_.reset(new BsplineOptimizer);
    bspline_optimizer_1d_.reset(new optimizer_1d(plan_corridor_, vert_sections_, tunnel_center_line_, tunnel_center_vel_, tunnel_center_acc_, tunnel_center_yaw_, tunnel_center_yaw_dot_, max_speed_, max_acc_, 3, 0.3, w_disturbance, W_VISION, W_HEURISTIC, W_TIME, 100000, 0.07, circle_net_, rect_net_, optical_flow_estimator_));

    bspline_optimizer_->setParam(nh_);
    bspline_optimizer_->setEnvironment(edf_map_generator_ptr_);
    bspline_optimizer_->setDist0(DISTANCE_COST_ORIGIN);
    bspline_optimizer_1d_->set_param(nh_);

    if(!adaptive_speed_){
        bspline_optimizer_1d_->set_des_speed(flight_speed_);
    }

    traj_id_ = 1;
    last_traj_data_.traj_id_ = 1;

    last_plan_time_ = ros::Time(0.0);

    traj_state_ = HOVER;

    in_vertical_section_ = false;

    plan_fail_ = false;

    detect_tunnel_cnt_ = 0;

    use_bspline_ = true;

    traj_end_time_ = 0.0;


    last_traj_data_.traj_valid_ = false;

    start_replan_thread();

}

void tunnel_planner::start_replan_thread(){
    replan_thread_ptr_ = make_unique<thread>(&tunnel_planner::replan_loop, this);
}

void tunnel_planner::replan_loop(){

    const std::chrono::duration<double> max_update_interval(1.0 / REPLAN_FREQ);

    std::chrono::time_point<std::chrono::system_clock> start_time;
    std::chrono::duration<double> process_time;

    while(true){
        start_time = std::chrono::system_clock::now();
        replan();
        process_time = std::chrono::system_clock::now() - start_time;

        // cout<<"process_time: "<<process_time.count()<<endl;

        if(process_time < max_update_interval){
            this_thread::sleep_for(max_update_interval - process_time);
        }

    }
}

void tunnel_planner::latest_odom_callback(const nav_msgs::OdometryConstPtr& odom){
    latest_odom_ = *odom;
    edf_map_generator_ptr_->set_latest_odom(odom);
}


void tunnel_planner::tunnel_entrance_callback(const geometry_msgs::PoseStamped::ConstPtr &entrance_pose)
{
    if(traj_state_ != HOVER) return;
    
    ROS_WARN("rcv entrance");
    
    tunnel_entrance_pos_.x() = entrance_pose->pose.position.x;
    tunnel_entrance_pos_.y() = entrance_pose->pose.position.y;
    tunnel_entrance_pos_.z() = entrance_pose->pose.position.z;

    Quaterniond entrance_q(entrance_pose->pose.orientation.w, entrance_pose->pose.orientation.x, entrance_pose->pose.orientation.y, entrance_pose->pose.orientation.z);
    tunnel_entrance_dir_ = entrance_q.toRotationMatrix().col(0).normalized();

    traj_state_ = BEFORE_TUNNEL;

    cout<<"entrance pos: "<<tunnel_entrance_pos_.transpose()<<endl;
    cout<<"entrance dir: "<<tunnel_entrance_dir_.transpose()<<endl;

}


bool tunnel_planner::get_position_cmd_t_new(ros::Time t, Vector3d& pos, Vector3d& vel, Vector3d& acc, double& yaw, double& yaw_dot, double& curve_length, Vector3d& curvature){

    if(last_traj_data_.start_time_ > t) {
        ROS_WARN("traj time ahead!");
        return false;
    }
    double dT = (t - last_traj_data_.start_time_).toSec();
    // ROS_ERROR("last traj start time: %lf, dt: %lf", last_traj_data_.start_time_.toSec(), dT);

    if(dT <= last_traj_data_.position_traj_1d_.getTimeSum()){

        curve_length = last_traj_data_.position_traj_1d_.evaluateDeBoorT(dT)(0);

        double tangent_speed = last_traj_data_.velocity_traj_1d_.evaluateDeBoorT(dT)(0);
        double tangent_acceleration = last_traj_data_.acceleration_traj_1d_.evaluateDeBoorT(dT)(0);

        double process = last_traj_data_.position_traj_.getTimeFromLength(curve_length);

        pos = last_traj_data_.position_traj_.evaluateDeBoorT(process);

        yaw = last_traj_data_.yaw_traj_.evaluateDeBoorT(process)(0);
        while(yaw > M_PI){
            yaw -= (M_PI * 2.0);
        }
        while(yaw < -M_PI){
            yaw += (M_PI * 2.0);
        }
        yaw_dot = last_traj_data_.yaw_dot_traj_.evaluateDeBoorT(process)(0) * tangent_speed;

        Vector3d center_line_vel = last_traj_data_.velocity_traj_.evaluateDeBoorT(process);
        Vector3d center_line_dir = center_line_vel.normalized();
        vel = center_line_dir * tangent_speed;
        
        Vector3d center_line_acc = last_traj_data_.acceleration_traj_.evaluateDeBoorT(process);
        double squared_center_line_v = center_line_vel.squaredNorm();

        Vector3d center_line_central_acc = center_line_acc - center_line_acc.dot(center_line_dir) * center_line_dir;
        double center_line_central_a = center_line_central_acc.norm();


        curvature = center_line_central_acc / squared_center_line_v;

        double curvature_norm = center_line_central_a / squared_center_line_v;

        Vector3d centripetal_acc = Vector3d::Zero();

        if(center_line_central_a > 1e-4){
            centripetal_acc = center_line_central_acc / center_line_central_a * tangent_speed * tangent_speed * curvature_norm;
        }

        acc = centripetal_acc + tangent_acceleration * center_line_dir;

        return true;
    }
    else{
        vel.setZero();
        acc.setZero();
    }

    ROS_WARN("Use cur pose!");
    return false;
}

void tunnel_planner::replan(){

    ros::Time t0 = ros::Time::now();

    std_msgs::Int16 traj_state2pub;
    traj_state2pub.data = traj_state_;
    traj_state_pub_.publish(traj_state2pub);

    Vector3d start_pos(latest_odom_.pose.pose.position.x, latest_odom_.pose.pose.position.y, latest_odom_.pose.pose.position.z);
    Vector3d start_vel(0.0, 0.0, 0.0);
    Vector3d start_acc(0.0, 0.0, 0.0);

    double start_curve_length = -1.0;
    Vector3d start_curvature = Vector3d::Zero();

    double yaw = atan2(2 * (latest_odom_.pose.pose.orientation.w * latest_odom_.pose.pose.orientation.z + latest_odom_.pose.pose.orientation.x * latest_odom_.pose.pose.orientation.y), 1 - 2 * (latest_odom_.pose.pose.orientation.y * latest_odom_.pose.pose.orientation.y + latest_odom_.pose.pose.orientation.z * latest_odom_.pose.pose.orientation.z));
    double yaw_dot = 0.0; 


    plan_odom_time_ = latest_odom_.header.stamp + time_commit_;


    plan_corridor_->clear();

    switch (traj_state_){
        case HOVER:
            break;
        case BEFORE_TUNNEL:
            before_tunnel_plan_new(start_pos);
            break;
        case TUNNEL_ENTRANCE:

            if(last_traj_data_.traj_valid_){
                get_position_cmd_t_new(plan_odom_time_, start_pos, start_vel, start_acc, yaw, yaw_dot, start_curve_length, start_curvature);
            }
            else{
                start_pos = tunnel_entrance_pos_ - 1.5 * tunnel_entrance_dir_;
            }

            tunnel_entrance_plan_new(tunnel_step_res_, plan_range_, start_pos, start_vel, start_acc, yaw, yaw_dot, start_curve_length, start_curvature);
            break;
        case IN_TUNNEL:
            
            if(last_traj_data_.traj_valid_){
                get_position_cmd_t_new(plan_odom_time_, start_pos, start_vel, start_acc, yaw, yaw_dot, start_curve_length, start_curvature);
            }

            in_tunnel_plan(start_pos, start_vel, start_acc, yaw, yaw_dot, start_curve_length, start_curvature);
            break;
        case AFTER_TUNNEL:
            ROS_INFO("after tunnel");
            edf_map_generator_ptr_->stop_edf();
            traj_state_ = HOVER;
            last_traj_data_.traj_valid_ = false;
            break;
        
        default:
            break;
    }
}


void tunnel_planner::before_tunnel_plan_new(const Vector3d& start_pos){

    if(!last_traj_data_.traj_valid_){
        Vector3d start_vel_rect(0.0, 0.0, 0.0);
        Vector3d start_acc_rect(0.0, 0.0, 0.0);

        Vector3d end_pos = tunnel_entrance_pos_ - 1.5 * tunnel_entrance_dir_;
        Vector3d end_vel(0.0, 0.0, 0.0);
        Vector3d end_acc(0.0, 0.0, 0.0);

        double time_before = 4.0 * (end_pos - start_pos).norm();
        time_before = max(time_before, 2.0);

        double ts = time_before;

        vector<Vector3d> start;
        start.emplace_back(start_pos);
        start.emplace_back(start_vel_rect);
        start.emplace_back(start_acc_rect);


        vector<Vector3d> end;
        end.emplace_back(end_pos);
        end.emplace_back(end_vel);
        end.emplace_back(end_acc);

        vector<Vector3d> start_end_derivative(4, Vector3d::Zero());

        vector<Vector3d> way_pts;
        way_pts.emplace_back(start_pos);
        way_pts.emplace_back(end_pos);

        MatrixX3d ctrl_pts;

        NonUniformBspline::parameterizeToBspline(ts, way_pts, start_end_derivative, 5, ctrl_pts);

        NonUniformBspline traj_before(ctrl_pts, 5, ts);

        double traj_duration = traj_before.getTimeSum();
        ROS_INFO("traj_duration: %lf", traj_duration);
        ROS_INFO_STREAM("end_pos: " <<end_pos.transpose());


        last_traj_data_.traj_valid_ = true;

        last_traj_data_.start_time_ = plan_odom_time_;
        last_traj_data_.duration_ = traj_before.getTimeSum();
        last_traj_data_.traj_id_++;

        Bspline_with_retiming traj2pub;
        traj2pub.have_retiming = false;
        traj2pub.start_time = plan_odom_time_;
        traj2pub.traj_id = last_traj_data_.traj_id_;
        traj2pub.yaw_strategy = yaw_stragety::CONSTANT_PT;

        MatrixXd ctrl_pt = traj_before.getControlPoint();
        VectorXd knot_pt = traj_before.getKnot();

        traj2pub.order_3d = 5;

        for(int ctrl_pt_idx = 0; ctrl_pt_idx < ctrl_pt.rows(); ctrl_pt_idx++){
            geometry_msgs::Point pos_pt;
            pos_pt.x = ctrl_pt(ctrl_pt_idx, 0);
            pos_pt.y = ctrl_pt(ctrl_pt_idx, 1);
            pos_pt.z = ctrl_pt(ctrl_pt_idx, 2);
            traj2pub.pos_pts_3d.emplace_back(pos_pt);
        }
        for(int knot_pt_idx = 0; knot_pt_idx < knot_pt.rows(); knot_pt_idx++){
            traj2pub.knots_3d.emplace_back(knot_pt(knot_pt_idx));
        }

        traj2pub.yaw_dir_pt.x = tunnel_entrance_pos_.x();
        traj2pub.yaw_dir_pt.y = tunnel_entrance_pos_.y();
        traj2pub.yaw_dir_pt.z = tunnel_entrance_pos_.z();

        traj_full_pub_.publish(traj2pub);

        pub_traj_vis(traj_before);

        edf_map_generator_ptr_->wait_for_edf_available(1e-4);
        edf_map_generator_ptr_->reset_edf_map();
        edf_map_generator_ptr_->release_edf_resource();


    }
    else{
        if((latest_odom_.header.stamp - last_traj_data_.start_time_).toSec() >= last_traj_data_.duration_){

            traj_state_ = TUNNEL_ENTRANCE;
            last_traj_data_.start_travel_distance_ = 0.0;
            last_traj_data_.traj_valid_ = false;

            ROS_INFO("Switch to entrance state at %lf", latest_odom_.header.stamp.toSec());

            edf_map_generator_ptr_->start_edf();
            this_thread::sleep_for(std::chrono::duration<double>(1.0));
        }
    }
}



void tunnel_planner::tunnel_entrance_plan_new(const double plan_step, const double range, const Vector3d& start_pos, const Vector3d& start_vel, const Vector3d& start_acc, const double& start_yaw, const double& start_yaw_dot, const double& start_curve_length, const Vector3d& start_curvature){

    Vector3d tunnel_entrance_pos_confident = tunnel_entrance_pos_ + 2.0 * cross_section_step_res_ * tunnel_entrance_dir_;
    double switch_dist_2 = pow(min(cross_section_step_res_, 0.2), 2);
    
    if((start_pos - tunnel_entrance_pos_confident).squaredNorm() < switch_dist_2){      
        traj_state_ = IN_TUNNEL;
        ROS_INFO("Switch to in-tunnel state.");
    
        in_tunnel_plan(start_pos, start_vel, start_acc, start_yaw, start_yaw_dot, start_curve_length, start_curvature);
    }
    else{

        cross_section cross_section_tunnel_entrance;
        cross_section_tunnel_entrance.cross_section_shape_ = tunnel_shape::BEFORE;
        cross_section_tunnel_entrance.is_predict = false;
        cross_section_tunnel_entrance.cross_section_data_.assign(1, 0.05);
        cv::Mat local_cross_section;

        Vector3d pos = start_pos;
        double entrance_curve_length = start_curve_length;
        for(double entrance_length = 0.0; (pos - tunnel_entrance_pos_confident).squaredNorm() >= switch_dist_2; pos += cross_section_step_res_ * tunnel_entrance_dir_, entrance_length += cross_section_step_res_, entrance_curve_length += cross_section_step_res_){

            cross_section_tunnel_entrance.center_ = pos;
            cross_section_tunnel_entrance.w_R_cs = cal_w_R_plane(tunnel_entrance_dir_);

            plan_corridor_->emplace_back(cross_section_tunnel_entrance);

            if(entrance_length > 5.0){
                ROS_ERROR("entrance fail");
                traj_state_ = HOVER;
                last_traj_data_.traj_valid_ = false;
                plan_corridor_->clear();
                return;
            }

        }
        tunnel_plan_new(plan_step, max(0.0, range - cross_section_step_res_ * plan_corridor_->size()), pos, start_vel, start_acc, start_yaw, start_yaw_dot, start_curve_length, start_curvature);

    }
}

void tunnel_planner::in_tunnel_plan(const Vector3d& start_pos, const Vector3d& start_vel, const Vector3d& start_acc, const double& start_yaw, const double& start_yaw_dot, const double& start_curve_length, const Vector3d& start_curvature)
{

    int plan_result = tunnel_plan_new(tunnel_step_res_, plan_range_, start_pos, start_vel, start_acc, start_yaw, start_yaw_dot, start_curve_length, start_curvature);
        
    if(plan_result == ACCEPT || plan_result == NO_EXTENSION){
        return;
    }
    else if(plan_result == REACH_EXIT){
        traj_state_ = AFTER_TUNNEL;
        return;
    }
    
    
    for (int fail_cnt = 0; fail_cnt < 3; fail_cnt++){
        sleep(0.01);
        plan_fail_ = false;
        ros::spinOnce();
        ros::spinOnce();
        ros::spinOnce();
        ros::spinOnce();

        Vector3d start_pos_fail(latest_odom_.pose.pose.position.x, latest_odom_.pose.pose.position.y, latest_odom_.pose.pose.position.z);
        Vector3d start_vel_fail(latest_odom_.twist.twist.linear.x, latest_odom_.twist.twist.linear.y, latest_odom_.twist.twist.linear.z);
        Vector3d start_acc_fail(0.0, 0.0, 0.0);

        double start_curve_length_fail = -1.0;
        Vector3d start_curvature_fail = Vector3d::Zero();

        double start_yaw_fail = atan2(2 * (latest_odom_.pose.pose.orientation.w * latest_odom_.pose.pose.orientation.z + latest_odom_.pose.pose.orientation.x * latest_odom_.pose.pose.orientation.y), 1 - 2 * (latest_odom_.pose.pose.orientation.y * latest_odom_.pose.pose.orientation.y + latest_odom_.pose.pose.orientation.z * latest_odom_.pose.pose.orientation.z));
        double start_yaw_dot_fail = 0.0;


        plan_odom_time_ = latest_odom_.header.stamp + time_commit_;

        if(last_traj_data_.traj_valid_){
            get_position_cmd_t_new(plan_odom_time_, start_pos_fail, start_vel_fail, start_acc_fail, start_yaw_fail, start_yaw_dot_fail, start_curve_length_fail, start_curvature_fail);
        }

        plan_result = tunnel_plan_new(tunnel_step_res_, plan_range_, start_pos_fail, start_vel_fail, start_acc_fail, start_yaw_fail, start_yaw_dot_fail, start_curve_length_fail, start_curvature_fail);

        if(plan_result == ACCEPT || plan_result == NO_EXTENSION){
            return;
        }
        else if(plan_result == REACH_EXIT){
            traj_state_ = AFTER_TUNNEL;
            return;
        }
        
    }

    plan_fail_ = false;

}



void tunnel_planner::map_odom_callback(const sensor_msgs::PointCloudConstPtr& pcd, const sensor_msgs::PointCloudConstPtr& pcd_free, const nav_msgs::OdometryConstPtr& odom){

    ros::Time t_start = ros::Time::now();

    Vector3d body_t(odom->pose.pose.position.x, odom->pose.pose.position.y, odom->pose.pose.position.z);

    Vector3i edf_min_coord;
    Vector3i edf_max_coord;

    edf_map_generator_ptr_->cal_update_range(edf_min_coord, edf_max_coord);

    edf_map_generator_ptr_->reset_edf_map(edf_min_coord, edf_max_coord);

    for(auto pt : pcd->points){
        edf_map_generator_ptr_->get_edf_map_ptr()->map_data[edf_map_generator_ptr_->get_edf_map_ptr()->pos2idx(Vector3d(pt.x, pt.y, pt.z))].type = edf_voxel::OCC;
    }
    for(auto pt : pcd_free->points){
        edf_map_generator_ptr_->get_edf_map_ptr()->map_data[edf_map_generator_ptr_->get_edf_map_ptr()->pos2idx(Vector3d(pt.x, pt.y, pt.z))].type = edf_voxel::FREE; 
    }

    Vector3i drone_min_coord = edf_map_generator_ptr_->get_edf_map_ptr()->pos2coord(body_t - drone_radius_ * Vector3d::Ones());
    Vector3i drone_max_coord = edf_map_generator_ptr_->get_edf_map_ptr()->pos2coord(body_t + drone_radius_ * Vector3d::Ones());
    drone_min_coord.x() = max(0, drone_min_coord.x());
    drone_min_coord.y() = max(0, drone_min_coord.y());
    drone_min_coord.z() = max(0, drone_min_coord.z());
    drone_max_coord.x() = min(edf_map_generator_ptr_->get_edf_map_ptr()->map_size.x()-1, drone_max_coord.x());
    drone_max_coord.y() = min(edf_map_generator_ptr_->get_edf_map_ptr()->map_size.y()-1, drone_max_coord.y());
    drone_max_coord.z() = min(edf_map_generator_ptr_->get_edf_map_ptr()->map_size.z()-1, drone_max_coord.z());
    edf_map_generator_ptr_->set_edf_map_free(drone_min_coord, drone_max_coord);

    edf_map_generator_ptr_->update_edf_map(edf_min_coord, edf_max_coord);

    edf_map_generator_ptr_->publish_edf(edf_min_coord, edf_max_coord, body_t);

}

void tunnel_planner::map_free_callback(const sensor_msgs::PointCloudConstPtr& pcd_free){
    for(auto pt : pcd_free->points){
        auto& voxel_type = edf_map_generator_ptr_->get_edf_map_ptr()->map_data[edf_map_generator_ptr_->get_edf_map_ptr()->pos2idx(Vector3d(pt.x, pt.y, pt.z))].type; 
        if(voxel_type == edf_voxel::UNKNOWN)
            voxel_type = edf_voxel::FREE;
    }
}


int tunnel_planner::tunnel_plan_new(const double plan_step, const double range, const Vector3d& start_pos, const Vector3d& start_vel, const Vector3d& start_acc, const double& start_yaw, const double& start_yaw_dot, const double& start_curve_length, const Vector3d& start_curvature){

    double plan_dist = range;
    cross_section start_cross_section;
    start_cross_section.center_ = start_pos;

    if(!plan_corridor_->empty()){
        start_cross_section.w_R_cs = plan_corridor_->back().w_R_cs;
        start_cross_section.cross_section_data_.assign(1, 0.2);
        start_cross_section.cross_section_shape_ = UNKNOWN;
    }
    else if(start_vel.squaredNorm() < 1e-2 && !past_corridor_->empty()){
        start_cross_section.w_R_cs = past_corridor_->back().w_R_cs;
        start_cross_section.cross_section_data_.assign(1, 0.2);
        start_cross_section.cross_section_shape_ = UNKNOWN;
    }
    else{
        start_cross_section.w_R_cs = cal_w_R_plane(start_vel.normalized());       
        start_cross_section.cross_section_data_.assign(1, 0.2);
        start_cross_section.cross_section_shape_ = UNKNOWN;

        if(last_traj_data_.traj_valid_ && !last_traj_data_.corridor_.empty()){
            if(start_curve_length >= 0.0){

                for(unsigned int start_cs_idx = last_traj_data_.corridor_.size() - 2; start_cs_idx >= 0; start_cs_idx--){
                    if(start_curve_length >= last_traj_data_.corridor_[start_cs_idx].curve_length_){
                        if(!last_traj_data_.corridor_[start_cs_idx+1].is_predict && last_traj_data_.corridor_[start_cs_idx+1].cross_section_shape_ != tunnel_shape::BEFORE){

                            if(last_traj_data_.corridor_[start_cs_idx+1].cross_section_shape_ == last_traj_data_.corridor_[start_cs_idx].cross_section_shape_){
                                start_cross_section.cross_section_shape_ = last_traj_data_.corridor_[start_cs_idx+1].cross_section_shape_;
                                start_cross_section.cross_section_data_ = last_traj_data_.corridor_[start_cs_idx+1].cross_section_data_;
                            }

                        }
                        break;
                    }
                }
            }
        }

    }


    int corridor_search_status = search_corridor_path_hough(start_cross_section, plan_step, plan_dist, 0.52 * tunnel_dim_, 0.75);

    if(plan_corridor_->size() <= 2){
        plan_corridor_->clear();
        ROS_ERROR("no corridor");
        return plan_status::SHORT_TRAJECTORY;
    }

    pub_init_corridor();

    double total_dist = 0.0;

    for(auto it = plan_corridor_->begin(), next_it = next(it); next_it != plan_corridor_->end(); it = next_it, next_it = next(it)){
        total_dist += (next_it->center_ - it->center_).norm();

    }

    double ts = total_dist / (plan_corridor_->size() - 1) / virtual_flight_progress_speed_;

    *tunnel_center_line_ = extract_tunnel_center_line(ts, start_curvature);   
    pub_traj_init_vis(*tunnel_center_line_);
    bool reach_end = detect_corridor(ts);


    *tunnel_center_line_ = extract_tunnel_center_line(ts, start_curvature);
    *tunnel_center_vel_ = tunnel_center_line_->getDerivative();
    *tunnel_center_acc_ = tunnel_center_vel_->getDerivative();

    pub_traj_vis(*tunnel_center_line_);
    pub_corridor();


    *tunnel_center_yaw_ = plan_tunnel_center_yaw(ts, start_yaw, start_yaw_dot, !reach_end);
    *tunnel_center_yaw_dot_ = tunnel_center_yaw_->getDerivative();

    double predict_start_length = 0.0;


    if(!get_inter_cross_section_curve_length_vertical_cross_section_range(*tunnel_center_line_, predict_start_length)){
        return plan_status::UNKNOWN_FAILURE;
    }


    if(past_corridor_->empty()){
        past_corridor_->emplace_back(plan_corridor_->front());
    }
    else{
        if((start_pos - past_corridor_->back().center_).squaredNorm() >= plan_step * plan_step){
            past_corridor_->emplace_back(plan_corridor_->front());
        }
    }
    // pub_corridor();
    // pub_past_corridor();


    if(last_traj_data_.traj_valid_ && (start_curve_length + predict_start_length < last_traj_data_.predict_start_length_ || start_curve_length + plan_corridor_->back().curve_length_ < last_traj_data_.corridor_.back().curve_length_) && !reach_end){
        
        if(start_curve_length > 0.5 && start_curve_length + predict_start_length > last_traj_data_.predict_start_length_ - 0.1 && start_curve_length + plan_corridor_->back().curve_length_ > last_traj_data_.corridor_.back().curve_length_ - 0.1){
            ROS_DEBUG("Accept new traj even no extension.");
        }
        else{
            return plan_status::NO_EXTENSION;
        }

    }

    optical_flow_estimator_->set_forward_cross_sections();

   
    
    double traj_time_res = 0.4;
    vector<double> traj_length_sample;

    double end_v = 0.0;
    double start_v = start_vel.norm();

    int search_state = bspline_optimizer_1d_->NO_PATH;
    
    if(adaptive_speed_){
        if(!reach_end){
            search_state = bspline_optimizer_1d_->astar_search(start_v);
            traj_length_sample = bspline_optimizer_1d_->get_init_sample(traj_time_res, end_v);
        }
        else{
            search_state = bspline_optimizer_1d_->const_dcc_search(start_v);
            end_v = 0.0;
            double tmp_param = 0.0;
            traj_length_sample = bspline_optimizer_1d_->get_init_sample(traj_time_res, tmp_param);
            ROS_INFO("Exit, start dcc!");
        }

    }
    else{
        if(!reach_end){
            end_v = flight_speed_;
            search_state = bspline_optimizer_1d_->const_speed_search(start_v, end_v);
        }
        else{
            search_state = bspline_optimizer_1d_->const_dcc_search(start_v);
            end_v = 0.0;
        }

        double tmp_param = 0.0;
        traj_length_sample = bspline_optimizer_1d_->get_init_sample(traj_time_res, tmp_param);

    }

    if(search_state == bspline_optimizer_1d_->NO_PATH){
        // ROS_ERROR("search fail!");
        plan_corridor_->clear();
        return plan_status::UNKNOWN_FAILURE;
    }
    

    ros::Time final_opt_time = ros::Time::now();
    

    double tangent_acc = start_vel.squaredNorm() > 1e-4 ?
        start_acc.dot(start_vel.normalized()) : 0.0;

    NonUniformBspline final_traj_1d = traj_opt_1D(traj_time_res, traj_length_sample, start_vel.norm(), tangent_acc, end_v);
    

    double process = tunnel_center_line_->getTimeFromLength(final_traj_1d.evaluateDeBoorT(0.0)(0));
    Vector3d center_line_vel = tunnel_center_vel_->evaluateDeBoorT(process);
    Vector3d opt_start_vel = final_traj_1d.getDerivative().evaluateDeBoorT(process)(0) * center_line_vel / center_line_vel.norm();




    Vector3d traj_start_pos = tunnel_center_line_->evaluateDeBoorT(0.0);
      
    last_traj_data_.start_time_ = plan_odom_time_;
    last_traj_data_.traj_id_++;
    last_traj_data_.corridor_ = *plan_corridor_;
    last_traj_data_.start_pos_ = traj_start_pos;
    last_traj_data_.traj_valid_ = true;

    last_traj_data_.position_traj_ = *tunnel_center_line_;
    last_traj_data_.velocity_traj_ = *tunnel_center_vel_;
    last_traj_data_.acceleration_traj_ = *tunnel_center_acc_;

    last_traj_data_.yaw_traj_ = *tunnel_center_yaw_;
    last_traj_data_.yaw_dot_traj_ = *tunnel_center_yaw_dot_;

    last_traj_data_.position_traj_1d_ = final_traj_1d;
    last_traj_data_.velocity_traj_1d_ = final_traj_1d.getDerivative();
    last_traj_data_.acceleration_traj_1d_ = last_traj_data_.velocity_traj_1d_.getDerivative();
    
    last_traj_data_.predict_start_length_ = predict_start_length;


    Bspline_with_retiming traj2pub;
    traj2pub.have_retiming = true;
    traj2pub.start_time = plan_odom_time_;
    traj2pub.traj_id = last_traj_data_.traj_id_;
    traj2pub.yaw_strategy = yaw_stragety::PLAN;

    MatrixXd ctrl_pt_3d = tunnel_center_line_->getControlPoint();
    VectorXd knot_pt_3d = tunnel_center_line_->getKnot();

    traj2pub.order_3d = 3;

    for(int ctrl_pt_idx = 0; ctrl_pt_idx < ctrl_pt_3d.rows(); ctrl_pt_idx++){
        geometry_msgs::Point pos_pt;
        pos_pt.x = ctrl_pt_3d(ctrl_pt_idx, 0);
        pos_pt.y = ctrl_pt_3d(ctrl_pt_idx, 1);
        pos_pt.z = ctrl_pt_3d(ctrl_pt_idx, 2);
        traj2pub.pos_pts_3d.emplace_back(pos_pt);
    }
    for(int knot_pt_idx = 0; knot_pt_idx < knot_pt_3d.rows(); knot_pt_idx++){
        traj2pub.knots_3d.emplace_back(knot_pt_3d(knot_pt_idx));
    }

    VectorXd ctrl_pt_yaw = tunnel_center_yaw_->getControlPoint();
    VectorXd knot_pt_yaw = tunnel_center_yaw_->getKnot();

    traj2pub.order_yaw = 3;

    for(int ctrl_pt_idx = 0; ctrl_pt_idx < ctrl_pt_yaw.rows(); ctrl_pt_idx++){
        traj2pub.yaw_pts.emplace_back(ctrl_pt_yaw(ctrl_pt_idx));
    }
    for(int knot_pt_idx = 0; knot_pt_idx < knot_pt_yaw.rows(); knot_pt_idx++){
        traj2pub.knots_yaw.emplace_back(knot_pt_yaw(knot_pt_idx));
    }


    VectorXd ctrl_pt_1d = final_traj_1d.getControlPoint();
    VectorXd knot_pt_1d = final_traj_1d.getKnot();

    traj2pub.order_1d = 3;

    for(int ctrl_pt_idx = 0; ctrl_pt_idx < ctrl_pt_1d.rows(); ctrl_pt_idx++){
        traj2pub.pos_pts_1d.emplace_back(ctrl_pt_1d(ctrl_pt_idx));
    }

    for(int knot_pt_idx = 0; knot_pt_idx < knot_pt_1d.rows(); knot_pt_idx++){
        traj2pub.knots_1d.emplace_back(knot_pt_1d(knot_pt_idx));
    }


    traj_full_pub_.publish(traj2pub);


    if(reach_end){
        return plan_status::REACH_EXIT;
    }
    else{
        return plan_status::ACCEPT;
    }

    
}


NonUniformBspline tunnel_planner::extract_tunnel_center_line(double ts, const Vector3d& start_curvature){

    vector<Vector3d> start(3, Vector3d::Zero());
    start[0] = plan_corridor_->front().center_;
    start[1] = plan_corridor_->front().w_R_cs.col(2) * virtual_flight_progress_speed_;
    start[2] = virtual_flight_progress_speed_ * virtual_flight_progress_speed_ * start_curvature;

    vector<Vector3d> end(3, Vector3d::Zero());
    end[0] = plan_corridor_->back().center_;
    end[1] = plan_corridor_->back().w_R_cs.col(2) * virtual_flight_progress_speed_;

    vector<Vector3d> start_end_derivative(4, Vector3d::Zero());
    start_end_derivative[0] = start[1];
    start_end_derivative[2] = end[1];

    vector<Vector3d> center_pts;
    for(auto& cs : *plan_corridor_){
        center_pts.emplace_back(cs.center_);
    }

    MatrixX3d ctrl_pts;
    NonUniformBspline::parameterizeToBspline(ts, center_pts, start_end_derivative, 3, ctrl_pts);


    Matrix3d start_ctrl_pts;
    NonUniformBspline::solveFirst3CtrlPts(ts, start, start_ctrl_pts);

    ctrl_pts.topRows(3) = start_ctrl_pts.topRows(3);


    NonUniformBspline pos_init = NonUniformBspline(ctrl_pts, 3, ts);


    vector<int> way_pt_idx;
    for (int i = 0; i < center_pts.size(); way_pt_idx.emplace_back(i), i++);
    
    int cost_function = bspline_optimizer_->WAY_PT_JERK_VEL_START_HARD_PHASE;

    bspline_optimizer_->setBoundaryStates(start, end);
    bspline_optimizer_->setWaypoints(center_pts, way_pt_idx);

    bspline_optimizer_->optimize(ctrl_pts, ts, cost_function, 0, 0);

    return NonUniformBspline(ctrl_pts, 3, ts);
}

bool tunnel_planner::detect_corridor(double& ts){

    bool reach_exit = false;

    int outside_cs_cnt = 0;

    double total_time = tunnel_center_line_->getTimeSum();
    int seg_num = int(ceil(total_time / ts));
    ts = total_time / seg_num;

    NonUniformBspline center_line_vel = tunnel_center_line_->getDerivative();


    Vector3d enter_tunnel_pos = tunnel_entrance_pos_;
    Vector3d enter_tunnel_dir = tunnel_entrance_dir_;

    int predict_start_idx = plan_corridor_->size() - 1;
    for(; predict_start_idx >= 0 && plan_corridor_->at(predict_start_idx).is_predict; predict_start_idx--);

    Vector3d predict_start_pos = plan_corridor_->at(predict_start_idx).center_;
    Vector3d predict_start_dir = plan_corridor_->at(predict_start_idx).w_R_cs.col(2);   

    plan_corridor_->resize(1);


    Vector3d pos = plan_corridor_->front().center_, last_pos = pos;
    Vector3d front_dir = plan_corridor_->front().w_R_cs.col(2);

    cv::Mat local_cross_section;
    cross_section plan_cross_section = plan_corridor_->front();

    bool enter_tunnel = plan_corridor_->front().cross_section_shape_ != tunnel_shape::BEFORE;

    const int OUTSIDE_CS_CNT_ENOUGH_NUM = 2;

    const double out_check_step = 0.2;
    const double max_out_check_dist = 0.8;
    if(enter_tunnel){
        int outside_cs_cnt = 0;
        for(double out_check_dist = 0.0; out_check_dist <= max_out_check_dist; out_check_dist += out_check_step){
            Vector3d cur_check_pos = plan_corridor_->front().center_ + out_check_dist * front_dir;
            bool out_cross_section_exist = construct_cross_section(cur_check_pos, front_dir, plan_cross_section.w_R_cs, local_cross_section);
            
            if(!out_cross_section_exist){
                outside_cs_cnt++;
            }
            else{
                outside_cs_cnt = 0;
            }

            if(outside_cs_cnt >= OUTSIDE_CS_CNT_ENOUGH_NUM){
                reach_exit = true;

                Vector3d out_step = front_dir * ts;

                double past_valid_dist = 0.0;
                for(auto past_it = past_corridor_->rbegin(), next_past_it = next(past_it); next_past_it != past_corridor_->rend(); past_it = next_past_it, next_past_it = next(next_past_it)){
                    past_valid_dist += (next_past_it->center_ - past_it->center_).norm();

                    if(past_valid_dist > 0.5){
                        out_step = (plan_corridor_->front().center_ - next_past_it->center_).normalized() * ts;
                        break;
                    }

                }

                plan_corridor_->resize(1);

                double out_step_norm = out_step.norm();

                plan_cross_section.w_R_cs = plan_corridor_->back().w_R_cs;
                plan_cross_section.center_ = plan_corridor_->back().center_ + out_step;
                plan_cross_section.cross_section_shape_ = tunnel_shape::OUTSIDE;

                const double extend_dist = 0.6;
                
                for(double dist = 0.0; dist < extend_dist; dist += out_step_norm, plan_cross_section.center_ = plan_cross_section.center_ + out_step){
                    if(edf_map_generator_ptr_->get_dist(plan_cross_section.center_) <= drone_radius_ && plan_corridor_->size() > 2 && dist > 0.0){
                        plan_corridor_->pop_back();
                        ROS_ERROR("unsafe end pt outside!");
                        break;
                    }                
                    plan_corridor_->emplace_back(plan_cross_section);
                }

                return true;
            }

        }

    }

    ros::Time t0 = ros::Time::now();
    for(int way_pt_idx = 1; way_pt_idx <= seg_num; way_pt_idx++, last_pos = pos){

        double way_pt_time = way_pt_idx * ts;
        
        pos = tunnel_center_line_->evaluateDeBoorT(way_pt_time);
        Vector3d vel = center_line_vel.evaluateDeBoorT(way_pt_time);
        double v_norm = vel.norm();
        Vector3d dir = vel / v_norm;
        

        if(!enter_tunnel){
            plan_cross_section.center_ = pos;
            plan_cross_section.w_R_cs = cal_w_R_plane(dir);
            plan_cross_section.cross_section_shape_ = tunnel_shape::BEFORE;

            if((last_pos - tunnel_entrance_pos_).dot(tunnel_entrance_dir_) * (pos - tunnel_entrance_pos_).dot(tunnel_entrance_dir_) <= 0.0){
                enter_tunnel = true;
                plan_corridor_->emplace_back(plan_cross_section);
                plan_cross_section.cross_section_shape_ = tunnel_shape::UNKNOWN;
                continue;              
            }
        }
        else{
            bool cross_section_exist = construct_cross_section(pos, dir, plan_cross_section.w_R_cs, local_cross_section);
            if(!cross_section_exist){
                outside_cs_cnt++;
            }
            else{
                outside_cs_cnt = 0;
            }


            plan_cross_section = detect_cross_section_shape(local_cross_section, plan_cross_section.w_R_cs, pos, plan_cross_section, cross_section_exist);
        }

        plan_corridor_->emplace_back(plan_cross_section);

        if(!plan_cross_section.is_predict){
            if((last_pos - predict_start_pos).dot(predict_start_dir) * (pos - predict_start_pos).dot(predict_start_dir) <= 0.0){
                plan_cross_section.is_predict = true;
            }
        }

    }

    return reach_exit;

}

NonUniformBspline tunnel_planner::plan_tunnel_center_yaw(double ts, const double& start_yaw, const double& start_yaw_dot, const bool change_yaw){
    
    vector<double> yaw_init_way_pts, yaw_way_pts;
    vector<int> way_pt_idx;
    yaw_init_way_pts.emplace_back(start_yaw);
    yaw_way_pts.emplace_back(start_yaw);
    way_pt_idx.emplace_back(0);

    vector<double> start;
    start.emplace_back(start_yaw);
    start.emplace_back(start_yaw_dot);
    start.emplace_back(0.0);

    double total_time = tunnel_center_line_->getTimeSum();
    int seg_num = int(ceil(total_time / ts));
    ts = total_time / seg_num;

    double max_yaw_change_per_interval = ts * virtual_flight_progress_speed_ * max_yaw_change_over_distance_;


    double last_yaw = start_yaw;
    double cur_yaw = start_yaw;


    if(change_yaw){
        tunnel_center_line_->computeKnotLength(1e-2);
        VectorXd& knot_vec = tunnel_center_line_->getKnotLength();
        int knot_idx_off_set = 0;
        for(int knot_idx = 0; knot_idx < knot_vec.rows(); knot_idx++){
            if(knot_vec(knot_idx) > 1e-4){
                knot_idx_off_set = knot_idx - 1;
                break;
            }
        }

        double center_line_total_length = knot_vec(knot_vec.rows()-1);

        double yaw_dir_curvature_ratio = 0.0;

        double out_check_step = 0.5 * edf_map_generator_ptr_->get_edf_map_ptr()->map_res;
        double max_out_check_dist = plan_range_;

        Vector3d front_dir = plan_corridor_->front().w_R_cs.col(2);
        bool free_ahead = true;

        for(double out_check_dist = 0.0; out_check_dist <= max_out_check_dist; out_check_dist += out_check_step){
            Vector3d cur_check_pos = plan_corridor_->front().center_ + out_check_dist * front_dir;
            if(edf_map_generator_ptr_->get_type(cur_check_pos) != edf_voxel::FREE){
                free_ahead = false;
                break;
            }
        }

        if(center_line_total_length < plan_range_ && !free_ahead){
            yaw_dir_curvature_ratio =  max_yaw_dir_curvature_ratio_ * pow(center_line_total_length / plan_range_ - 1.0, 2);
        }

        Vector3d center_line_end_pos = tunnel_center_line_->evaluateDeBoorT(total_time);
        Vector3d center_line_end_dir = tunnel_center_vel_->evaluateDeBoorT(total_time).normalized();
        
        
        Vector3d yaw_des_pos;
        Vector2d yaw_dir;


        for(int seg_idx = 1; seg_idx <= seg_num; seg_idx++){
            double cur_knot_length = knot_vec(knot_idx_off_set + seg_idx);
            double process = seg_idx * ts;
            Vector3d cur_pos = tunnel_center_line_->evaluateDeBoorT(process);
            Vector3d cur_dir = tunnel_center_vel_->evaluateDeBoorT(process).normalized();

            double yaw_des_length = cur_knot_length + yaw_ahead_length_;

            if(yaw_des_length < center_line_total_length){
                double yaw_des_t = tunnel_center_line_->getTimeFromLength(yaw_des_length, 1e-2);
                yaw_des_pos = tunnel_center_line_->evaluateDeBoorT(yaw_des_t);
            }
            else{
                double extend_length = yaw_des_length - center_line_total_length;
                yaw_des_pos = center_line_end_pos + extend_length * center_line_end_dir;

            }


            Vector3d yaw_dir_3d = (yaw_des_pos - cur_pos).normalized();

            if(fabs(yaw_dir_3d.z()) > 0.7){
                cur_yaw = last_yaw;
            }
            else if(fabs(cur_dir.z()) > 0.7){
                cur_yaw = calc_des_yaw(last_yaw, atan2(yaw_dir_3d.y(), yaw_dir_3d.x()), max_yaw_change_per_interval);
                yaw_way_pts.emplace_back(cur_yaw);
                way_pt_idx.emplace_back(seg_idx);
            }
            else{
                Vector3d vel = tunnel_center_vel_->evaluateDeBoorT(process);
                Vector2d vel_h = vel.head(2);
                Vector2d acc_h = tunnel_center_acc_->evaluateDeBoorT(process).head(2);

                double inv_v_squared_norm = 1.0 / (vel_h.squaredNorm());

                Vector2d cent_acc_h = acc_h - acc_h.dot(vel_h) * vel_h * inv_v_squared_norm;

                yaw_dir = yaw_dir_3d.head(2).normalized() + yaw_dir_curvature_ratio * inv_v_squared_norm * cent_acc_h;

                cur_yaw = calc_des_yaw(last_yaw, atan2(cur_dir.y(), cur_dir.x()),  atan2(yaw_dir.y(), yaw_dir.x()), max_yaw_change_per_interval, max_yaw_center_line_dir_diff_);

                yaw_way_pts.emplace_back(cur_yaw);
                way_pt_idx.emplace_back(seg_idx);
            }

            
            yaw_init_way_pts.emplace_back(cur_yaw);

            last_yaw = cur_yaw;

        }
    }
    else{
        for(int seg_idx = 1; seg_idx <= seg_num; seg_idx++){
            yaw_init_way_pts.emplace_back(cur_yaw);
        }
    }

    

    vector<double> end(3, 0.0);
    end[0] = yaw_init_way_pts.back();

    vector<double> start_end_derivative(4, 0.0);
    start_end_derivative[0] = start[1];
    start_end_derivative[1] = start[2];

    VectorXd yaw_ctrl_pts;
    NonUniformBspline::parameterizeToBspline(ts, yaw_init_way_pts, start_end_derivative, 3, yaw_ctrl_pts);

    Vector3d start_yaw_ctrl_pts;
    NonUniformBspline::solveFirst3CtrlPts(ts, start, start_yaw_ctrl_pts);
    yaw_ctrl_pts.head(3) = start_yaw_ctrl_pts; 

    NonUniformBspline yaw_traj_init = NonUniformBspline(yaw_ctrl_pts, 3, ts);


    bspline_optimizer_1d_->set_waypoints(yaw_way_pts, way_pt_idx);
    bspline_optimizer_1d_->set_waypoint_weight(W_YAW_WAYPT);

    int cost_function_1 = bspline_optimizer_1d_->WAY_PT_YAW_ACC_PHASE;
    bspline_optimizer_1d_->set_boundary_states(start, end);
    bspline_optimizer_1d_->optimize_1d(yaw_ctrl_pts, ts, cost_function_1);

    NonUniformBspline yaw_traj(yaw_ctrl_pts, 3, ts);

    return NonUniformBspline(yaw_ctrl_pts, 3, ts);
}

NonUniformBspline tunnel_planner::traj_opt_1D(double& ts, const vector<double>& way_pts, const double& start_v, const double& start_a, const double& end_v){
    vector<double> start;
    start.emplace_back(0.0);
    start.emplace_back(start_v);
    start.emplace_back(start_a);

    vector<double> end(3, 0.0);
    end[0] = way_pts.back();
    end[1] = end_v;

    vector<double> start_end_derivative(4, 0.0);
    start_end_derivative[0] = start[1];
    start_end_derivative[1] = start[2];

    start_end_derivative[2] = end[1];

    VectorXd traj_ctrl_pts_1d;
    NonUniformBspline::parameterizeToBspline(ts, way_pts, start_end_derivative, 3, traj_ctrl_pts_1d);

    Vector3d start_ctrl_pts;
    NonUniformBspline::solveFirst3CtrlPts(ts, start, start_ctrl_pts);

    traj_ctrl_pts_1d.head(3) = start_ctrl_pts; 

    NonUniformBspline traj_init_1d = NonUniformBspline(traj_ctrl_pts_1d, 3, ts);

    if(adaptive_speed_){
        vector<int> way_pt_idx;
        for (int i = 0; i < way_pts.size(); way_pt_idx.emplace_back(i), i++){
            // cout<<"way pt "<<i<<" : "<<way_pts[i]<<endl;
        }
        bspline_optimizer_1d_->set_waypoints(way_pts, way_pt_idx);
        bspline_optimizer_1d_->set_waypoint_weight(W_WAYPT);

        int cost_function_1 = bspline_optimizer_1d_->WAY_PT_ACC_JERK_FEASI_PHASE;
        
        bspline_optimizer_1d_->set_boundary_states(start, end);
        bspline_optimizer_1d_->optimize_1d(traj_ctrl_pts_1d, ts, cost_function_1);

        NonUniformBspline traj_final_1d = NonUniformBspline(traj_ctrl_pts_1d, 3, ts);

        int cost_function_2 = bspline_optimizer_1d_->ACC_JERK_OF_AD_START_END_HARD_PHASE;
        bspline_optimizer_1d_->set_boundary_states(start, end);
        bspline_optimizer_1d_->optimize_1d(traj_ctrl_pts_1d, ts, cost_function_2);
    }
    else{
        int cost_function = bspline_optimizer_1d_->JERK_FEASI_SPEED_PHASE;
        
        bspline_optimizer_1d_->set_boundary_states(start, end);
        bspline_optimizer_1d_->optimize_1d(traj_ctrl_pts_1d, ts, cost_function);
    }


    return NonUniformBspline(traj_ctrl_pts_1d, 3, ts);
}



bool tunnel_planner::get_inter_cross_section_curve_length_vertical_cross_section_range(NonUniformBspline& center_line, double& predict_start_length){

    vert_sections_->clear();

    plan_corridor_->front().curve_length_ = 0.0;
    
    Vector3d vel_a = tunnel_center_vel_->evaluateDeBoorT(0.0), vel_b;
    double va = vel_a.norm(), vb, v_half_ab;
    Vector3d pos_a = center_line.evaluateDeBoorT(0.0), pos_b;
    double res = min(5e-3, center_line.getKnotSpan());
    double dur = center_line.getTimeSum();

    Vector2d acc_h = tunnel_center_acc_->evaluateDeBoorT(0.0).head(2);
    Vector2d vel_a_h = vel_a.head(2);
    double squared_va_h = vel_a_h.squaredNorm();
    plan_corridor_->front().curvature_ = (acc_h - acc_h.dot(vel_a_h) / squared_va_h * vel_a_h).norm() / squared_va_h;
    
    plan_corridor_->front().yaw_ = tunnel_center_yaw_->evaluateDeBoorT(0.0)(0);
    plan_corridor_->front().yaw_dot_ = tunnel_center_yaw_dot_->evaluateDeBoorT(0.0)(0);

    int curve_idx = 1;
    Vector3d cs_center_compare = plan_corridor_->at(curve_idx).center_;
    Vector3d cs_normal = plan_corridor_->at(curve_idx).w_R_cs.col(2);

    bool in_vert = false;
    bool predict_start = false;
    vert_section cur_vert_section;
    cv::Mat local_cross_section;

    Eigen::VectorXd& center_line_knot = center_line.getKnotRef();
    int centerline_order = center_line.getOrder();
    Eigen::VectorXd& center_line_knot_length = center_line.getKnotLength();
    center_line_knot_length.setZero(center_line_knot.rows());
    double computing_knot_idx = centerline_order+1;
    double knot_t_offset = center_line_knot(centerline_order);


    double total_length = 0.0;
    for (double t = res, dl = 0.0, computing_knot_t = center_line_knot(computing_knot_idx)- knot_t_offset; t <= dur; t += res, va = vb, pos_a = pos_b, vel_a = vel_b)
    {

        vel_b = tunnel_center_vel_->evaluateDeBoorT(t);
        vb = vel_b.norm();
        pos_b = center_line.evaluateDeBoorT(t);
        v_half_ab = tunnel_center_vel_->evaluateDeBoorT(t-0.5*res).norm();
        dl = res / 6.0 * (va+ 4.0*v_half_ab+ vb);

        total_length += dl;

        if(t+res >= computing_knot_t){
            va = vb;
            vel_b = tunnel_center_vel_->evaluateDeBoorT(computing_knot_t);
            vb = vel_b.norm();
            pos_b = center_line.evaluateDeBoorT(computing_knot_t);
            v_half_ab = tunnel_center_vel_->evaluateDeBoorT(0.5*(t+computing_knot_t)).norm();
            dl = (computing_knot_t-t) / 6.0 * (va+ 4.0*v_half_ab+ vb);

            total_length += dl;
            t = computing_knot_t;

            center_line_knot_length(computing_knot_idx) = total_length;
            computing_knot_idx++;
            computing_knot_t = center_line_knot(computing_knot_idx)- knot_t_offset;

        }

        if(t == dur){
            
            plan_corridor_->at(curve_idx).curve_length_ = total_length;

            Vector2d acc_h = tunnel_center_acc_->evaluateDeBoorT(t).head(2);
            Vector2d vel_b_h = vel_b.head(2);
            double squared_vb_h = vel_b_h.squaredNorm();
            plan_corridor_->at(curve_idx).curvature_ = (acc_h - acc_h.dot(vel_b_h) / squared_vb_h * vel_b_h).norm() / squared_vb_h;

            plan_corridor_->at(curve_idx).yaw_ = tunnel_center_yaw_->evaluateDeBoorT(t)(0);
            plan_corridor_->at(curve_idx).yaw_dot_ = tunnel_center_yaw_dot_->evaluateDeBoorT(t)(0);

            if(!predict_start){
                predict_start_length = total_length;
            }

            if(curve_idx != plan_corridor_->size() - 1){
                // ROS_ERROR("idx error!!!!!!!!!!!!!");
                return false;
            }

            if(in_vert){
                cur_vert_section.exit_length = total_length;
                cur_vert_section.exit_t = t;
                cur_vert_section.exit_yaw = cur_vert_section.entry_yaw;


                vert_sections_->emplace_back(cur_vert_section);
                in_vert = false;
            }
            break;
        }


        if(!in_vert){
            if(abs(vel_b.z() / vb) > VERT_SECTION_COS_THRESHOLD){
                cur_vert_section.entry_length = total_length;
                cur_vert_section.entry_t = t;
                cur_vert_section.entry_yaw = atan2(vel_b.y(), vel_b.x());

                in_vert = true;
            }
        }
        else{
            if(abs(vel_b.z() / vb) < VERT_SECTION_COS_THRESHOLD){
                cur_vert_section.exit_length = total_length;
                cur_vert_section.exit_t = t;
                cur_vert_section.exit_yaw = atan2(vel_b.y(), vel_b.x());

                vert_sections_->emplace_back(cur_vert_section);
                in_vert = false;
            }
        }

        Vector3d pos_a_cs_diff = pos_a-cs_center_compare;
        Vector3d pos_b_cs_diff = pos_b-cs_center_compare;

        if((pos_a_cs_diff.dot(cs_normal)) * (pos_b_cs_diff.dot(cs_normal)) <= 0.0 && curve_idx + 1 < plan_corridor_->size()){
            
            plan_corridor_->at(curve_idx).curve_length_ = total_length;
            
            Vector2d acc_h = tunnel_center_acc_->evaluateDeBoorT(t).head(2);
            Vector2d vel_b_h = vel_b.head(2);
            double squared_vb_h = vel_b_h.squaredNorm();

            plan_corridor_->at(curve_idx).curvature_ = (acc_h - acc_h.dot(vel_b_h) / squared_vb_h * vel_b_h).norm() / squared_vb_h;
            
            plan_corridor_->at(curve_idx).yaw_ = tunnel_center_yaw_->evaluateDeBoorT(t)(0);
            plan_corridor_->at(curve_idx).yaw_dot_ = tunnel_center_yaw_dot_->evaluateDeBoorT(t)(0);
            
            if(!predict_start && plan_corridor_->at(curve_idx).is_predict){
                predict_start_length = total_length;
                predict_start = true;
            }

           
            curve_idx++;
            cs_center_compare = plan_corridor_->at(curve_idx).center_;
            cs_normal = plan_corridor_->at(curve_idx).w_R_cs.col(2);

        }

    }

    for(; computing_knot_idx < center_line_knot.rows(); computing_knot_idx++){
        center_line_knot_length(computing_knot_idx) = total_length;
    }

    return true;

}


int tunnel_planner::search_corridor_path_hough(const cross_section& start_cross_section, const double plan_step, double &range, const double max_edf_tol, const double min_dir_tol)
{
    Vector3d center_pt = start_cross_section.center_, last_center_pt = start_cross_section.center_;
    const Vector3d start_dir = start_cross_section.w_R_cs.col(2);
    Vector3d tunnel_dir = start_dir, last_dir = start_dir;


    cross_section plan_cross_section {start_cross_section};
    if(plan_cross_section.cross_section_data_.empty()){
        plan_cross_section.cross_section_data_.assign(1, 0.05);
    }
    Matrix3d w_R_cs = Matrix3d::Identity();

    cv::Mat local_cross_section;
    bool cross_section_exist = true;


    if(start_cross_section.cross_section_shape_ == tunnel_shape::UNKNOWN){
        // detect first shape
        cross_section_exist = construct_cross_section(start_cross_section.center_, start_cross_section.w_R_cs.col(2), w_R_cs, local_cross_section);

        if(plan_corridor_->empty()){
            plan_cross_section = detect_cross_section_shape(local_cross_section, w_R_cs, start_cross_section.center_, plan_cross_section, cross_section_exist);
        }
                
        plan_cross_section.center_ = center_pt;
        plan_cross_section.w_R_cs = w_R_cs;

        if(!cross_section_exist){
            plan_cross_section.cross_section_shape_ = tunnel_shape::OUTSIDE;
        }
    }
    
    center_pt = plan_cross_section.center_;

    plan_corridor_->emplace_back(plan_cross_section);

    double plan_dist = 0.0;

    bool predict_tunnel = false;

    ros::Time edf_wait_t = ros::Time::now();
    edf_map_generator_ptr_->wait_for_edf_available(1e-4);

    Vector3d edf_min_pos, edf_max_pos;
    edf_map_generator_ptr_->get_edf_range(edf_min_pos, edf_max_pos);

    const double CHECK_RES = 0.02;

    for (double last_radius = max_edf_tol, predict_max_edf_value = 0.0, outside_dist = 0.0, cs_interval_dist = 0.0; plan_dist < range; plan_dist += (center_pt - last_center_pt).norm(), cs_interval_dist += (center_pt - last_center_pt).norm(), last_center_pt = center_pt)
    {


        bool step_valid = true;

        for(double check_step = 0.0; check_step < plan_step; ){

            if(check_step + CHECK_RES > plan_step){
                check_step = plan_step;
            }
            else{
                check_step += CHECK_RES;
            }

            center_pt = last_center_pt + check_step * tunnel_dir;

            if (!edf_map_generator_ptr_->get_edf_map_ptr()->in_map(center_pt) ||
                center_pt.x() < edf_min_pos.x() || center_pt.y() < edf_min_pos.y() || center_pt.z() < edf_min_pos.z() ||
                center_pt.x() > edf_max_pos.x() || center_pt.y() > edf_max_pos.y() || center_pt.z() > edf_max_pos.z() ||
                edf_map_generator_ptr_->get_type(center_pt) != edf_voxel::FREE || start_dir.dot(tunnel_dir) < -0.3)
            {
                step_valid = false;
                break;
            }
        }

        if(!step_valid){
            break;
        }

        double max_edf_value = 0.0;
        Vector3d tmp_center_pt = center_pt; 
        if(predict_tunnel){

            max_edf_value = find_max_edf_in_plane(center_pt, tunnel_dir, edf_map_generator_ptr_->get_edf_map_ptr()->map_res, grad_max_res_, predict_max_edf_value);
            plan_cross_section.cross_section_data_[0] = max_edf_value;

            last_dir = tunnel_dir;
            tunnel_dir = (center_pt - plan_corridor_->back().center_).normalized();
            
        }
        else{
            max_edf_value = find_max_edf_in_plane(tmp_center_pt, tunnel_dir, edf_map_generator_ptr_->get_edf_map_ptr()->map_res, grad_max_res_, 1.5 * last_radius);
            if(max_edf_value > 1.1 * last_radius){
                max_edf_value = find_max_edf_in_plane(center_pt, tunnel_dir, edf_map_generator_ptr_->get_edf_map_ptr()->map_res, grad_max_res_, 1.0 * last_radius);
            }
            else{
                center_pt = tmp_center_pt;
            }

            last_dir = tunnel_dir;
            const double new_dir_ratio = 0.85;
            tunnel_dir = new_dir_ratio * find_tunnel_dir(center_pt, max_edf_value, tunnel_dir) + (1.0 - new_dir_ratio) * last_dir;
       
        }

        if(edf_map_generator_ptr_->get_dist(center_pt) < drone_radius_){
            // ROS_ERROR("small edf value, %lf", edf_map_generator_ptr_->get_dist(center_pt));
            break;
        }


        if((center_pt - last_center_pt).squaredNorm() < 1e-6){
            ROS_ERROR("no step forward");
            break;
        }

        if(cs_interval_dist >= cross_section_step_res_){
            cross_section_exist = construct_cross_section(center_pt, tunnel_dir, w_R_cs, local_cross_section);
            if(!cross_section_exist){
                cross_section_exist = construct_cross_section(center_pt, last_dir, w_R_cs, local_cross_section);
            }

            if(!cross_section_exist){
                plan_cross_section.cross_section_shape_ = tunnel_shape::OUTSIDE;
            }
        }

        plan_cross_section.center_ = center_pt;
        plan_cross_section.cross_section_data_[0] = max_edf_value;

 
        center_pt = plan_cross_section.center_;
        
        if(plan_cross_section.cross_section_shape_ == OUTSIDE){
            outside_dist += (center_pt - last_center_pt).norm();
            if(outside_dist > 0.2){
                for(int pop_size = 0; pop_size < 3 && plan_corridor_->size() > 2; pop_size++, plan_corridor_->pop_back());
                
                center_pt = plan_corridor_->back().center_;
                last_dir = plan_corridor_->back().w_R_cs.col(2);

                for(outside_dist = 0.0 ;outside_dist < 0.6; outside_dist += cross_section_step_res_){
                    center_pt += last_dir * cross_section_step_res_;
                    
                    plan_cross_section.center_ = center_pt;
                    plan_cross_section.w_R_cs = w_R_cs;
                    
                    plan_corridor_->emplace_back(plan_cross_section);
                }

                edf_map_generator_ptr_->release_edf_resource();
                return corridor_status::CORRIDOR_REACH_EXIT;
            }

        }

        if(predict_tunnel){
            plan_cross_section.is_predict = true;

            if (max_edf_value > max_edf_tol || last_dir.dot(tunnel_dir) < min_dir_tol){
                tunnel_dir = last_dir;
            }

        }
        else{
            if (max_edf_value > max_edf_tol || (max_edf_value > 1.5 * last_radius && plan_dist > 0.0) ||last_dir.dot(tunnel_dir) < min_dir_tol){
                tunnel_dir = last_dir;
                w_R_cs = plan_corridor_->back().w_R_cs;

                predict_tunnel = true;
                predict_max_edf_value = max_edf_value;
                // break;
            }
        }

        last_radius = max_edf_value;
        plan_cross_section.w_R_cs = w_R_cs;
        
        if(cs_interval_dist >= cross_section_step_res_){
            plan_corridor_->emplace_back(plan_cross_section);
            
            cs_interval_dist = 0.0;
        }
        
    }

    edf_map_generator_ptr_->release_edf_resource();

    range = plan_dist;
    return plan_corridor_->size() > 2 ? corridor_status::NORMAL : corridor_status::NO_CORRIDOR;

}

cross_section tunnel_planner::detect_cross_section_shape(const cv::Mat &cross_section_img, const Matrix3d& w_R_cs, const Vector3d& w_t_cs, const cross_section &prev_cross_section, bool cross_section_exist){

    cross_section cross_section_return(prev_cross_section);
    cross_section_return.center_ = w_t_cs;
    cross_section_return.w_R_cs = w_R_cs;
    
    if(cross_section_return.cross_section_shape_ == BEFORE){
        return cross_section_return;
    }

    if(!cross_section_exist){
        cross_section_return.cross_section_shape_ = OUTSIDE;
        return cross_section_return;
    }

    int shape = classify_shape(cross_section_img);

    switch(shape){
        case tunnel_shape::CIRCLE:{
            int circle_outlier_cnt = 0;
            bool have_circle = detect_circle(cross_section_img, w_R_cs, cross_section_return, circle_outlier_cnt);
           
            break;
        }
        case tunnel_shape::RECTANGLE:{
            int rect_outlier_cnt = 0;
            bool have_rect = detect_rectangle(cross_section_img, w_R_cs, cross_section_return, rect_outlier_cnt);

            break;
        }

        default:
            break;
    }

       
    return cross_section_return;

}

int tunnel_planner::classify_shape(const cv::Mat &cross_section_img){

    auto t0 = ros::Time::now();

    int crop_rows = cross_section_img.rows % shape_classifier_input_dim_.x();
    int crop_cols = cross_section_img.cols % shape_classifier_input_dim_.y();

    int reserve_rows = cross_section_img.rows - crop_rows;
    int reserve_cols = cross_section_img.rows - crop_cols;

    int offsetH = crop_rows / 2;
    int offsetW = crop_cols / 2;

    const Rect roi(offsetW, offsetH, reserve_cols, reserve_rows);

    cv::Mat cropped_img = cross_section_img(roi).clone();

    cv::resize(cropped_img, cropped_img, Size(shape_classifier_input_dim_.x(), shape_classifier_input_dim_.y()), 0, 0, INTER_LINEAR);

    vector<MatrixXf> classifier_input(1, MatrixXf::Zero(shape_classifier_input_dim_.x(), shape_classifier_input_dim_.y()));
    cv2eigen(cropped_img, classifier_input[0]);

    auto data_ptr = classifier_input[0].data();
    for(int i = 0; i < classifier_input[0].size(); i++, data_ptr++){
        *data_ptr = (*data_ptr) > 127.0f ? 1.0f : 0.0f;
    }


    vector<MatrixXf> classifier_output;
    shape_classifier_net_->inference_net(classifier_input, classifier_output);


    return classifier_output[0].data()[0] >= classifier_output[0].data()[1] ? tunnel_shape::CIRCLE : tunnel_shape::RECTANGLE;

}


bool tunnel_planner::detect_circle(const cv::Mat &cross_section_img, const Matrix3d& w_R_cs, cross_section &cross_section_result, int& outlier_cnt){


    circleShape circle_result;
    hough_circle_detector_->set_image(cross_section_img);

    if(hough_circle_detector_->detect_circle(circle_result, hough_circle_threshold_, outlier_cnt)){
        cross_section_result.cross_section_shape_ = CIRCLE;

        Vector2i img_center(cross_section_img.rows/2, cross_section_img.cols/2);
        Vector3d center_diff = Vector3d::Zero();
        center_diff.head(2) = (circle_result.get_center() - img_center.cast<double>()) * MAP_RES;
        cross_section_result.center_ += w_R_cs * center_diff;
        cross_section_result.cross_section_data_.resize(1);
        cross_section_result.cross_section_data_[0] = circle_result.get_r() * MAP_RES;

        return true;
    }

    return false;
}

bool tunnel_planner::detect_rectangle(const cv::Mat &cross_section_img, const Matrix3d& w_R_cs, cross_section &cross_section_result, int& outlier_cnt){
    

    hough_rectangle_detector_->set_image(cross_section_img);

    rectangleShape detected_rectangle;

    bool detect_rect = false;

   
    if(hough_rectangle_detector_->detect_rectangle(hough_rectangle_threshold_, 10.0, 8.0, 8.0, 5.0, 100000.0, detected_rectangle, outlier_cnt)){
        cross_section_result.cross_section_shape_ = RECTANGLE;

        Vector2i img_center(cross_section_img.rows/2, cross_section_img.cols/2);
        Vector3d center_diff = Vector3d::Zero();
        center_diff.head(2) = (detected_rectangle.get_center() - img_center.cast<double>()) * MAP_RES;
        cross_section_result.center_ += w_R_cs * center_diff;
        
        cross_section_result.cross_section_data_.resize(3);

        double angle = detected_rectangle.get_angle();

       
        Vector2d rect_edge = detected_rectangle.get_edge() * MAP_RES;

        // LUF coord, data in h, w, la-x-axis angle 
        if(abs(angle) < M_PI_4){
            cross_section_result.cross_section_data_[0] = rect_edge(1);
            cross_section_result.cross_section_data_[1] = rect_edge(0);
            cross_section_result.cross_section_data_[2] = angle;
        }
        else{
            cross_section_result.cross_section_data_[0] = rect_edge(0);
            cross_section_result.cross_section_data_[1] = rect_edge(1);
            cross_section_result.cross_section_data_[2] = angle > 0.0 ? M_PI_2 - angle : M_PI_2 + angle;
        }

        
        detect_rect = true;
    }

    return detect_rect;
}

double tunnel_planner::find_max_edf_in_plane(Vector3d &pt_in_plane, const Vector3d &plane_dir, double step, const double max_res, const double max_edf_value)
{

    const int max_iter = 100;

    Vector3d grad(0.0, 0.0, 0.0);

    double next_dist = 0.0;
    double cur_dist = edf_map_generator_ptr_->get_dist_grad(pt_in_plane, grad);

    if (grad.squaredNorm() == 0.0)
        return cur_dist;

    Vector3d ascend_dir = (grad - grad.dot(plane_dir) * plane_dir).normalized();

    double ori_step = step;


    double iter = 0;
    while (iter < max_iter && step >= max_res && cur_dist > max_edf_value && !isnan(ascend_dir.squaredNorm()))
    {
        next_dist = edf_map_generator_ptr_->get_dist_grad(-ascend_dir * step + pt_in_plane, grad);
        while (next_dist > cur_dist)
        {   
            step *= 0.5;

            next_dist = edf_map_generator_ptr_->get_dist_grad(-ascend_dir * step + pt_in_plane, grad);
            if (step < max_res){
                step = 0.0;
                next_dist = cur_dist;
                break;
            }
        }

        pt_in_plane = -ascend_dir * step + pt_in_plane;
        cur_dist = next_dist;
        ascend_dir = (grad - grad.dot(plane_dir) * plane_dir).normalized();
        iter++;

    }


    step = ori_step;
    iter = 0;
    while (iter < max_iter && step >= max_res && !isnan(ascend_dir.squaredNorm()))
    {
        next_dist = edf_map_generator_ptr_->get_dist_grad(ascend_dir * step + pt_in_plane, grad);
        while (next_dist <= cur_dist || next_dist > max_edf_value)
        {   
            step *= 0.5;

            next_dist = edf_map_generator_ptr_->get_dist_grad(ascend_dir * step + pt_in_plane, grad);
            if (step < max_res){
                step = 0.0;
                next_dist = cur_dist;
                break;
            }
        }

        pt_in_plane = ascend_dir * step + pt_in_plane;
        cur_dist = next_dist;
        ascend_dir = (grad - grad.dot(plane_dir) * plane_dir).normalized();
        iter++;

    }

    return cur_dist;
}

Vector3d tunnel_planner::find_tunnel_dir(const Vector3d &max_edf_pt, const double max_edf_value, const Vector3d &init_tunnel_dir)
{

    vector<Vector3d> min_edf_pt_on_sphere;

    while(min_edf_pt_on_sphere.size() < 32){
        double longitude_rand = fRand(0.0, M_PI_2);
        double latitude_rand = fRand(0.0, M_PI_2);

        for (int i = 0; i < 8; i++)
        {
            double longitude = (i % 4 - 2) * M_PI_2 + longitude_rand;
            double latitude = (i / 4 - 1) * M_PI_2 + latitude_rand;
            Vector3d pt_on_sphere = max_edf_value * Vector3d(cos(latitude) * cos(longitude), cos(latitude) * sin(longitude), sin(latitude)) + max_edf_pt;
            if(edf_map_generator_ptr_->get_type(pt_on_sphere) == edf_voxel::UNKNOWN){
                continue;
            }
            sphere_descend(pt_on_sphere, max_edf_pt, max_edf_value, edf_map_generator_ptr_->get_edf_map_ptr()->map_res, grad_max_res_);

            if(edf_map_generator_ptr_->get_type(pt_on_sphere) == edf_voxel::UNKNOWN){
                continue;
            }
            // center at the sphere center
            pt_on_sphere -= max_edf_pt;
            min_edf_pt_on_sphere.emplace_back(pt_on_sphere);
        }
    }

    Vector3d plane_dir = plane_dir_fitting(min_edf_pt_on_sphere);

    return plane_dir.dot(init_tunnel_dir) > 0.0 ? plane_dir : -plane_dir;
}

double tunnel_planner::sphere_descend(Vector3d &pt_on_sphere, const Vector3d &sphere_center, const double radius, double step, const double max_res)
{

    Vector3d grad(0.0, 0.0, 0.0);

    double cur_dist = edf_map_generator_ptr_->get_dist_grad(pt_on_sphere, grad);
    double next_dist = cur_dist;

    Vector3d plane_dir = (pt_on_sphere - sphere_center).normalized();
    Vector3d descend_dir = -(grad - grad.dot(plane_dir) * plane_dir).normalized();

    while (step > max_res && next_dist > max_res && !isnan(descend_dir.squaredNorm()))
    {

        while ((next_dist = edf_map_generator_ptr_->get_dist_grad((descend_dir * step + pt_on_sphere - sphere_center).normalized() * radius + sphere_center, grad)) >= cur_dist)
        {
            if (step < max_res)
                break;
            step *= 0.5;
        }

        pt_on_sphere = (descend_dir * step + pt_on_sphere - sphere_center).normalized() * radius + sphere_center;
        cur_dist = next_dist;
        descend_dir = -(grad - grad.dot(plane_dir) * plane_dir).normalized();
    }

    return cur_dist;
}

Vector3d tunnel_planner::plane_dir_fitting(vector<Vector3d> &pts)
{

    MatrixXd pts_mat;
    pts_mat.resize(3, pts.size());

    for (unsigned int i = 0; i < pts.size(); i++)
    {
        pts_mat.col(i) = pts[i];
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd = pts_mat.jacobiSvd(ComputeFullU | ComputeThinV);
    return svd.matrixU().col(2).normalized();
}

Matrix3d tunnel_planner::cal_w_R_plane(const Vector3d &dir)
{

    Vector3d normal_dir = dir.normalized();

    Vector3d plane_x_dir = Vector3d::UnitX();
    Vector3d plane_y_dir = Vector3d::UnitY();

    // LUF coord
    // if normal ~ up/down, X?F coord

    if(abs(normal_dir.z()) > VERT_SECTION_COS_THRESHOLD){
        // approx z dir
        plane_y_dir =  (normal_dir.cross(plane_x_dir)).normalized();
        plane_x_dir = (plane_y_dir.cross(normal_dir)).normalized();
    }
    else{
        plane_x_dir = (Vector3d::UnitZ().cross(normal_dir)).normalized();

        plane_y_dir = (normal_dir.cross(plane_x_dir)).normalized();
    }

    Matrix3d w_R_plane;

    w_R_plane.col(0) = plane_x_dir;
    w_R_plane.col(1) = plane_y_dir;
    w_R_plane.col(2) = normal_dir;

    return w_R_plane;
}

bool tunnel_planner::construct_cross_section(const Vector3d &pt, const Vector3d &dir, Matrix3d& w_R_plane, cv::Mat& cross_section_mat)
{

    w_R_plane = cal_w_R_plane(dir);
    Vector3d plane_x_dir = w_R_plane.col(0);
    Vector3d plane_y_dir = w_R_plane.col(1);
    

    cross_section_mat = cv::Mat::zeros(2*tunnel_dim_pixel_+1, 2*tunnel_dim_pixel_+1, CV_8U);
    
    int free_cnt = 0;

    int dir_free[4] = {1,1,1,1};
    int min_check = int(floor(0.5 * tunnel_dim_pixel_));
    int max_check = int(ceil(1.5 * tunnel_dim_pixel_));
    for(int x = 0; x < 2*tunnel_dim_pixel_+1; x++){
        for(int y = 0; y < 2*tunnel_dim_pixel_+1; y++){
            Vector3d pos_check = pt + ((x - tunnel_dim_pixel_) * plane_x_dir + (y - tunnel_dim_pixel_) * plane_y_dir) * edf_map_generator_ptr_->get_edf_map_ptr()->map_res;
            int voxel_type = edf_map_generator_ptr_-> get_type(pos_check);
            if(voxel_type == edf_voxel::OCC){
                cross_section_mat.at<uchar>(x,y) = 255;

                if(y >= min_check && y < max_check){
                    if(x >= min_check && x < tunnel_dim_pixel_){
                        dir_free[0] = 0;
                    }
                    if(x <= max_check && x > tunnel_dim_pixel_){
                        dir_free[2] = 0;
                    }
                }

                if(x >= min_check && x < max_check){
                    if(y >= min_check && y < tunnel_dim_pixel_){
                        dir_free[1] = 0;
                    }
                    if(y <= max_check && y > tunnel_dim_pixel_){
                        dir_free[3] = 0;
                    }
                }
            }
            else if(voxel_type == edf_voxel::FREE){
                free_cnt++;
            }
            else if(voxel_type == edf_voxel::UNKNOWN){

                if(y >= min_check && y < max_check){
                    if(x >= min_check && x < tunnel_dim_pixel_){
                        dir_free[0] = 0;
                    }
                    if(x <= max_check && x > tunnel_dim_pixel_){
                        dir_free[2] = 0;
                    }
                }

                if(x >= min_check && x < max_check){
                    if(y >= min_check && y < tunnel_dim_pixel_){
                        dir_free[1] = 0;
                    }
                    if(y <= max_check && y > tunnel_dim_pixel_){
                        dir_free[3] = 0;
                    }
                }
               
            }             
        }

    }

    int free_dir_cnt = dir_free[0] + dir_free[1] + dir_free[2] + dir_free[3];

    
    bool return_value = free_dir_cnt < 1; // || (!enough_free_space);
    return return_value;
}

double tunnel_planner::calc_des_yaw(const double& last_yaw, const double& cur_yaw, const double& max_yaw_change){
    double round_last = last_yaw;

    double rectified_cur_yaw = cur_yaw;

    while (round_last < -M_PI) {
        round_last += 2 * M_PI;
    }
    while (round_last > M_PI) {
        round_last -= 2 * M_PI;
    }

    while (rectified_cur_yaw < -M_PI) {
        rectified_cur_yaw += 2 * M_PI;
    }
    while (rectified_cur_yaw > M_PI) {
        rectified_cur_yaw -= 2 * M_PI;
    }

    double diff = rectified_cur_yaw - round_last;
    if (fabs(diff) <= M_PI) {
        rectified_cur_yaw = last_yaw + diff;
    } else if (diff > M_PI) {
        rectified_cur_yaw = last_yaw + diff - 2 * M_PI;
    } else if (diff < -M_PI) {
        rectified_cur_yaw = last_yaw + diff + 2 * M_PI;
    }

    double diff_tunnel_last = rectified_cur_yaw - round_last;
    if (fabs(diff) <= M_PI) {
        rectified_cur_yaw = last_yaw + diff;
    } else if (diff > M_PI) {
        rectified_cur_yaw = last_yaw + diff - 2 * M_PI;
    } else if (diff < -M_PI) {
        rectified_cur_yaw = last_yaw + diff + 2 * M_PI;
    }


    double rectified_diff = rectified_cur_yaw - last_yaw;
    double abs_max_yaw_change = fabs(max_yaw_change);
    if(rectified_diff > abs_max_yaw_change){
        rectified_cur_yaw = last_yaw + abs_max_yaw_change;
    }
    else if(rectified_diff < -abs_max_yaw_change){
        rectified_cur_yaw = last_yaw - abs_max_yaw_change;

    }

    return rectified_cur_yaw;
}

double tunnel_planner::calc_des_yaw(const double& last_yaw, const double& tunnel_dir_yaw, const double& cur_yaw, const double& max_yaw_change, const double& max_yaw_center_line_dir_diff){
    double round_last = last_yaw;
    double round_tunnel_yaw = tunnel_dir_yaw;

    double rectified_cur_yaw = cur_yaw;

    while (round_last < -M_PI) {
        round_last += 2 * M_PI;
    }
    while (round_last > M_PI) {
        round_last -= 2 * M_PI;
    }

    while (round_tunnel_yaw < -M_PI) {
        round_tunnel_yaw += 2 * M_PI;
    }
    while (round_tunnel_yaw > M_PI) {
        round_tunnel_yaw -= 2 * M_PI;
    }

    while (rectified_cur_yaw < -M_PI) {
        rectified_cur_yaw += 2 * M_PI;
    }
    while (rectified_cur_yaw > M_PI) {
        rectified_cur_yaw -= 2 * M_PI;
    }

    double diff = rectified_cur_yaw - round_last;
    if (fabs(diff) <= M_PI) {
        rectified_cur_yaw = last_yaw + diff;
    } else if (diff > M_PI) {
        rectified_cur_yaw = last_yaw + diff - 2 * M_PI;
    } else if (diff < -M_PI) {
        rectified_cur_yaw = last_yaw + diff + 2 * M_PI;
    }

    double diff_tunnel_last = rectified_cur_yaw - round_last;
    if (fabs(diff) <= M_PI) {
        rectified_cur_yaw = last_yaw + diff;
    } else if (diff > M_PI) {
        rectified_cur_yaw = last_yaw + diff - 2 * M_PI;
    } else if (diff < -M_PI) {
        rectified_cur_yaw = last_yaw + diff + 2 * M_PI;
    }

    double diff_tunnel = round_tunnel_yaw - round_last;
    if (fabs(diff_tunnel) <= M_PI) {
        round_tunnel_yaw = last_yaw + diff_tunnel;
    } else if (diff_tunnel > M_PI) {
        round_tunnel_yaw = last_yaw + diff_tunnel - 2 * M_PI;
    } else if (diff_tunnel < -M_PI) {
        round_tunnel_yaw = last_yaw + diff_tunnel + 2 * M_PI;
    }

    double abs_max_yaw_change = fabs(max_yaw_change);
    rectified_cur_yaw = max(last_yaw - abs_max_yaw_change, min(rectified_cur_yaw, last_yaw + abs_max_yaw_change));

    const double abs_max_yaw_center_line_dir_diff = abs(max_yaw_center_line_dir_diff);

    rectified_cur_yaw = max(round_tunnel_yaw - abs_max_yaw_center_line_dir_diff, min(rectified_cur_yaw, round_tunnel_yaw + abs_max_yaw_center_line_dir_diff));

    return rectified_cur_yaw;
}


void tunnel_planner::pub_corridor(vector<tuple<Vector3d, double, Vector3d>> &plan_corridor)
{

    if (plan_corridor.empty())
        return;

    visualization_msgs::MarkerArray mk_array;
    mk_array.markers.resize(plan_corridor.size());
    mk_array.markers[0].action = visualization_msgs::Marker::DELETEALL;
    corridor_pub_.publish(mk_array);

    geometry_msgs::PoseArray pose_array;
    pose_array.header = latest_odom_.header;
    pose_array.header.frame_id = "world";
    pose_array.poses.resize(plan_corridor.size());

    nav_msgs::Path path;
    path.header = latest_odom_.header;
    path.header.frame_id = "world";
    path.poses.resize(plan_corridor.size());

    for (unsigned int i = 0; i < plan_corridor.size(); i++)
    {

        Vector3d pos = get<0>(plan_corridor[i]);
        double radius = get<1>(plan_corridor[i]);
        Vector3d dir = get<2>(plan_corridor[i]);

        mk_array.markers[i].header = latest_odom_.header;
        mk_array.markers[i].header.frame_id = "world";
        mk_array.markers[i].action = visualization_msgs::Marker::ADD;
        mk_array.markers[i].type = visualization_msgs::Marker::SPHERE;
        mk_array.markers[i].id = i;

        mk_array.markers[i].pose.orientation.x = 0.0;
        mk_array.markers[i].pose.orientation.y = 0.0;
        mk_array.markers[i].pose.orientation.z = 0.0;
        mk_array.markers[i].pose.orientation.w = 1.0;

        mk_array.markers[i].color.r = 0.0;
        mk_array.markers[i].color.g = 0.0;
        mk_array.markers[i].color.b = 0.0;
        mk_array.markers[i].color.a = 0.2;

        mk_array.markers[i].pose.position.x = pos.x();
        mk_array.markers[i].pose.position.y = pos.y();
        mk_array.markers[i].pose.position.z = pos.z();

        mk_array.markers[i].scale.x = 2 * radius;
        mk_array.markers[i].scale.y = 2 * radius;
        mk_array.markers[i].scale.z = 2 * radius;

        Matrix3d R_dir;
        R_dir.col(0) = dir;
        R_dir.col(2) = dir.cross(Vector3d(0, 1, 0));
        R_dir.col(1) = R_dir.col(2).cross(dir);

        Quaterniond Q_dir(R_dir);

        pose_array.poses[i].position.x = pos.x();
        pose_array.poses[i].position.y = pos.y();
        pose_array.poses[i].position.z = pos.z();

        pose_array.poses[i].orientation.w = Q_dir.w();
        pose_array.poses[i].orientation.x = Q_dir.x();
        pose_array.poses[i].orientation.y = Q_dir.y();
        pose_array.poses[i].orientation.z = Q_dir.z();

        path.poses[i].pose.position.x = pos.x();
        path.poses[i].pose.position.y = pos.y();
        path.poses[i].pose.position.z = pos.z();

        path.poses[i].pose.orientation.w = 1.0;
        path.poses[i].pose.orientation.x = 0.0;
        path.poses[i].pose.orientation.y = 0.0;
        path.poses[i].pose.orientation.z = 0.0;
    }

    corridor_pub_.publish(mk_array);
    corridor_center_pub_.publish(pose_array);
    corridor_center_path_pub_.publish(path);
}

void tunnel_planner::pub_init_corridor()
{

    if (plan_corridor_->empty())
        return;

    geometry_msgs::PoseArray pose_array;
    pose_array.header = latest_odom_.header;
    pose_array.header.frame_id = "world";
    pose_array.poses.resize(plan_corridor_->size());


    for (unsigned int i = 0; i < plan_corridor_->size(); i++)
    {

        Vector3d pos = plan_corridor_->at(i).center_;
        Vector3d dir = plan_corridor_->at(i).w_R_cs.col(2);


        Matrix3d R_dir;
        R_dir.col(0) = plan_corridor_->at(i).w_R_cs.col(2);
        R_dir.col(2) = plan_corridor_->at(i).w_R_cs.col(1);
        R_dir.col(1) = plan_corridor_->at(i).w_R_cs.col(0);

        Quaterniond Q_dir(R_dir);

        pose_array.poses[i].position.x = pos.x();
        pose_array.poses[i].position.y = pos.y();
        pose_array.poses[i].position.z = pos.z();

        pose_array.poses[i].orientation.w = Q_dir.w();
        pose_array.poses[i].orientation.x = Q_dir.x();
        pose_array.poses[i].orientation.y = Q_dir.y();
        pose_array.poses[i].orientation.z = Q_dir.z();

    }

    corridor_center_init_pub_.publish(pose_array);
}

void tunnel_planner::pub_corridor()
{

    if (plan_corridor_->empty())
        return;

    visualization_msgs::MarkerArray mk_array;
    mk_array.markers.resize(plan_corridor_->size());
    mk_array.markers[0].action = visualization_msgs::Marker::DELETEALL;
    corridor_pub_.publish(mk_array);

    geometry_msgs::PoseArray pose_array;
    pose_array.header = latest_odom_.header;
    pose_array.header.frame_id = "world";
    pose_array.poses.resize(plan_corridor_->size());

    nav_msgs::Path path;
    path.header = latest_odom_.header;
    path.header.frame_id = "world";
    path.poses.resize(plan_corridor_->size());

    for (unsigned int i = 0; i < plan_corridor_->size(); i++)
    {

        Vector3d pos = plan_corridor_->at(i).center_;
        Vector3d dir = plan_corridor_->at(i).w_R_cs.col(2);

        double cs_length = 0.0;
        if(i == plan_corridor_->size() - 1){
            cs_length = 0.1;
            // cs_length = MAX_RAYCAST_LENGTH;
        }
        else{
            cs_length = (plan_corridor_->at(i+1).center_ - pos).norm();
        }
        Vector3d cs_marker_center_pos = pos + 0.5*cs_length*dir;

        if(plan_corridor_->at(i).cross_section_shape_ == RECTANGLE){
            mk_array.markers[i].type = visualization_msgs::Marker::CUBE;
            double height = plan_corridor_->at(i).cross_section_data_[0];
            double width = plan_corridor_->at(i).cross_section_data_[1];
            double angle = plan_corridor_->at(i).cross_section_data_[2];

            Matrix3d rect_angle_R;
            rect_angle_R << cos(angle), -sin(angle), 0.0,
                            sin(angle),  cos(angle), 0.0,
                                   0.0,         0.0, 1.0;
            Quaterniond Q(plan_corridor_->at(i).w_R_cs * rect_angle_R);
            Q.normalize();
            mk_array.markers[i].pose.orientation.x = Q.x();
            mk_array.markers[i].pose.orientation.y = Q.y();
            mk_array.markers[i].pose.orientation.z = Q.z();
            mk_array.markers[i].pose.orientation.w = Q.w();

            mk_array.markers[i].scale.x = width;
            mk_array.markers[i].scale.y = height;
            mk_array.markers[i].scale.z = cs_length;

        }
        else{
            mk_array.markers[i].type = visualization_msgs::Marker::CYLINDER;

            double radius = 0.1;
            if(plan_corridor_->at(i).cross_section_shape_ != OUTSIDE &&  plan_corridor_->at(i).cross_section_shape_ != BEFORE){
               radius =  plan_corridor_->at(i).cross_section_data_[0];
            }
            mk_array.markers[i].scale.x = 2 * radius;
            mk_array.markers[i].scale.y = 2 * radius;
            mk_array.markers[i].scale.z = cs_length;

            Quaterniond Q(plan_corridor_->at(i).w_R_cs);
            Q.normalize();
            mk_array.markers[i].pose.orientation.x = Q.x();
            mk_array.markers[i].pose.orientation.y = Q.y();
            mk_array.markers[i].pose.orientation.z = Q.z();
            mk_array.markers[i].pose.orientation.w = Q.w();
            
        }


        mk_array.markers[i].header = latest_odom_.header;
        mk_array.markers[i].header.frame_id = "world";
        mk_array.markers[i].action = visualization_msgs::Marker::ADD;
        mk_array.markers[i].id = i;        

        mk_array.markers[i].color.r = 0.0;
        mk_array.markers[i].color.g = 0.0;
        mk_array.markers[i].color.b = 0.0;
        mk_array.markers[i].color.a = 0.2;

        mk_array.markers[i].pose.position.x = cs_marker_center_pos.x();
        mk_array.markers[i].pose.position.y = cs_marker_center_pos.y();
        mk_array.markers[i].pose.position.z = cs_marker_center_pos.z();


        Matrix3d R_dir;
        R_dir.col(0) = plan_corridor_->at(i).w_R_cs.col(2);
        R_dir.col(2) = plan_corridor_->at(i).w_R_cs.col(1);
        R_dir.col(1) = plan_corridor_->at(i).w_R_cs.col(0);

        Quaterniond Q_dir(R_dir);

        pose_array.poses[i].position.x = pos.x();
        pose_array.poses[i].position.y = pos.y();
        pose_array.poses[i].position.z = pos.z();

        pose_array.poses[i].orientation.w = Q_dir.w();
        pose_array.poses[i].orientation.x = Q_dir.x();
        pose_array.poses[i].orientation.y = Q_dir.y();
        pose_array.poses[i].orientation.z = Q_dir.z();

        path.poses[i].pose.position.x = pos.x();
        path.poses[i].pose.position.y = pos.y();
        path.poses[i].pose.position.z = pos.z();

        path.poses[i].pose.orientation.w = 1.0;
        path.poses[i].pose.orientation.x = 0.0;
        path.poses[i].pose.orientation.y = 0.0;
        path.poses[i].pose.orientation.z = 0.0;
    }

    corridor_pub_.publish(mk_array);
    corridor_center_pub_.publish(pose_array);
    corridor_center_path_pub_.publish(path);
}

void tunnel_planner::pub_past_corridor()
{

    if (past_corridor_->empty())
        return;

    visualization_msgs::MarkerArray mk_array;
    mk_array.markers.resize(past_corridor_->size());
    mk_array.markers[0].action = visualization_msgs::Marker::DELETEALL;
    corridor_pub_.publish(mk_array);

    geometry_msgs::PoseArray pose_array;
    pose_array.header = latest_odom_.header;
    pose_array.header.frame_id = "world";
    pose_array.poses.resize(past_corridor_->size());

    nav_msgs::Path path;
    path.header = latest_odom_.header;
    path.header.frame_id = "world";
    path.poses.resize(past_corridor_->size());

    for (unsigned int i = 0; i < past_corridor_->size() ; i++)
    {

        Vector3d pos = past_corridor_->at(i).center_;
        Vector3d dir = past_corridor_->at(i).w_R_cs.col(2);

        double cs_length = 0.0;

        if(i == past_corridor_->size() - 1){
            cs_length = (plan_corridor_->front().center_ - pos).norm();
        }
        else{
            cs_length = (past_corridor_->at(i+1).center_ - pos).norm();
        }

        Vector3d cs_marker_center_pos = pos + 0.5*cs_length*dir;

        if(past_corridor_->at(i).cross_section_shape_ == RECTANGLE){
            mk_array.markers[i].type = visualization_msgs::Marker::CUBE;
            double height = past_corridor_->at(i).cross_section_data_[0];
            double width = past_corridor_->at(i).cross_section_data_[1];
            double angle = past_corridor_->at(i).cross_section_data_[2];

            Matrix3d rect_angle_R;
            rect_angle_R << cos(angle), -sin(angle), 0.0,
                            sin(angle),  cos(angle), 0.0,
                                   0.0,         0.0, 1.0;
            Quaterniond Q(past_corridor_->at(i).w_R_cs * rect_angle_R);
            Q.normalize();
            mk_array.markers[i].pose.orientation.x = Q.x();
            mk_array.markers[i].pose.orientation.y = Q.y();
            mk_array.markers[i].pose.orientation.z = Q.z();
            mk_array.markers[i].pose.orientation.w = Q.w();

            mk_array.markers[i].scale.x = width;
            mk_array.markers[i].scale.y = height;
            mk_array.markers[i].scale.z = cs_length;

        }
        else{
            mk_array.markers[i].type = visualization_msgs::Marker::CYLINDER;

            double radius = 0.1;
            if(past_corridor_->at(i).cross_section_shape_ != OUTSIDE &&  past_corridor_->at(i).cross_section_shape_ != BEFORE){
               radius =  past_corridor_->at(i).cross_section_data_[0];
            }
            mk_array.markers[i].scale.x = 2 * radius;
            mk_array.markers[i].scale.y = 2 * radius;
            mk_array.markers[i].scale.z = cs_length;

            Quaterniond Q(past_corridor_->at(i).w_R_cs);
            Q.normalize();
            mk_array.markers[i].pose.orientation.x = Q.x();
            mk_array.markers[i].pose.orientation.y = Q.y();
            mk_array.markers[i].pose.orientation.z = Q.z();
            mk_array.markers[i].pose.orientation.w = Q.w();
            
        }


        mk_array.markers[i].header = latest_odom_.header;
        mk_array.markers[i].header.frame_id = "world";
        mk_array.markers[i].action = visualization_msgs::Marker::ADD;
        mk_array.markers[i].id = i;        

        mk_array.markers[i].color.r = 0.6;
        mk_array.markers[i].color.g = 0.0;
        mk_array.markers[i].color.b = 0.2;
        mk_array.markers[i].color.a = 0.2;

        mk_array.markers[i].pose.position.x = cs_marker_center_pos.x();
        mk_array.markers[i].pose.position.y = cs_marker_center_pos.y();
        mk_array.markers[i].pose.position.z = cs_marker_center_pos.z();


        Matrix3d R_dir;
        R_dir.col(0) = past_corridor_->at(i).w_R_cs.col(2);
        R_dir.col(2) = past_corridor_->at(i).w_R_cs.col(1);
        R_dir.col(1) = past_corridor_->at(i).w_R_cs.col(0);

        Quaterniond Q_dir(R_dir);

        pose_array.poses[i].position.x = pos.x();
        pose_array.poses[i].position.y = pos.y();
        pose_array.poses[i].position.z = pos.z();

        pose_array.poses[i].orientation.w = Q_dir.w();
        pose_array.poses[i].orientation.x = Q_dir.x();
        pose_array.poses[i].orientation.y = Q_dir.y();
        pose_array.poses[i].orientation.z = Q_dir.z();

        path.poses[i].pose.position.x = pos.x();
        path.poses[i].pose.position.y = pos.y();
        path.poses[i].pose.position.z = pos.z();

        path.poses[i].pose.orientation.w = 1.0;
        path.poses[i].pose.orientation.x = 0.0;
        path.poses[i].pose.orientation.y = 0.0;
        path.poses[i].pose.orientation.z = 0.0;
    }

    past_corridor_pub_.publish(mk_array);
    past_corridor_center_pub_.publish(pose_array);
    past_corridor_center_path_pub_.publish(path);
}

void tunnel_planner::pub_traj_vis(NonUniformBspline &bspline)
{
    if (bspline.getControlPoint().size() == 0)
        return;

    nav_msgs::Path traj_path;
    traj_path.header = latest_odom_.header;
    traj_path.header.frame_id = "world";
    geometry_msgs::PoseStamped traj_pt_pose;
    traj_pt_pose.header.frame_id = "world";
    traj_pt_pose.header.seq = 0;
    traj_pt_pose.pose.orientation.w = 1.0;
    traj_pt_pose.pose.orientation.x = 0.0;
    traj_pt_pose.pose.orientation.y = 0.0;
    traj_pt_pose.pose.orientation.z = 0.0;

    //   vector<Eigen::Vector3d> traj_pts;
    double tm, tmp;
    bspline.getTimeSpan(tm, tmp);

    for (double t = tm; t <= tmp; t += 0.01)
    {
        Eigen::Vector3d pt = bspline.evaluateDeBoor(t);

        traj_pt_pose.header.stamp = ros::Time(t);
        traj_pt_pose.pose.position.x = pt.x();
        traj_pt_pose.pose.position.y = pt.y();
        traj_pt_pose.pose.position.z = pt.z();
        traj_pt_pose.header.seq++;

        traj_path.poses.push_back(traj_pt_pose);

    }

    traj_path_vis_pub_.publish(traj_path);

}

void tunnel_planner::pub_traj_init_vis(NonUniformBspline &bspline)
{
    if (bspline.getControlPoint().size() == 0)
        return;

    nav_msgs::Path traj_path;
    traj_path.header = latest_odom_.header;
    traj_path.header.frame_id = "world";
    geometry_msgs::PoseStamped traj_pt_pose;
    traj_pt_pose.header.frame_id = "world";
    traj_pt_pose.header.seq = 0;
    traj_pt_pose.pose.orientation.w = 1.0;
    traj_pt_pose.pose.orientation.x = 0.0;
    traj_pt_pose.pose.orientation.y = 0.0;
    traj_pt_pose.pose.orientation.z = 0.0;

    double tm, tmp;
    bspline.getTimeSpan(tm, tmp);

    for (double t = tm; t <= tmp; t += 0.01)
    {
        Eigen::Vector3d pt = bspline.evaluateDeBoor(t);

        traj_pt_pose.header.stamp = ros::Time(t);
        traj_pt_pose.pose.position.x = pt.x();
        traj_pt_pose.pose.position.y = pt.y();
        traj_pt_pose.pose.position.z = pt.z();
        traj_pt_pose.header.seq++;

        traj_path.poses.push_back(traj_pt_pose);

    }

    traj_path_init_vis_pub_.publish(traj_path);
}

}
