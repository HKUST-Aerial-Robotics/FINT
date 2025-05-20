#include "optical_flow_estimator/optical_flow_estimator.hpp"

using namespace std;
using namespace Eigen;

namespace tunnel_planner{

optical_flow_estimator::optical_flow_estimator(ros::NodeHandle& n, const vector<camera_module_info>& camera_parameter_vec, shared_ptr<edf_map_generator> edf_map_generator_ptr, const shared_ptr<vector<cross_section>>& forward_corridor, const shared_ptr<vector<cross_section>>& backward_corridor, const double max_raycast_length, const int cal_res): 
    res_(cal_res), max_raycast_length_(max_raycast_length), num_cam_(camera_parameter_vec.size()), edf_map_generator_ptr_(edf_map_generator_ptr), forward_cross_sections_ptr_(forward_corridor), backward_cross_sections_ptr_(backward_corridor), nh_(n){


    for(auto& cam_param : camera_parameter_vec){
        cam_data_vec_.emplace_back(cam_param, cal_res);
    }

    auto map_ptr = edf_map_generator_ptr_->get_edf_map_ptr();
    rc_.setParams(map_ptr->map_res, Vector3d(map_ptr->xmin, map_ptr->ymin, map_ptr->zmin));

    ROS_WARN("optical flow estimator param");
    cout<<"camera_data_vec size: "<<cam_data_vec_.size()<<endl;
    cout<<"cal_row_idx_:\n"<<cam_data_vec_[0].cal_row_idx_.transpose()<<endl;
    cout<<"cal_col_idx_:\n"<<cam_data_vec_[0].cal_col_idx_.transpose()<<endl;
    // cout<<"inv proj dir:\n"<<cam_data_vec_[0].inv_proj_dir_mat_<<endl;

    ray_publisher_ = nh_.advertise<visualization_msgs::MarkerArray>("predict_raycast", 10);

}

void optical_flow_estimator::set_forward_cross_sections(){
    num_cross_section_ = forward_cross_sections_ptr_->size();

    for(auto& cam_data: cam_data_vec_){
        int size_diff = num_cross_section_ - cam_data.cross_section_dist_vector_.size();
        if(size_diff > 0){
            cam_data.cross_section_dist_vector_.insert(cam_data.cross_section_dist_vector_.end(), size_diff, MatrixXd::Zero(cam_data.cal_rows_, cam_data.cal_cols_));
        }
        // else{
        //     cross_section_dist_vector_.resize(cross_sections.size());
        // }
    }

    cross_section_yaw.resize(num_cross_section_);
    for(int i = 0; i < cross_section_yaw.size(); i++){
        Vector2d cs_hor_dir = forward_cross_sections_ptr_->at(i).w_R_cs.col(2).head(2);
        cross_section_yaw[i] = atan2(cs_hor_dir.y(), cs_hor_dir.x());
    }

    cal_depth();

}

void optical_flow_estimator::cal_depth(){

    ros::Time t0 = ros::Time::now();
    
    double last_cross_section_extension_length = max_raycast_length_;

    const cross_section& last_cross_section = forward_cross_sections_ptr_->back();
    

    auto map_ptr = edf_map_generator_ptr_->get_edf_map_ptr();

    for(unsigned int cross_section_idx = 0; cross_section_idx < forward_cross_sections_ptr_->size(); cross_section_idx++){

        const Vector3d& cs_pos = forward_cross_sections_ptr_->at(cross_section_idx).center_;
        const Vector3d cs_dir = forward_cross_sections_ptr_->at(cross_section_idx).w_R_cs.col(2);
        double yaw = forward_cross_sections_ptr_->at(cross_section_idx).yaw_;
        Matrix3d ori;
        ori <<  cos(yaw), -sin(yaw), 0.0,
                sin(yaw),  cos(yaw), 0.0,
                     0.0,       0.0, 1.0;

        for(unsigned int cam_idx = 0; cam_idx < cam_data_vec_.size(); cam_idx++){
            
            auto& cam_data = cam_data_vec_[cam_idx];
            Matrix3d cam_ori = ori * cam_data.ric_;
            Vector3d cam_pos = ori * cam_data.tic_ + cs_pos;

            // cout<<"cam pos: "<<cam_pos.transpose()<<endl;
            // cout<<"cam ori:\n"<<cam_ori.transpose()<<endl;
            
            for(int row_idx = 0; row_idx < cam_data.cal_rows_; row_idx++){
                for(int col_idx = 0; col_idx < cam_data.cal_cols_; col_idx++){
                    // Vector3d ray_cast_dir = cam_ori * inv_proj_mat * Vector3d(cal_row_idx_(row_idx), cal_col_idx_(col_idx), 1.0);
                    Vector3d ray_cast_dir = cam_ori * cam_data.inv_proj_dir_mat_(row_idx, col_idx);
                    // Vector3d ray_cast_start_pt = 0.2 * ray_cast_dir + cam_pos;
                    Vector3d ray_cast_start_pt = cam_pos;

                    double ray_proj_dir = ray_cast_dir.dot(cs_dir);

                    // ROS_ERROR("new ray");
                    // cout<<"ray_cast_start_pt: "<<ray_cast_start_pt.transpose()<<endl;
                    // cout<<"ray_cast_dir: "<<ray_cast_dir.transpose()<<endl;

                    double intersect_dist = -1.0;
                    if(ray_proj_dir == 0.0){
                        double ray_cast_start_pt_dir = (ray_cast_start_pt - cs_pos).dot(cs_dir);
                        if(ray_cast_start_pt_dir >= 0.0){
                            intersect_dist = intersect_forward_tunnel(ray_cast_start_pt, ray_cast_dir);
                        }
                        else{
                            intersect_dist = intersect_backward_tunnel(ray_cast_start_pt, ray_cast_dir);
                        }

                    }else if(ray_proj_dir > 0.0){
                        intersect_dist = intersect_forward_tunnel(ray_cast_start_pt, ray_cast_dir);
                    }else{
                        intersect_dist = intersect_backward_tunnel(ray_cast_start_pt, ray_cast_dir);
                    }

                    

                                
                    cam_data.cross_section_dist_vector_[cross_section_idx](row_idx, col_idx) = intersect_dist;


                }
            }


        }

    }

    // pub_raycast_result();

}

int optical_flow_estimator::intersect_tunnel(const Vector3d& ray_cast_start_pt, const Vector3d& ray_cast_dir, const cross_section& cross_section, double cross_section_length, bool forward_dir, double& dist){

    int cs_shape = cross_section.cross_section_shape_;
    Vector3d cs_normal = cross_section.w_R_cs.col(2);
    Vector3d cs_center = cross_section.center_;

    Matrix3d cs_R_w = cross_section.w_R_cs.transpose();

    if(!forward_dir){
        cs_center = cs_center + cross_section_length * cs_normal;
        cross_section_length = -cross_section_length;
    }

    // cross section frame LUF or X?F when F ~ U/D
    if(cs_shape == RECTANGLE){

        double half_height = 0.5 * cross_section.cross_section_data_[0];
        double half_width = 0.5 * cross_section.cross_section_data_[1];
        double angle = cross_section.cross_section_data_[2];

        Matrix3d rect_R_cs;
        rect_R_cs <<  cos(angle), sin(angle), 0.0,
                     -sin(angle), cos(angle), 0.0,
                             0.0,        0.0, 1.0;

        Vector3d line_dir_in_rect = rect_R_cs * cs_R_w * ray_cast_dir;
        Vector3d line_start_pt_in_rect = rect_R_cs * cs_R_w * (ray_cast_start_pt - cs_center);

        bool start_pt_in_cross_section_z = forward_dir ? 
            line_start_pt_in_rect.z() >= 0.0 && line_start_pt_in_rect.z() < cross_section_length :
            line_start_pt_in_rect.z() <= 0.0 && line_start_pt_in_rect.z() > cross_section_length;


        bool start_pt_in_cross_section_region = 
            start_pt_in_cross_section_z && 
            line_start_pt_in_rect.x() > -half_width && line_start_pt_in_rect.x() <  half_width &&
            line_start_pt_in_rect.y() > -half_height && line_start_pt_in_rect.y() <  half_height;
        
        if(start_pt_in_cross_section_region){
            
            if(line_dir_in_rect.z() == 0.0){
                // ROS_INFO("parallel to rect plane");
            }
            else{
                dist = (cross_section_length - line_start_pt_in_rect.z()) / line_dir_in_rect.z();
                Vector3d intersect_pt_exit = dist * line_dir_in_rect + line_start_pt_in_rect;

                bool exit_pt_in_rect = 
                    intersect_pt_exit.x() > -half_width && 
                    intersect_pt_exit.x() <  half_width && 
                    intersect_pt_exit.y() > -half_height && 
                    intersect_pt_exit.y() <  half_height;
                
                if(exit_pt_in_rect){
                    return intersection_status::PASS_THROUGH;
                }
                else{
                    // normal intersection
                }
            }
        }
        else{

            if(forward_dir){
                if(line_start_pt_in_rect.z() > 0.0){
                    return intersection_status::REVERSE_DIR;
                }
            }
            else{
                if(line_start_pt_in_rect.z() < 0.0){
                    return intersection_status::REVERSE_DIR;
                }
            }
            

            if(line_dir_in_rect.z() == 0.0){
                ROS_INFO("parallel to rect plane");
                return intersection_status::NO_ENTRY;
            }
            else{
                Vector3d intersect_pt_entry = (0.0 - line_start_pt_in_rect.z()) / line_dir_in_rect.z() * line_dir_in_rect + line_start_pt_in_rect;

                bool entry_pt_in_rect = 
                    intersect_pt_entry.x() > -half_width && 
                    intersect_pt_entry.x() <  half_width && 
                    intersect_pt_entry.y() > -half_height && 
                    intersect_pt_entry.y() <  half_height;

                if(entry_pt_in_rect){
                    
                    // set exit dist
                    dist = (cross_section_length - line_start_pt_in_rect.z()) / line_dir_in_rect.z();
                    Vector3d intersect_pt_exit = dist * line_dir_in_rect + line_start_pt_in_rect;

                    bool exit_pt_in_rect = 
                        intersect_pt_exit.x() > -half_width && 
                        intersect_pt_exit.x() <  half_width && 
                        intersect_pt_exit.y() > -half_height && 
                        intersect_pt_exit.y() <  half_height;
                    
                    if(exit_pt_in_rect){
                        return intersection_status::PASS_THROUGH;
                    }
                    else{
                        // normal intersection
                    }
                    

                }
                else{
                    // ROS_INFO("no entry rect");
                    return intersection_status::NO_ENTRY;
                }
            }
            
        
        }


        // 0 x dir
        if(line_dir_in_rect.x() == 0.0){
            // neg y-axis
            if(line_dir_in_rect.y() < 0.0){
                double y_diff = -half_height - line_start_pt_in_rect.y();                
                dist = y_diff / line_dir_in_rect.y();
            }
            // normal to plane
            else if(line_dir_in_rect.y() == 0.0){
                // dist = -1.0;
                ROS_ERROR("normal to cross section, should not occur");
            }
            // pos y-axis
            else{
                double y_diff = half_height - line_start_pt_in_rect.y();
                dist = y_diff / line_dir_in_rect.y();
            }
        }
        // neg x dir
        else if(line_dir_in_rect.x() < 0.0){
            // neg x, neg y
            if(line_dir_in_rect.y() < 0.0){

                // set neg_half_width_intersection_y dist
                dist = (-half_width - line_start_pt_in_rect.x()) / line_dir_in_rect.x();
                double neg_half_width_intersection_y = dist * line_dir_in_rect.y() + line_start_pt_in_rect.y();

                if(neg_half_width_intersection_y >= -half_height && neg_half_width_intersection_y <= half_height){
                    // neg half width intersection
                }
                else{
                    dist = (-half_height - line_start_pt_in_rect.y()) / line_dir_in_rect.y();
                    double neg_half_height_intersection_x = dist * line_dir_in_rect.x() + line_start_pt_in_rect.x();
                    if(neg_half_height_intersection_x >= -half_width || neg_half_height_intersection_x <= half_width){
                        // neg half height intersection
                    }
                    else{
                        ROS_ERROR("no intersection in (-x, -y) dir, should not occur");
                    }

                }
                
            }
            // neg x-axis
            else if(line_dir_in_rect.y() == 0.0){
                double x_diff = -half_width - line_start_pt_in_rect.x();
                dist = x_diff / line_dir_in_rect.x();
            }
            // neg x, pos y
            else{
                // set neg_half_width_intersection_y dist
                dist = (-half_width - line_start_pt_in_rect.x()) / line_dir_in_rect.x();
                double neg_half_width_intersection_y = dist * line_dir_in_rect.y() + line_start_pt_in_rect.y();

                if(neg_half_width_intersection_y >= -half_height && neg_half_width_intersection_y <= half_height){
                    // neg half width intersection
                }
                else{
                    dist = (half_height - line_start_pt_in_rect.y()) / line_dir_in_rect.y();
                    double pos_half_height_intersection_x = dist * line_dir_in_rect.x() + line_start_pt_in_rect.x();
                    if(pos_half_height_intersection_x >= -half_width || pos_half_height_intersection_x <= half_width){
                        // pos half height intersection
                    }
                    else{
                        ROS_ERROR("no intersection in (-x, +y) dir, should not occur");
                    }

                }

            }

        }
        // pos x dir
        else{
            // pos x, neg y
            if(line_dir_in_rect.y() < 0.0){

                if(line_dir_in_rect.y() < 0.0){

                    // set pos_half_width_intersection_y dist
                    dist = (half_width - line_start_pt_in_rect.x()) / line_dir_in_rect.x();
                    double pos_half_width_intersection_y = dist * line_dir_in_rect.y() + line_start_pt_in_rect.y();

                    if(pos_half_width_intersection_y >= -half_height && pos_half_width_intersection_y <= half_height){
                        // neg half width intersection
                    }
                    else{
                        dist = (-half_height - line_start_pt_in_rect.y()) / line_dir_in_rect.y();
                        double neg_half_height_intersection_x = dist * line_dir_in_rect.x() + line_start_pt_in_rect.x();
                        if(neg_half_height_intersection_x >= -half_width || neg_half_height_intersection_x <= half_width){
                            // neg half height intersection
                        }
                        else{
                            ROS_ERROR("no intersection in (+x, -y) dir, should not occur");
                        }

                    }
                    
                }
                
            }
            // pos x-axis
            else if(line_dir_in_rect.y() == 0.0){
                double x_diff = half_width - line_start_pt_in_rect.x();
                dist = x_diff / line_dir_in_rect.x();
            }
            // pos x, pos y
            else{

                // set pos_half_width_intersection_y dist
                dist = (half_width - line_start_pt_in_rect.x()) / line_dir_in_rect.x();
                double pos_half_width_intersection_y = dist * line_dir_in_rect.y() + line_start_pt_in_rect.y();

                if(pos_half_width_intersection_y >= -half_height && pos_half_width_intersection_y <= half_height){
                    // pos half width intersection
                }
                else{
                    dist = (half_height - line_start_pt_in_rect.y()) / line_dir_in_rect.y();
                    double pos_half_height_intersection_x = dist * line_dir_in_rect.x() + line_start_pt_in_rect.x();
                    if(pos_half_height_intersection_x >= -half_width || pos_half_height_intersection_x <= half_width){
                        // pos half height intersection
                    }
                    else{
                        ROS_ERROR("no intersection in (+x, +y) dir, should not occur");
                    }

                }

            }

        }

        return intersection_status::INTERSECT_TUNNEL;

    }
    else if(cs_shape == CIRCLE || cs_shape == IRREGULAR){

        Vector3d line_dir_in_circle= cs_R_w * ray_cast_dir;
        Vector3d line_start_pt_in_circle = cs_R_w * (ray_cast_start_pt - cs_center);
        double squared_r = cross_section.cross_section_data_[0] * cross_section.cross_section_data_[0];

        bool start_pt_in_cross_section_z = forward_dir ? 
            line_start_pt_in_circle.z() > 0.0 && line_start_pt_in_circle.z() < cross_section_length :
            line_start_pt_in_circle.z() < 0.0 && line_start_pt_in_circle.z() > cross_section_length;

        bool start_pt_in_cross_section_region = 
            start_pt_in_cross_section_z && 
            line_start_pt_in_circle.head(2).squaredNorm() < squared_r;

        if(start_pt_in_cross_section_region){
            // ROS_WARN("pt in cross section region");
            
            if(line_dir_in_circle.z() == 0.0){
                ROS_INFO("parallel to circle plane");
            }
            else{
                dist = (cross_section_length - line_start_pt_in_circle.z()) / line_dir_in_circle.z();
                Vector3d intersect_pt_exit = dist * line_dir_in_circle + line_start_pt_in_circle;

                bool exit_pt_in_circle = intersect_pt_exit.head(2).squaredNorm() < squared_r;
                
                if(exit_pt_in_circle){
                    return intersection_status::PASS_THROUGH;
                }
                else{
                    // normal intersection
                }
            }
        }
        else{

            if(forward_dir){
                if(line_start_pt_in_circle.z() > 0.0){
                    return intersection_status::REVERSE_DIR;
                }
            }
            else{
                if(line_start_pt_in_circle.z() < 0.0){
                    return intersection_status::REVERSE_DIR;
                }
            }
            

            if(line_dir_in_circle.z() == 0.0){
                ROS_INFO("parallel to circle plane");
                return intersection_status::NO_ENTRY;
            }
            else{
                double entry_dist = (0.0 - line_start_pt_in_circle.z()) / line_dir_in_circle.z();
                Vector3d intersect_pt_entry = entry_dist * line_dir_in_circle + line_start_pt_in_circle;
                bool entry_pt_in_circle = intersect_pt_entry.head(2).squaredNorm() <= squared_r;

                if(entry_pt_in_circle){
                    // set exit dist
                    dist = (cross_section_length - line_start_pt_in_circle.z()) / line_dir_in_circle.z();
                    Vector3d intersect_pt_exit = dist * line_dir_in_circle + line_start_pt_in_circle;

                    bool exit_pt_in_circle = intersect_pt_exit.head(2).squaredNorm() <= squared_r;
                    
                    if(exit_pt_in_circle){
                        return intersection_status::PASS_THROUGH;
                    }
                    else{
                        // normal intersection
                    }

                }
                else{
                    // ROS_INFO("no entry circle");
                    return intersection_status::NO_ENTRY;
                }
            }
            
        
        }


        dist = (0.0 - line_start_pt_in_circle.z()) / line_dir_in_circle.z();
        Vector3d intersect_pt_entry = dist * line_dir_in_circle + line_start_pt_in_circle;
        bool entry_pt_in_circle = intersect_pt_entry.head(2).squaredNorm() < squared_r;

        if(entry_pt_in_circle){
            // set exit dist
            dist = (cross_section_length - line_start_pt_in_circle.z()) / line_dir_in_circle.z();
            Vector3d intersect_pt_exit = dist * line_dir_in_circle + line_start_pt_in_circle;

            bool exit_pt_in_circle = intersect_pt_exit.head(2).squaredNorm() < squared_r;
            
            if(exit_pt_in_circle){
                return intersection_status::PASS_THROUGH;
            }
            else{
                // normal intersection
            }

        }
        else{
            // ROS_INFO("no entry circle");
            return intersection_status::NO_ENTRY;
        }

        
        // normal to plane
        if(line_dir_in_circle.head(2).squaredNorm() == 0.0){
            ROS_ERROR("normal to cross section, should not occur");
        }
        else{
            Vector2d line_dir_2d = line_dir_in_circle.head(2).normalized();
            Vector2d line_start_pt_2d = line_start_pt_in_circle.head(2);

            double dist_from_closest_pt_to_cs_center_to_start_pt = line_start_pt_2d.dot(line_dir_2d);

            Vector2d closest_pt_to_cs_center = line_start_pt_in_circle.head(2) - dist_from_closest_pt_to_cs_center_to_start_pt * line_dir_2d;

            double line_to_cs_center_squared_dist = closest_pt_to_cs_center.squaredNorm();

            if(line_to_cs_center_squared_dist > squared_r){
                ROS_ERROR("no intersection for circle, should not occur");
            }
            else if(line_to_cs_center_squared_dist == squared_r){
                ROS_ERROR("tangent to circle, should not occur");
            }
            else{
                // normal intersection
                double half_string_length = sqrt(squared_r - closest_pt_to_cs_center.squaredNorm());
                dist = sqrt((half_string_length * line_dir_2d + closest_pt_to_cs_center - line_start_pt_2d).squaredNorm() / line_dir_in_circle.head(2).squaredNorm());
            }

        }

        return intersection_status::INTERSECT_TUNNEL;

    }

    return intersection_status::INTERSECTION_ERROR;
    
}

double optical_flow_estimator::intersect_forward_tunnel(const Vector3d& ray_cast_start_pt, const Vector3d& ray_cast_dir){

    // ROS_WARN("intersect forward");

    double intersect_dist = -1.0;

    for(int intersect_cs_idx = 0; intersect_cs_idx < forward_cross_sections_ptr_->size(); intersect_cs_idx++){
        // ROS_INFO("check intersect with cross section %d", intersect_cs_idx);
        const int cs_shape = forward_cross_sections_ptr_->at(intersect_cs_idx).cross_section_shape_;
        if(cs_shape == tunnel_shape::BEFORE || cs_shape == tunnel_shape::OUTSIDE || cs_shape == tunnel_shape::FREE_SHAPE){
            continue;
        }

        int cs_intersection_status = intersection_status::INTERSECTION_ERROR;
        if(intersect_cs_idx == forward_cross_sections_ptr_->size() - 1){
            cs_intersection_status = intersect_tunnel(ray_cast_start_pt, ray_cast_dir, forward_cross_sections_ptr_->at(intersect_cs_idx), max_raycast_length_, true, intersect_dist);
        }
        else{
            cs_intersection_status = intersect_tunnel(ray_cast_start_pt, ray_cast_dir, forward_cross_sections_ptr_->at(intersect_cs_idx), (forward_cross_sections_ptr_->at(intersect_cs_idx+1).center_ - forward_cross_sections_ptr_->at(intersect_cs_idx).center_).norm(), true, intersect_dist);

        }

        if(cs_intersection_status == intersection_status::REVERSE_DIR){
            if(intersect_dist < 0.0){
                continue;
            }
            else{
                break;
            }
        }

        if(cs_intersection_status != intersection_status::PASS_THROUGH){
            // ROS_WARN("intersect with cross section %d", intersect_cs_idx);
            break;
        }                  
    }

    return intersect_dist;
}

double optical_flow_estimator::intersect_backward_tunnel(const Vector3d& ray_cast_start_pt, const Vector3d& ray_cast_dir){
    double intersect_dist = -1.0;

    // ROS_WARN("intersect backward");
    // ROS_INFO("backward cross section size: %d", backward_cross_sections_ptr_->size());

    if(backward_cross_sections_ptr_->empty()){
        // ROS_ERROR("backward corridor empty");
        return intersect_dist;
    }

    for(auto it = backward_cross_sections_ptr_->end(), prev_it = prev(it); it != backward_cross_sections_ptr_->begin(); it = prev_it, prev_it = prev(prev_it)){
        // ROS_INFO("check intersect with cross section %d", prev_it - backward_cross_sections_ptr_->begin());
        // cout<<it->center_<<endl;
        // cout<<prev_it->center_<<endl;

        const int cs_shape = it->cross_section_shape_;

        if(cs_shape == tunnel_shape::BEFORE){
            break;
        }
        else if(cs_shape == tunnel_shape::OUTSIDE || cs_shape == tunnel_shape::FREE_SHAPE){
            continue;
        }

        int cs_intersection_status = intersection_status::INTERSECTION_ERROR;
        double cs_length = it == backward_cross_sections_ptr_->end() ? 
            (prev_it->center_ - forward_cross_sections_ptr_->begin()->center_).norm() : 
            (prev_it->center_ - it->center_).norm();

        cs_intersection_status = intersect_tunnel(ray_cast_start_pt, ray_cast_dir, *prev_it, cs_length, false, intersect_dist);
    
        if(cs_intersection_status == intersection_status::REVERSE_DIR){
            if(intersect_dist < 0.0){
                // ROS_ERROR("first intersect reverse");
                continue;
            }
            else{
                // ROS_ERROR("intersect reverse");
                break;
            }
        }

        if(cs_intersection_status != intersection_status::PASS_THROUGH){
            // ROS_WARN("intersect with cross section %d", prev_it - backward_cross_sections_ptr_->begin());
            break;
        }     

    }

    return intersect_dist;
}

double optical_flow_estimator::cal_total_mean_optical_flow(int cross_section_idx, double v, double yaw_dot){
    double total_inv_mean_optical_flow = 0.0;

    double mean_opical_flow = 0.0;

    int cross_section_shape = forward_cross_sections_ptr_->at(cross_section_idx).cross_section_shape_;

    if(cross_section_shape == tunnel_shape::OUTSIDE || cross_section_shape == tunnel_shape::BEFORE){
        return 0.0;
    }

    for(int i = 0; i < cam_data_vec_.size(); i++){
        mean_opical_flow = cal_mean_optical_flow(i, cross_section_idx, v, yaw_dot);
        if(mean_opical_flow > 1e-8){
            total_inv_mean_optical_flow += 1.0 / mean_opical_flow;
        }
        else{
            return 0.0;
        }
        
    }

    return 1.0 / total_inv_mean_optical_flow;

}


double optical_flow_estimator::cal_mean_optical_flow(int cam_id, int cross_section_idx, double v, double yaw_dot){
    
    Vector3d vel = forward_cross_sections_ptr_->at(cross_section_idx).w_R_cs.col(2) * v;
    Vector3d omega(0, 0, yaw_dot);

    // Matrix3d cs_R_w = cross_sections_ptr_->at(cross_section_idx).w_R_cs.transpose();

    return cal_mean_optical_flow(cam_id, cross_section_idx, vel, omega);
}

double optical_flow_estimator::cal_mean_optical_flow(int cam_id, int cross_section_idx, Vector3d& v, Vector3d& omega){

    // double& yaw = cross_section_yaw[cross_section_idx];
    double& yaw = forward_cross_sections_ptr_->at(cross_section_idx).yaw_;
    auto& cam_data = cam_data_vec_[cam_id];
    Matrix3d b_R_w;
    b_R_w << cos(yaw), sin(yaw), 0.0,
            -sin(yaw), cos(yaw), 0.0,
                  0.0,      0.0, 1.0;
    Matrix3d rci_ = cam_data.ric_.transpose();
    
    Vector3d omega_pt_body = b_R_w * (-omega);
    Vector3d v_pt_body = b_R_w * (-v);
    Vector3d v_cam_body = omega.cross(cam_data.tic_);
    Vector3d v_pt_cam = rci_ * (v_pt_body - v_cam_body);

    double mean_optical_flow = 0.0;

    for(int row_idx = 0; row_idx < cam_data.cal_rows_; row_idx++){
        for(int col_idx = 0; col_idx < cam_data.cal_cols_; col_idx++){
            const double& dist = cam_data.cross_section_dist_vector_[cross_section_idx](row_idx, col_idx);

            // cout<<"dist of "<<row_idx<<", "<<col_idx<<": "<<dist<<endl;

            if(dist < 0.0 || dist > max_raycast_length_){
                continue;
            } 

            const Vector3d& inv_proj_dir = cam_data.inv_proj_dir_mat_(row_idx, col_idx);
            Vector3d pt_body = cam_data.ric_ * inv_proj_dir * dist + cam_data.tic_;
            Vector3d total_v_pt_cam =  v_pt_cam + rci_ * omega_pt_body.cross(pt_body);
            
            double depth = dist * inv_proj_dir.z();
            
            double optical_flow_speed = total_v_pt_cam.head(2).norm() / depth;
            mean_optical_flow += optical_flow_speed;
        }
    }

    mean_optical_flow /= (cam_data.cal_rows_ * cam_data.cal_cols_);

    return mean_optical_flow;
}

void optical_flow_estimator::pub_raycast_result(){

    visualization_msgs::MarkerArray mk_array;
    mk_array.markers.resize(1);
    mk_array.markers[0].action = visualization_msgs::Marker::DELETEALL;
    ray_publisher_.publish(mk_array);

    mk_array.markers.clear();

    visualization_msgs::Marker mk;
    mk.action = visualization_msgs::Marker::ADD;
    mk.color.r = 1.0;
    mk.color.g = 0.3;
    mk.color.b = 0.0;
    mk.color.a = 1.0;
    mk.header.stamp = ros::Time::now();
    mk.header.frame_id = "world";
    mk.id = 0;
    mk.type = visualization_msgs::Marker::ARROW;
    mk.points.resize(2);
    mk.scale.x = 0.005;
    mk.scale.y = 0.01;
    mk.scale.z = 0;

    mk.pose.orientation.w = 1.0;
    mk.pose.orientation.x = 0.0;
    mk.pose.orientation.y = 0.0;
    mk.pose.orientation.z = 0.0;

    for(auto& cam_data : cam_data_vec_){
        for(int cs_idx = 0; cs_idx < 1; cs_idx++){

            double yaw = forward_cross_sections_ptr_->at(cs_idx).yaw_;

            Matrix3d ori;
            ori <<  cos(yaw), -sin(yaw), 0.0,
                    sin(yaw),  cos(yaw), 0.0,
                        0.0,       0.0, 1.0;

            Vector3d cam_pos = ori * cam_data.tic_ + forward_cross_sections_ptr_->at(cs_idx).center_;
            Matrix3d cam_ori = ori * cam_data.ric_;

            mk.points[0].x = cam_pos.x();
            mk.points[0].y = cam_pos.y();
            mk.points[0].z = cam_pos.z();

            for(int row_idx = 0; row_idx < cam_data.cal_rows_; row_idx++){
                for(int col_idx = 0; col_idx < cam_data.cal_cols_; col_idx++){ 
                    const double& dist = cam_data.cross_section_dist_vector_[cs_idx](row_idx, col_idx);

                    if(dist > 0.0){
                        Vector3d ray_end_pos = cam_ori * cam_data.inv_proj_dir_mat_(row_idx, col_idx) * dist + cam_pos;
                        mk.points[1].x = ray_end_pos.x();
                        mk.points[1].y = ray_end_pos.y();
                        mk.points[1].z = ray_end_pos.z();     
                        mk_array.markers.emplace_back(mk); 
                        mk.id++;              
                    }

                }
            }

        }
    }

    ray_publisher_.publish(mk_array);

}

}