/*******************************************************
Copyright (C) 2025, Aerial Robotics Group, Hong Kong University of Science and Technology
This file is part of FINT.
Licensed under the GNU General Public License v3.0;
you may not use this file except in compliance with the License.
*******************************************************/
#include "kinodynamic_astar.h"
#include <sstream>

using namespace std;
using namespace Eigen;

namespace tunnel_planner{

vector<double> optimizer_1d::solve_cubic(double a, double b, double c, double d) {
    vector<double> dts;

    double a2 = b / a;
    double a1 = c / a;
    double a0 = d / a;

    double Q = (3 * a1 - a2 * a2) / 9;
    double R = (9 * a1 * a2 - 27 * a0 - 2 * a2 * a2 * a2) / 54;
    double D = Q * Q * Q + R * R;
    if (D > 0) {
        double S = std::cbrt(R + sqrt(D));
        double T = std::cbrt(R - sqrt(D));
        dts.push_back(-a2 / 3 + (S + T));
        return dts;
    } else if (D == 0) {
        double S = std::cbrt(R);
        dts.push_back(-a2 / 3 + S + S);
        dts.push_back(-a2 / 3 - S);
        return dts;
    } else {
        double theta = acos(R / sqrt(-Q * Q * Q));
        dts.push_back(2 * sqrt(-Q) * cos(theta / 3) - a2 / 3);
        dts.push_back(2 * sqrt(-Q) * cos((theta + 2 * M_PI) / 3) - a2 / 3);
        dts.push_back(2 * sqrt(-Q) * cos((theta + 4 * M_PI) / 3) - a2 / 3);
        return dts;
    }
}

vector<double> optimizer_1d::solve_quartic(double a, double b, double c, double d, double e) {
    vector<double> dts;

    double a3 = b / a;
    double a2 = c / a;
    double a1 = d / a;
    double a0 = e / a;

    vector<double> ys = solve_cubic(1, -a2, a1 * a3 - 4 * a0, 4 * a2 * a0 - a1 * a1 - a3 * a3 * a0);
    double         y1 = ys.front();
    double         r  = a3 * a3 / 4 - a2 + y1;
    if (r < 0) return dts;

    double R = sqrt(r);
    double D, E;
    if (R != 0) {
        D = sqrt(0.75 * a3 * a3 - R * R - 2 * a2 + 0.25 * (4 * a3 * a2 - 8 * a1 - a3 * a3 * a3) / R);
        E = sqrt(0.75 * a3 * a3 - R * R - 2 * a2 - 0.25 * (4 * a3 * a2 - 8 * a1 - a3 * a3 * a3) / R);
    } else {
        D = sqrt(0.75 * a3 * a3 - 2 * a2 + 2 * sqrt(y1 * y1 - 4 * a0));
        E = sqrt(0.75 * a3 * a3 - 2 * a2 - 2 * sqrt(y1 * y1 - 4 * a0));
    }

    if (!std::isnan(D)) {
        dts.push_back(-a3 / 4 + R / 2 + D / 2);
        dts.push_back(-a3 / 4 + R / 2 - D / 2);
    }
    if (!std::isnan(E)) {
        dts.push_back(-a3 / 4 - R / 2 + E / 2);
        dts.push_back(-a3 / 4 - R / 2 - E / 2);
    }

    return dts;
}

double optimizer_1d::compute_cross_section_disturbance(float speed, float pitch, const cross_section& cross_section){
    if(pitch < 0.0){
        pitch = -pitch;
        speed = -speed;
    }

    double disturbance = 0.0;

    if(cross_section.cross_section_shape_ == tunnel_shape::OUTSIDE || cross_section.cross_section_shape_ == tunnel_shape::BEFORE){

    }
    else if(cross_section.cross_section_shape_ == tunnel_shape::RECTANGLE){
        // input: h,w,pitch(positive),speed
        disturbance = rect_net_->inference(Vector4f(cross_section.cross_section_data_[0], cross_section.cross_section_data_[1], pitch, speed));        
    }
    else if(cross_section.cross_section_shape_ == tunnel_shape::CIRCLE){
        disturbance =  circle_net_->inference(Vector3f(cross_section.cross_section_data_[0], pitch, speed));
    }

    return disturbance;
}

double optimizer_1d::get_curvature(const double& x){
    double t = tunnel_center_line_->getTimeFromLength(x);
    
    Vector2d vel_a_h = tunnel_center_vel_->evaluateDeBoorT(t).head(2);
    Vector2d acc_h = tunnel_center_acc_->evaluateDeBoorT(t).head(2);
    
    double squared_va_h = vel_a_h.squaredNorm();

    double curvature = (acc_h - acc_h.dot(vel_a_h) / squared_va_h * vel_a_h).norm() / squared_va_h;

    return curvature;

}


double optimizer_1d::calculate_vision_disturbance_cost(const Eigen::Vector2d& x1, const Eigen::Vector2d& x2, const double& acc, double& vision_cost, double& disturbance_cost){


    if(w_disturbance_ == 0.0 && w_vision_ == 0.0){
        return 0.0;
    }

    int start_cross_section_idx = 0, end_cross_section_idx = 0;

    double start_p = x1(0), end_p = x2(0);
    bool in_range = false;
    for(int i = 0; i < cross_sections_ptr_->size(); i++){
        if(start_p < cross_sections_ptr_->at(i).curve_length_){
            start_cross_section_idx = i;
            end_cross_section_idx = i;
            break;
        }
    }

    double interval_start_p = x1(0), interval_end_p;
    double interval_start_v = x1(1), interval_end_v;

    double interval_mean_v;
    double interval_dt;

    auto cross_section_1 = &(cross_sections_ptr_->at(start_cross_section_idx-1));
    auto cross_section_2 = &(cross_sections_ptr_->at(start_cross_section_idx));

    double curve_length_1 = cross_section_1->curve_length_;
    double curve_length_2 = cross_section_2->curve_length_;

    float pitch1 = asin(cross_section_1->w_R_cs.col(2).z()), pitch2 = asin(cross_section_2->w_R_cs.col(2).z());

    double interval_start_cross_section_1_ratio = (curve_length_2 - start_p) / (curve_length_2 - curve_length_1);

    double cross_section_1_disturbance = 0.0, cross_section_2_disturbance = 0.0;
    double cross_section_1_mean_optical_flow = 0.0, cross_section_2_mean_optical_flow = 0.0;

    double total_vision_cost = 0.0, total_disturbance_cost = 0.0;

    // start end between the same pair of cross sections
    if(end_p <= curve_length_2){

        interval_end_p = x2(0);
        interval_end_v = x2(1);
        interval_mean_v = 0.5 * (interval_start_v + interval_end_v);
        interval_dt = acc == 0.0 ? (interval_end_p - interval_start_p) / interval_mean_v :(interval_end_v - interval_start_v) / acc;
        
        double interval_end_cross_section_1_ratio = (curve_length_2 - end_p) / (curve_length_2 - curve_length_1);

        if(w_disturbance_ > 0.0){

            cross_section_1_disturbance = compute_cross_section_disturbance(interval_mean_v, pitch1, *cross_section_1);
            cross_section_2_disturbance = compute_cross_section_disturbance(interval_mean_v, pitch2, *cross_section_2);
            
            double interval_start_disturbance = interval_start_cross_section_1_ratio * cross_section_1_disturbance + (1.0 - interval_start_cross_section_1_ratio) * cross_section_2_disturbance;
            double interval_end_disturbance = interval_end_cross_section_1_ratio * cross_section_1_disturbance + (1.0 - interval_end_cross_section_1_ratio) * cross_section_2_disturbance;
            
            double mean_disturbance = 0.5 * (interval_start_disturbance + interval_end_disturbance);
            
            total_disturbance_cost = mean_disturbance * mean_disturbance * interval_dt;
        }


        if(w_vision_ > 0.0){

            const double& yaw_dot_cs1 = cross_section_1->yaw_dot_;
            const double& yaw_dot_cs2 = cross_section_2->yaw_dot_;

            double yaw_dot_start = interval_start_cross_section_1_ratio * yaw_dot_cs1 + (1.0 - interval_start_cross_section_1_ratio) * yaw_dot_cs2;
            double yaw_dot_end = interval_end_cross_section_1_ratio * yaw_dot_cs1 + (1.0 - interval_end_cross_section_1_ratio) * yaw_dot_cs2;
            double yaw_dot = interval_mean_v / VIRTUAL_FLIGHT_PROGRESS_SPEED * 0.5 * (yaw_dot_start + yaw_dot_end);

            cross_section_1_mean_optical_flow = optical_flow_estimator_->cal_total_mean_optical_flow(start_cross_section_idx-1, interval_mean_v, yaw_dot);
            cross_section_2_mean_optical_flow = optical_flow_estimator_->cal_total_mean_optical_flow(start_cross_section_idx, interval_mean_v, yaw_dot);

            double interval_start_mean_optical_flow = interval_start_cross_section_1_ratio * cross_section_1_mean_optical_flow + (1.0 - interval_start_cross_section_1_ratio) * cross_section_2_mean_optical_flow;
            double interval_end_mean_optical_flow = interval_end_cross_section_1_ratio * cross_section_1_mean_optical_flow + (1.0 - interval_end_cross_section_1_ratio) * cross_section_2_mean_optical_flow;

            double mean_optical_flow = 0.5 * (interval_start_mean_optical_flow + interval_end_mean_optical_flow);

            total_vision_cost = mean_optical_flow * mean_optical_flow * interval_dt;
        }
    }
    // across multiple cross sections
    else{

        // interpolate start point
        interval_end_p = cross_sections_ptr_->at(start_cross_section_idx).curve_length_;
        interval_end_v = sqrt(interval_start_v * interval_start_v + 2.0 * acc * (interval_end_p - interval_start_p));
        interval_mean_v = 0.5 * (interval_start_v + interval_end_v);        
        interval_dt = acc == 0.0 ? (interval_end_p - interval_start_p) / interval_mean_v :(interval_end_v - interval_start_v) / acc;

        if(w_disturbance_ > 0.0){
            cross_section_1_disturbance = compute_cross_section_disturbance(interval_mean_v, pitch1, *cross_section_1);
            cross_section_2_disturbance = compute_cross_section_disturbance(interval_mean_v, pitch2, *cross_section_2);
            double interval_start_disturbance = interval_start_cross_section_1_ratio * cross_section_1_disturbance + (1.0 - interval_start_cross_section_1_ratio) * cross_section_2_disturbance;
            
            double mean_disturbance = 0.5 * (interval_start_disturbance + cross_section_2_disturbance);

            total_disturbance_cost += mean_disturbance * mean_disturbance * interval_dt;
        }


        if(w_vision_ > 0.0){


            const double& yaw_dot_cs1 = cross_section_1->curve_length_;
            const double& yaw_dot_cs2 = cross_section_2->curve_length_;

            double yaw_dot_start = interval_start_cross_section_1_ratio * yaw_dot_cs1 + (1.0 - interval_start_cross_section_1_ratio) * yaw_dot_cs2;
            double yaw_dot_end = yaw_dot_cs2;

            double yaw_dot = interval_mean_v / VIRTUAL_FLIGHT_PROGRESS_SPEED * 0.5 * (yaw_dot_start + yaw_dot_end);

            cross_section_1_mean_optical_flow = optical_flow_estimator_->cal_total_mean_optical_flow(start_cross_section_idx-1, interval_mean_v, yaw_dot);
            cross_section_2_mean_optical_flow = optical_flow_estimator_->cal_total_mean_optical_flow(start_cross_section_idx, interval_mean_v, yaw_dot);

            double interval_start_mean_optical_flow = interval_start_cross_section_1_ratio * cross_section_1_mean_optical_flow + (1.0 - interval_start_cross_section_1_ratio) * cross_section_2_mean_optical_flow;
            
            double mean_optical_flow = 0.5 * (interval_start_mean_optical_flow + cross_section_2_mean_optical_flow);
            
            total_vision_cost += mean_optical_flow * mean_optical_flow * interval_dt;
        }


        // compute between cross sections
        for(int i = start_cross_section_idx+1; i < cross_sections_ptr_->size(); i++){

            cross_section_1 = cross_section_2;
            cross_section_2 = &(cross_sections_ptr_->at(i));

            interval_start_p = interval_end_p;
            interval_start_v = interval_end_v;

            pitch1 = pitch2;
            pitch2 = asin(cross_section_2->w_R_cs.col(2).z());

            if(end_p <= cross_sections_ptr_->at(i).curve_length_){
                end_cross_section_idx = i;
                break;
            }

            interval_end_p = cross_sections_ptr_->at(i).curve_length_;
            interval_end_v = sqrt(interval_start_v * interval_start_v + 2.0 * acc * (interval_end_p - interval_start_p));
            interval_mean_v = 0.5 * (interval_start_v + interval_end_v);        
            interval_dt = acc == 0.0 ? (interval_end_p - interval_start_p) / interval_mean_v :(interval_end_v - interval_start_v) / acc;

            if(w_disturbance_ > 0.0){
                cross_section_1_disturbance = compute_cross_section_disturbance(interval_mean_v, pitch1, *cross_section_1);
                cross_section_2_disturbance = compute_cross_section_disturbance(interval_mean_v, pitch2, *cross_section_2);


                double mean_disturbance = 0.5 * (cross_section_1_disturbance + cross_section_2_disturbance);

                total_disturbance_cost += mean_disturbance * mean_disturbance * interval_dt;
            }


            if(w_vision_ > 0.0){

                double yaw_dot_start = cross_sections_ptr_->at(i-1).yaw_dot_;
                double yaw_dot_end = cross_sections_ptr_->at(i).yaw_dot_;

                double yaw_dot = interval_mean_v / VIRTUAL_FLIGHT_PROGRESS_SPEED * 0.5 * (yaw_dot_start + yaw_dot_end);

                cross_section_1_mean_optical_flow = optical_flow_estimator_->cal_total_mean_optical_flow(i-1, interval_mean_v, yaw_dot);
                cross_section_2_mean_optical_flow = optical_flow_estimator_->cal_total_mean_optical_flow(i, interval_mean_v, yaw_dot);
            
                double mean_optical_flow = 0.5 * (cross_section_1_mean_optical_flow + cross_section_2_mean_optical_flow);

                total_vision_cost += mean_optical_flow * mean_optical_flow * interval_dt;
            }

        }

        // interpolate end point

        interval_end_p = x2(0);
        interval_end_v = x2(1);
        interval_mean_v = 0.5 * (interval_start_v + interval_end_v);        
        interval_dt = acc == 0.0 ? (interval_end_p - interval_start_p) / interval_mean_v :(interval_end_v - interval_start_v) / acc;

        curve_length_1 = cross_sections_ptr_->at(end_cross_section_idx-1).curve_length_;
        curve_length_2 = cross_sections_ptr_->at(end_cross_section_idx).curve_length_;

        double interval_end_cross_section_1_ratio = (curve_length_2 - interval_end_p) / (curve_length_2 - curve_length_1);

        if(w_disturbance_ > 0.0){
            cross_section_1_disturbance = compute_cross_section_disturbance(interval_mean_v, pitch1, *cross_section_1);
            cross_section_2_disturbance = compute_cross_section_disturbance(interval_mean_v, pitch2, *cross_section_2);

            double interval_end_disturbance = interval_end_cross_section_1_ratio * cross_section_1_disturbance + (1.0 - interval_end_cross_section_1_ratio) * cross_section_2_disturbance;
            
            double mean_disturbance = 0.5 * (cross_section_1_disturbance + interval_end_disturbance);

            total_disturbance_cost += mean_disturbance * mean_disturbance * interval_dt;
        }
       
        if(w_vision_ > 0.0){

            const double& yaw_dot_cs1 = cross_sections_ptr_->at(end_cross_section_idx-1).yaw_dot_;
            const double& yaw_dot_cs2 = cross_sections_ptr_->at(end_cross_section_idx).yaw_dot_;

            double yaw_dot_start = yaw_dot_cs1;
            double yaw_dot_end = interval_end_cross_section_1_ratio * yaw_dot_cs1 + (1.0 - interval_end_cross_section_1_ratio) * yaw_dot_cs2;;

            double yaw_dot = interval_mean_v / VIRTUAL_FLIGHT_PROGRESS_SPEED * 0.5 * (yaw_dot_start + yaw_dot_end);

            cross_section_1_mean_optical_flow = optical_flow_estimator_->cal_total_mean_optical_flow(end_cross_section_idx-1, interval_mean_v, yaw_dot);
            cross_section_2_mean_optical_flow = optical_flow_estimator_->cal_total_mean_optical_flow(end_cross_section_idx, interval_mean_v, yaw_dot);

            double interval_end_mean_optical_flow = interval_end_cross_section_1_ratio * cross_section_1_mean_optical_flow + (1.0 - interval_end_cross_section_1_ratio) * cross_section_2_mean_optical_flow;

            double mean_optical_flow = 0.5 * (cross_section_1_mean_optical_flow + interval_end_mean_optical_flow);

            total_vision_cost += mean_optical_flow * mean_optical_flow * interval_dt;
        }
        
    }

    double interval_disturbance_cost = w_disturbance_ * total_disturbance_cost;
    double interval_vision_cost = w_vision_ * total_vision_cost;

    disturbance_cost += interval_disturbance_cost;
    vision_cost += interval_vision_cost;

    return interval_disturbance_cost + interval_vision_cost;
}

double optimizer_1d::estimate_heuristic(Eigen::Vector2d& x1, Eigen::Vector2d& x2, double& optimal_time) {
    const double dp = x2(0) - x1(0);
    const double v0 = x1(1);
    const double v1 = x2(1);

    double c1 = -36 * dp * dp;
    double c2 = 24 * (v0 + v1) * dp;
    double c3 = -4 * (v0 * v0 + v0 * v1 + v1 * v1);
    double c4 = 0;
    double c5 = w_time_;

    std::vector<double> ts = solve_quartic(c5, c4, c3, c2, c1);

    if((x2 - x1).squaredNorm() < 1e-8)
        return 0.0;

    double t_bar = abs(x2(0) - x1(0)) / max_speed_;
    ts.push_back(t_bar);

    double cost = numeric_limits<double>::max();
    double t_d  = t_bar;

    for (auto t : ts) {
        if (t < t_bar) continue;
        
        double c = -c1 / (3 * t * t * t) - c2 / (2 * t * t) - c3 / t + w_time_ * t;
        
        if (c < cost) {
            cost = c;
            t_d  = t;
        }
    }

    optimal_time = t_d;

    return tie_breaker_ * cost;
}

void optimizer_1d::state_transit(Vector2d& state0, Vector2d& state1, double um, double tau){

    state1(1) = state0(1) + um * tau;
    state1(0) = state0(0) + 0.5 * (state0(1) + state1(1)) * tau;
}


void optimizer_1d::retrieve_path(path_node* end_node_ptr){
    path_node* cur_node = end_node_ptr;
    search_result_.emplace_back(cur_node);

    while (cur_node->parent_ != nullptr) {
        cur_node = cur_node->parent_;
        search_result_.emplace_back(cur_node);
    }
}

int optimizer_1d::astar_search(const double& start_v){

    reset();

    /* ---------- initialize ---------- */
    auto new_node_it = path_node_pool_.begin();
    path_node* cur_node_ptr = &(*new_node_it);
    cur_node_ptr->parent_ = nullptr;
    cur_node_ptr->state_ << 0.0, start_v;
    cur_node_ptr->g_score_ = 0.0;
    cur_node_ptr->path_step_ = 0;
    cur_node_ptr->time_from_parent_ = 0.0;
    cur_node_ptr->total_time_up_to_node_ = 0.0;

    cur_node_ptr->disturbance_cost = 0.0;
    cur_node_ptr->vision_cost = 0.0;


    Vector2d end_state(cross_sections_ptr_->back().curve_length_, 0.0);
    double time_to_goal;
    if(free_end_v_){
        cur_node_ptr->f_score_ = 0.0;
    }
    else{
        cur_node_ptr->f_score_    = w_heuristic_ * estimate_heuristic(cur_node_ptr->state_, end_state, time_to_goal);
    }
    cur_node_ptr->node_search_state_ = IN_OPEN;

    open_set_.insert(cur_node_ptr);

    new_node_it++;
    use_node_num_ += 1;

    path_node* cur_best_node_ptr = nullptr;

    std::chrono::time_point<std::chrono::system_clock> search_start_time = std::chrono::system_clock::now();

    /* ---------- search loop ---------- */
    while (!open_set_.empty()) {
        /* ---------- get lowest f_score node ---------- */

        cur_node_ptr = *(open_set_.begin());


        /* ---------- determine termination ---------- */
        if(free_end_v_){
            if((std::chrono::system_clock::now() - search_start_time).count() >= max_search_time_ * 1e9){
                // std::cout << "[Astar out of time]:---------------------- " << use_node_num_ << std::endl;
                if(cur_best_node_ptr){              
                    // std::cout << "use node num: " << use_node_num_ << std::endl;
                    // std::cout << "iter num: " << iter_num_ << std::endl;
 
                    retrieve_path(cur_best_node_ptr);

                    return SUCCESS;
                }
                else{
                    return NO_PATH;
                }
            }
        }
        else{
            if((cur_node_ptr->state_ - end_state).squaredNorm() < 1e-8){
                // std::cout << "[Astar]:---------------------- " << use_node_num_ << std::endl;
                // std::cout << "use node num: " << use_node_num_ << std::endl;
                // std::cout << "iter num: " << iter_num_ << std::endl;

                retrieve_path(cur_node_ptr);

                return SUCCESS;
            }

            if((std::chrono::system_clock::now() - search_start_time).count() >= max_search_time_ * 1e9){
                // std::cout << "[Astar out of time]:---------------------- " << use_node_num_ << std::endl;
                if(cur_best_node_ptr){              
                    // std::cout << "use node num: " << use_node_num_ << std::endl;
                    // std::cout << "iter num: " << iter_num_ << std::endl;

                    retrieve_path(cur_best_node_ptr);

                    return SUCCESS;
                }
                else{
                    return NO_PATH;
                }
            }
            
        }

        /* ---------- pop node and add to close set ---------- */
        // open_set_.pop();
        open_set_.erase(open_set_.begin());

        close_states_.emplace_back(cur_node_ptr->state_);
        
        cur_node_ptr->node_search_state_ = IN_CLOSE;
        iter_num_ += 1;

        /* ---------- init state propagation ---------- */

        Vector2d cur_state = cur_node_ptr->state_;
        Vector2d pro_state;
        double um;
        double pro_t;


        /* ---------- state propagation loop ---------- */

        for (int i = 0; i < inputs_.size(); ++i){
            um = inputs_[i];
            double dt = time_res_;
            state_transit(cur_state, pro_state, um, dt);
            pro_t = cur_node_ptr->total_time_up_to_node_ + dt;
            double pro_x = pro_state(0);

            // over the end
            if (pro_x >= end_state(0)) {

                if(free_end_v_){
                    // maintain input um
                    pro_state = end_state;
                    double end_v = sqrt(cur_state(1) * cur_state(1) + 2.0 * um * (pro_state(0) - cur_state(0)));
                    if(fabs(um) > 1e-4){
                        dt = (end_v - cur_state(1)) / um;
                    }
                    else{
                        dt = fabs((pro_state(0) - cur_state(0)) / cur_state(1));
                    }
                    
                    pro_t = cur_node_ptr->total_time_up_to_node_ + dt;
                    pro_state(1) = end_v;
                }
                else{
                    dt = 2.0 * (end_state(0) - cur_state(0)) / cur_state(1);
                    um = -cur_state(1) / dt;
                    pro_t = cur_node_ptr->total_time_up_to_node_ + dt;
                    pro_state = end_state;
                    pro_x = pro_state(0); 
                }

            }

            /* check reverse travel distance*/
            if (pro_x <= cur_state(0)) {
                continue;
            }


            /* vel feasibe */
            double pro_v = pro_state(1);
            if (pro_v < 0.0 || pro_v > max_speed_ || ((!free_end_v_) && pro_v * pro_v > 2.0 * max_acc_ * (end_state(0) - pro_x))) {
                continue;
            }

            /* not in close set */
            bool in_close = false;
            for (auto& close_state : close_states_) {
                if ((close_state - pro_state).squaredNorm() < 1e-8) {
                    in_close = true;
                    break;
                }
            }
            if(in_close){
                continue;
            }


            /* ---------- compute cost ---------- */
            double time_to_goal, tmp_g_score, tmp_f_score;
            tmp_g_score = (um * um + w_time_) * dt + cur_node_ptr->g_score_;
            

            double tmp_disturbance_cost = cur_node_ptr->disturbance_cost, tmp_vision_cost = cur_node_ptr->vision_cost;

            if(w_disturbance_ > 0.0 || w_vision_ > 0.0){
                double vision_disturbance_cost = calculate_vision_disturbance_cost(cur_state, pro_state, um, tmp_vision_cost, tmp_disturbance_cost);
                if(isnan(vision_disturbance_cost)){
                    ROS_ERROR("disturbance cost nan");
                }

                tmp_g_score += vision_disturbance_cost;

            }

            tmp_f_score = tmp_g_score;

            if(!free_end_v_){
                tmp_f_score += w_heuristic_ * estimate_heuristic(pro_state, end_state, time_to_goal);
            } 

            /* ---------- compare expanded node in this loop ---------- */


            bool prune = false;        
            for (auto it = open_set_.begin(); it != open_set_.end(); it++) {
                if (((*it)->state_ - pro_state).squaredNorm() < 1e-8) {
                        if (tmp_f_score <= (*it)->f_score_) {
                            open_set_.erase(it);
                        }
                        else{
                            prune = true;
                        }
                    break;
                }
            }

            /* ---------- new neighbor in this loop ---------- */

            if(!prune){
                path_node* pro_node_ptr  = &(*new_node_it);
                pro_node_ptr->state_      = pro_state;
                pro_node_ptr->f_score_    = tmp_f_score;
                pro_node_ptr->g_score_    = tmp_g_score;
                pro_node_ptr->prev_input_ = um;
                pro_node_ptr->total_time_up_to_node_ = pro_t;
                pro_node_ptr->time_from_parent_ = dt;
                pro_node_ptr->parent_     = cur_node_ptr;
                pro_node_ptr->node_search_state_ = IN_OPEN;

                pro_node_ptr->disturbance_cost = tmp_disturbance_cost;
                pro_node_ptr->vision_cost = tmp_vision_cost;


                if(free_end_v_){
                    if(fabs(pro_state(0) - end_state(0)) < 1e-8){
                        if(!cur_best_node_ptr || cur_best_node_ptr->f_score_ > tmp_f_score){
                            cur_best_node_ptr = pro_node_ptr;
                        }
                    }
                }
                else{
                    if((pro_state - end_state).squaredNorm() < 1e-8){
                        if(!cur_best_node_ptr || cur_best_node_ptr->f_score_ > tmp_f_score){
                            cur_best_node_ptr = pro_node_ptr;
                        }
                    }
                }

                open_set_.insert(pro_node_ptr);

                new_node_it++;
                use_node_num_ += 1;
                if (use_node_num_ == max_node_num_) {
                    std::cout << "run out of memory." << endl;
                    return NO_PATH;
                }
            }


            /* ----------  ---------- */
        }
    }

    if(free_end_v_){        
        // std::cout << "[Astar free end v]:---------------------- " << use_node_num_ << std::endl;
        if(cur_best_node_ptr){              
            // std::cout << "use node num: " << use_node_num_ << std::endl;
            // std::cout << "iter num: " << iter_num_ << std::endl;
            retrieve_path(cur_best_node_ptr);

            return SUCCESS;
        }
        else{
            return NO_PATH;
        }
        
    }

    /* ---------- open set empty, no path ---------- */
    std::cout << "open set empty, no path!" << endl;
    std::cout << "use node num: " << use_node_num_ << endl;
    std::cout << "iter num: " << iter_num_ << endl;
    return NO_PATH;
}

int optimizer_1d::const_speed_search(const double& start_v, double& end_v){
    reset();

    double end_p = cross_sections_ptr_->back().curve_length_;

    /* ---------- initialize ---------- */
    auto new_node_it = path_node_pool_.begin();
    path_node* init_node_ptr = &(*new_node_it);

    new_node_it++;
    use_node_num_ += 1;

    init_node_ptr->parent_ = nullptr;
    init_node_ptr->state_ << 0.0, start_v;
    init_node_ptr->g_score_ = 0.0;
    init_node_ptr->path_step_ = 0;
    init_node_ptr->time_from_parent_ = 0.0;
    init_node_ptr->total_time_up_to_node_ = 0.0;

    init_node_ptr->disturbance_cost = 0.0;
    init_node_ptr->vision_cost = 0.0;

    init_node_ptr->prev_input_ = 0.0;

    /* ---------- accelerate to desire speed ---------- */
    path_node* acc_node_ptr = &(*new_node_it);

    new_node_it++;
    use_node_num_ += 1;

    acc_node_ptr->parent_ = init_node_ptr;

    acc_node_ptr->path_step_ = 1;

    double acc_dist = 0.0;

    if(start_v > des_speed_ + 1e-4){
        acc_node_ptr->prev_input_ = (-max_acc_);
        acc_node_ptr->time_from_parent_ = (start_v - des_speed_) / max_acc_;
        acc_dist = 0.5 * (start_v + des_speed_) * acc_node_ptr->time_from_parent_;
    }else if(start_v < des_speed_ - 1e-4){
        acc_node_ptr->prev_input_ = max_acc_;
        acc_node_ptr->time_from_parent_ = (des_speed_ - start_v) / max_acc_;
        acc_dist = 0.5 * (start_v + des_speed_) * acc_node_ptr->time_from_parent_;
    }else{
        acc_node_ptr->prev_input_ = 0.0;
        acc_node_ptr->time_from_parent_ = end_p / des_speed_;
        acc_dist = end_p;
    }

    if(acc_dist > end_p){
        acc_dist = end_p;
        acc_node_ptr->prev_input_ = max_acc_;
        end_v = sqrt(start_v * start_v + 2.0 * max_acc_ * acc_dist);
        acc_node_ptr->time_from_parent_ = (end_v - start_v) / max_acc_;

        acc_node_ptr->total_time_up_to_node_ = acc_node_ptr->time_from_parent_;
        acc_node_ptr->state_ << end_p, end_v;

        search_result_.emplace_back(acc_node_ptr);
        search_result_.emplace_back(init_node_ptr);

        return SUCCESS;
    }
    else if(acc_dist == end_p){
        acc_node_ptr->total_time_up_to_node_ = acc_node_ptr->time_from_parent_;
        acc_node_ptr->state_ << end_p, des_speed_;

        search_result_.emplace_back(acc_node_ptr);
        search_result_.emplace_back(init_node_ptr);

        end_v = des_speed_;

        return SUCCESS;
    }
    else{
        acc_node_ptr->state_ << acc_dist, des_speed_;
        acc_node_ptr->total_time_up_to_node_ = acc_node_ptr->time_from_parent_;
    }

    /* ---------- constant speed ---------- */
    path_node* const_speed_node_ptr = &(*new_node_it);

    new_node_it++;
    use_node_num_ += 1;

    const_speed_node_ptr->parent_ = acc_node_ptr;
    const_speed_node_ptr->path_step_ = 2;

    const_speed_node_ptr->prev_input_ = 0.0;
    const_speed_node_ptr->time_from_parent_ = (end_p - acc_dist) / des_speed_;

    const_speed_node_ptr->total_time_up_to_node_ = acc_node_ptr->total_time_up_to_node_ + const_speed_node_ptr->time_from_parent_;
    const_speed_node_ptr->state_ << end_p, des_speed_;

    search_result_.emplace_back(const_speed_node_ptr);
    search_result_.emplace_back(acc_node_ptr);
    search_result_.emplace_back(init_node_ptr);

    end_v = des_speed_;

    return SUCCESS;

}

int optimizer_1d::const_dcc_search(const double& start_v){
    reset();

    double end_p = cross_sections_ptr_->back().curve_length_;

    /* ---------- initialize ---------- */
    auto new_node_it = path_node_pool_.begin();
    path_node* init_node_ptr = &(*new_node_it);

    new_node_it++;
    use_node_num_ += 1;

    init_node_ptr->parent_ = nullptr;
    init_node_ptr->state_ << 0.0, start_v;
    init_node_ptr->g_score_ = 0.0;
    init_node_ptr->path_step_ = 0;
    init_node_ptr->time_from_parent_ = 0.0;
    init_node_ptr->total_time_up_to_node_ = 0.0;

    init_node_ptr->disturbance_cost = 0.0;
    init_node_ptr->vision_cost = 0.0;

    init_node_ptr->prev_input_ = 0.0;

    /* ---------- decelerate to 0 speed ---------- */
    path_node* dcc_node_ptr = &(*new_node_it);

    new_node_it++;
    use_node_num_ += 1;

    dcc_node_ptr->parent_ = init_node_ptr;

    dcc_node_ptr->path_step_ = 1;

    double dcc_dist = end_p;
    dcc_node_ptr->time_from_parent_ = abs(2.0 * dcc_dist / start_v);
    dcc_node_ptr->prev_input_ = -0.5 * (start_v * start_v) / dcc_dist;

    dcc_node_ptr->total_time_up_to_node_ = dcc_node_ptr->time_from_parent_;
    dcc_node_ptr->state_ << end_p, 0.0;

    search_result_.emplace_back(dcc_node_ptr);
    search_result_.emplace_back(init_node_ptr);

    return SUCCESS;

}

vector<double> optimizer_1d::get_init_sample(double& delta_t, double& end_v){

    double total_traj_time = search_result_.front()->total_time_up_to_node_;
    int num_seg = static_cast<int>(ceil(total_traj_time / delta_t));
    delta_t = total_traj_time / num_seg;
    
    vector<double> pos_list;
    Vector2d cur_state = search_result_.front()->state_;
    pos_list.emplace_back(cur_state(0));
    path_node* path_node_seg = search_result_.front();

    if(free_end_v_){
        end_v = cur_state(1);
    }
    else{
        end_v = 0.0;
    }


    for(int seg_idx = num_seg; seg_idx > 0; seg_idx--){

        double seg_end_time = seg_idx * delta_t;
        double seg_start_time = (seg_idx - 1) * delta_t;

        // find path_node_seg time after seg_end_time
        while(seg_end_time < path_node_seg->parent_->total_time_up_to_node_){
            path_node_seg = path_node_seg->parent_;
        }
        cur_state = path_node_seg->state_;

        Vector2d prev_state = cur_state;

        // find path_node_seg time after seg_start_time
        while(seg_start_time < path_node_seg->parent_->total_time_up_to_node_){
            path_node_seg = path_node_seg->parent_;
        }
        cur_state = path_node_seg->state_;
        double remain_t = path_node_seg->total_time_up_to_node_ - seg_start_time;

        state_transit(cur_state, prev_state, path_node_seg->prev_input_, -remain_t);
        pos_list.emplace_back(prev_state(0));
        
    }

    reverse(pos_list.begin(), pos_list.end());

    if(abs(pos_list.front()) < 1e-4){
        pos_list.front() = 0.0;
    }

    return pos_list;
}

void optimizer_1d::set_param(ros::NodeHandle &nh){

    algorithm1_ = QUAD_ALGORITHM_ID;
    algorithm2_ = NON_QUAD_ALGORITHM_ID;

    w_smooth_jerk_ = W_SMOOTH_1d_JERK;
    w_smooth_acc_ = W_SMOOTH_1d_ACC;
    w_smooth_yaw_ = W_SMOOTH_YAW;
    w_start_ = W_START;
    w_end_ = W_END;
    w_end_hard_ = W_END_HARD;
    w_waypt_ = W_WAYPT;
    w_feasi_ = W_FEASI;

    max_iteration_num_ = MAX_ITERATION_NUM2;
    max_iteration_time_ = MAX_ITERATION_TIME2;

}

void optimizer_1d::set_cost_function(const int &cost_code)
{
    cost_function_ = cost_code;

    // print optimized cost function
    string cost_str("| ");
    if (cost_function_ & SMOOTHNESS_JERK)
        cost_str += "smooth jerk |";
    if (cost_function_ & SMOOTHNESS_ACC)
        cost_str += "smooth acc |";
    if (cost_function_ & SMOOTHNESS_YAW_JERK)
        cost_str += "smooth yaw jerk |";
    if (cost_function_ & SMOOTHNESS_YAW_ACC)
        cost_str += "smooth yaw acc |";
    if (cost_function_ & SMOOTHNESS_YAW_SPEED)
        cost_str += "smooth yaw speed |";
    if (cost_function_ & START){
        cost_str += " start |";
        hard_start_constraint_ = false;
    }
    else{
        hard_start_constraint_ = true;
    }
    if (cost_function_ & END)
        cost_str += " end   |";
    if (cost_function_ & END_HARD)
        cost_str += " end hard |";
    if (cost_function_ & WAYPOINTS)
        cost_str += " waypt |";
    if (cost_function_ & FEASIBILITY)
        cost_str += " feasi |";
    if (cost_function_ & VISION)
        cost_str += " vision |";
    if (cost_function_ & AIR_FLOW_DISTURBANCE)
        cost_str += " disturbance |";
    if (cost_function_ & CONST_SPEED)
        cost_str += " const speed |";


    ROS_DEBUG_STREAM("cost func 1d: " << cost_str);
}



void optimizer_1d::set_boundary_states(const vector<double> &start, const vector<double> &end)
{
    start_state_ = start;
    end_state_ = end;
}

void optimizer_1d::set_waypoints(const vector<double> &waypts, const vector<int> &waypt_idx)
{
    waypoints_ = waypts;
    waypt_idx_ = waypt_idx;
}

void optimizer_1d::add_gradient(vector<double> &grad, const double& weight, const vector<double>& g){
    for (int i = 0; i < point_num_; i++)
        grad[i] += weight * g[i];
}

bool optimizer_1d::isQuadratic()
{
    if (cost_function_ == WAY_PT_JERK_PHASE)
    {
        return true;
    }
    else if (cost_function_ == SMOOTHNESS_JERK)
    {
        return true;
    }
    else if (cost_function_ == (SMOOTHNESS_JERK | START | END_HARD) || cost_function_ == (SMOOTHNESS_JERK | START | END))
    {
        return true;
    }

    return false;
}

void optimizer_1d::optimize_1d(Eigen::VectorXd &points, double &dt, const int &cost_function)
{
    if (start_state_.empty())
    {
        ROS_ERROR("Initial state undefined!");
        return;
    }
    control_points_ = points;
    knot_span_ = dt;

    // Set necessary data and flag
    point_num_ = control_points_.rows();
    variable_num_ = point_num_;

    if (variable_num_ <= 0)
    {
        ROS_ERROR("Empty varibale to optimization solver.");
        return;
    }

    set_cost_function(cost_function);


    pt_dist_ = 0.0;
    for (int i = 0; i < control_points_.rows() - 1; ++i)
    {
        pt_dist_ += abs(control_points_(i + 1) - control_points_(i));
    }

    pt_dist_ /= point_num_;


    iter_num_ = 0;
    min_cost_ = std::numeric_limits<double>::max();

    g_q_.resize(point_num_);
    g_smoothness_jerk_.resize(point_num_);
    g_smoothness_acc_.resize(point_num_);
    g_smoothness_speed_.resize(point_num_);
    g_start_.resize(point_num_);
    g_feasibility_.resize(point_num_);
    g_end_.resize(point_num_);
    g_waypoints_.resize(point_num_);
    g_vision_.resize(point_num_);
    g_disturbance_.resize(point_num_);

    optimize_1d();

    points = control_points_;
    dt = knot_span_;
    start_state_.clear();
}


void optimizer_1d::optimize_1d()
{
    // Optimize all control points and maybe knot span dt
    // Use NLopt solver

    optimizer_.reset(new nlopt::opt(nlopt::algorithm(isQuadratic() ? algorithm1_ : algorithm2_), variable_num_));
    optimizer_->set_min_objective(optimizer_1d::cost_function, this);

    optimizer_->set_maxeval(max_iteration_num_);
    optimizer_->set_maxtime(max_iteration_time_);

    optimizer_->set_xtol_rel(1e-6);


    vector<double> q(variable_num_);
    // Variables for control points
    for (int i = 0; i < point_num_; ++i)    
        q[i] = control_points_(i);

    best_variable_ = q;
    vector<double> grad_temp;
    combine_cost(q, grad_temp, min_cost_);


    vector<double> lb(variable_num_), ub(variable_num_);
    const double bound = 10.0;
    for (int i = 0; i < point_num_; ++i)
    {
        lb[i] = q[i] - bound;
        ub[i] = q[i] + bound;
    }

    optimizer_->set_lower_bounds(lb);
    optimizer_->set_upper_bounds(ub);


    try
    {
        double final_cost;
        nlopt::result result = optimizer_->optimize(q, final_cost);
    }
    catch (std::exception &e)
    {
        // cout << e.what() << endl;
    }
    for (int i = 0; i < point_num_; ++i)
            control_points_(i) = best_variable_[i];

}


void optimizer_1d::calc_smoothness_jerk_cost(const vector<double> &q, const double &dt,
                                          double &cost, vector<double> &gradient_q)
{
    cost = 0.0;
    std::fill(gradient_q.begin(), gradient_q.end(), 0.0);
    double temp_j;

    for (int i = 0; i < q.size() - 3; i++)
    {

        // Evaluate jerk cost
        double ji = (q[i + 3] - 3 * q[i + 2] + 3 * q[i + 1] - q[i]) / pt_dist_;
        cost += (ji * ji);
        temp_j = 2 * ji / pt_dist_;

        gradient_q[i + 0] += -temp_j;
        gradient_q[i + 1] += 3.0 * temp_j;
        gradient_q[i + 2] += -3.0 * temp_j;
        gradient_q[i + 3] += temp_j;

    }

    if(hard_start_constraint_){
        for(int i = 0; i < 3; i++){
            gradient_q[i] = 0.0;
        }
    }

}

void optimizer_1d::calc_smoothness_acc_cost(const vector<double> &q, const double &dt,
                                          double &cost, vector<double> &gradient_q)
{
    cost = 0.0;
    std::fill(gradient_q.begin(), gradient_q.end(), 0.0);
    double temp_j;

    for (int i = 0; i < q.size() - 2; i++)
    {

        // Evaluate acc cost
        double ji = (q[i + 2] - 2 * q[i + 1] + q[i]) / pt_dist_;
        cost += (ji * ji);
        temp_j = 2 * ji / pt_dist_;

        gradient_q[i + 0] += temp_j;
        gradient_q[i + 1] += (-2.0 * temp_j);
        gradient_q[i + 2] += temp_j;

    }

    if(hard_start_constraint_){
        for(int i = 0; i < 3; i++){
            gradient_q[i] = 0.0;
        }
    }

}

void optimizer_1d::calc_smoothness_speed_cost(const vector<double> &q, const double &dt,
                                          double &cost, vector<double> &gradient_q)
{
    cost = 0.0;
    std::fill(gradient_q.begin(), gradient_q.end(), 0.0);
    double temp_j;

    for (int i = 0; i < q.size() - 1; i++)
    {

        // Evaluate speed cost
        double ji = (q[i + 1] - q[i]) / pt_dist_;
        cost += (ji * ji);
        temp_j = 2 * ji / pt_dist_;

        gradient_q[i + 0] -= temp_j;
        gradient_q[i + 1] += temp_j;

    }

    if(hard_start_constraint_){
        for(int i = 0; i < 3; i++){
            gradient_q[i] = 0.0;
        }
    }

}

void optimizer_1d::calc_const_speed_cost(const vector<double> &q, const double &dt,
                                          double &cost, vector<double> &gradient_q)
{
    cost = 0.0;
    std::fill(gradient_q.begin(), gradient_q.end(), 0.0);

    // Abbreviation of params
    const double dt_inv = 1.0 / dt;

    for (int i = 0; i < q.size() - 1; ++i)
    {
        // Control point of velocity
        double vi = (q[i + 1] - q[i]) * dt_inv;
        double vd = abs(vi) - des_speed_;
        cost += pow(vd, 2);
        double tmp = 2.0 * vd * dt_inv;
        gradient_q[i] -= tmp;
        gradient_q[i+1] += tmp;
    }

    if(hard_start_constraint_){
        for(int i = 0; i < 3; i++){
           gradient_q[i] = 0.0;
        }
    } 

}

void optimizer_1d::calc_start_cost(const vector<double> &q, const double &dt, double &cost,
                                     vector<double> &gradient_q)
{
    cost = 0.0;
    std::fill(gradient_q.begin(), gradient_q.end(), 0.0);

    double q1, q2, q3, dq;
    q1 = q[0];
    q2 = q[1];
    q3 = q[2];

    // Start position
    static const double w_pos = 10.0;
    dq = 1 / 6.0 * (q1 + 4 * q2 + q3) - start_state_[0];
    cost += w_pos * dq * dq;
    gradient_q[0] += w_pos * 2 * dq * (1 / 6.0);
    gradient_q[1] += w_pos * 2 * dq * (4 / 6.0);
    gradient_q[2] += w_pos * 2 * dq * (1 / 6.0);

    // Start velocity
    dq = 1 / (2 * dt) * (q3 - q1) - start_state_[1];
    cost += (dq * dq);
    gradient_q[0] += 2 * dq * (-1.0) / (2 * dt);
    gradient_q[2] += 2 * dq * 1.0 / (2 * dt);

    // Start acceleration
    dq = 1 / (dt * dt) * (q1 - 2 * q2 + q3) - start_state_[2];
    cost += (dq * dq);
    gradient_q[0] += 2 * dq * 1.0 / (dt * dt);
    gradient_q[1] += 2 * dq * (-2.0) / (dt * dt);
    gradient_q[2] += 2 * dq * 1.0 / (dt * dt);
}

void optimizer_1d::calc_end_cost(const vector<double> &q, const double &dt, double &cost,
                                   vector<double> &gradient_q)
{
    cost = 0.0;
    std::fill(gradient_q.begin(), gradient_q.end(), 0.0);

    double q_3, q_2, q_1, dq;
    q_3 = q[q.size() - 3];
    q_2 = q[q.size() - 2];
    q_1 = q[q.size() - 1];

    // End position
    dq = 1 / 6.0 * (q_1 + 4 * q_2 + q_3) - end_state_[0];
    cost += (dq * dq);
    gradient_q[q.size() - 1] += 2 * dq * (1 / 6.0);
    gradient_q[q.size() - 2] += 2 * dq * (4 / 6.0);
    gradient_q[q.size() - 3] += 2 * dq * (1 / 6.0);

    if (end_state_.size() >= 2)
    {
        // End velocity
        dq = 1 / (2 * dt) * (q_1 - q_3) - end_state_[1];
        cost += (dq * dq);
        gradient_q[q.size() - 1] += 2 * dq * 1.0 / (2 * dt);
        gradient_q[q.size() - 3] += 2 * dq * (-1.0) / (2 * dt);
    }
    if (end_state_.size() == 3)
    {
        // End acceleration
        dq = 1 / (dt * dt) * (q_1 - 2 * q_2 + q_3) - end_state_[2];
        cost += (dq * dq);
        gradient_q[q.size() - 1] += 2 * dq * 1.0 / (dt * dt);
        gradient_q[q.size() - 2] += 2 * dq * (-2.0) / (dt * dt);
        gradient_q[q.size() - 3] += 2 * dq * 1.0 / (dt * dt);
    }


    if(hard_start_constraint_){
        for(int i = 0; i < 3; i++){
            gradient_q[i] = 0.0;
        }
    }
}

void optimizer_1d::calc_waypoints_cost(const vector<double> &q, double &cost,
                                         vector<double> &gradient_q)
{
    cost = 0.0;
    std::fill(gradient_q.begin(), gradient_q.end(), 0.0);

    double q1, q2, q3, dq;

    for (int i = 0; i < waypoints_.size(); ++i)
    {
        double waypt = waypoints_[i];
        int idx = waypt_idx_[i];

        q1 = q[idx];
        q2 = q[idx + 1];
        q3 = q[idx + 2];

        dq = 1 / 6.0 * (q1 + 4 * q2 + q3) - waypt;
        cost += (dq * dq);

        gradient_q[idx] += dq * (2.0 / 6.0);     // 2*dq*(1/6)
        gradient_q[idx + 1] += dq * (8.0 / 6.0); // 2*dq*(4/6)
        gradient_q[idx + 2] += dq * (2.0 / 6.0);
    }

    if(hard_start_constraint_){
        for(int i = 0; i < 3; i++){
            gradient_q[i] = 0.0;
        }
    }
}

void optimizer_1d::calc_feasibility_cost(const vector<double> &q, const double &dt,
                                           double &cost, vector<double> &gradient_q)
{
    cost = 0.0;
    std::fill(gradient_q.begin(), gradient_q.end(), 0.0);

    // Abbreviation of params
    const double dt_inv = 1 / dt;
    const double dt_inv2 = dt_inv * dt_inv;

    for (int i = 0; i < q.size() - 1; ++i)
    {

        // Control point of velocity
        double vi = (q[i + 1] - q[i]) * dt_inv;

        double vd = fabs(vi) - max_speed_;
        if (vd > 0.0)
        {
            cost += (vd * vd);
            double sign = vi > 0 ? 1.0 : -1.0;
            double tmp = 2 * vd * sign * dt_inv;
            gradient_q[i] += -tmp;
            gradient_q[i + 1] += tmp;
        }

    }

    // Acc feasibility cost
    for (int i = 0; i < q.size() - 2; ++i)
    {
        double ai = (q[i + 2] - 2 * q[i + 1] + q[i]) * dt_inv2;
        
        double ad = fabs(ai) - max_acc_;
        if (ad > 0.0)
        {
            cost += (ad * ad);
            double sign = ai > 0 ? 1.0 : -1.0;
            double tmp = 2 * ad * sign * dt_inv2;
            gradient_q[i] += tmp;
            gradient_q[i + 1] += -2 * tmp;
            gradient_q[i + 2] += tmp;
        }
    }

    if(hard_start_constraint_){
        for(int i = 0; i < 3; i++){
            gradient_q[i] = 0.0;
        }
    }
}

void optimizer_1d::calc_vision_disturbance_cost(const vector<double> &q, double &vision_cost, vector<double> &gradient_vision, double &disturbance_cost, vector<double> &gradient_disturbance){

    vision_cost = 0.0;
    disturbance_cost = 0.0;

    std::fill(gradient_vision.begin(), gradient_vision.end(), 0.0);
    std::fill(gradient_disturbance.begin(), gradient_disturbance.end(), 0.0);

    if(w_vision_ <= 0.0 && w_disturbance_ <= 0.0){
        return;
    }

    double q1, q2, q3, knot, speed;
    
    for (int knot_idx = 0; knot_idx <= q.size() - 3; knot_idx++)
    {
        q1 = q[knot_idx];
        q2 = q[knot_idx + 1];
        q3 = q[knot_idx + 2];

        knot = 1 / 6.0 * (q1 + 4 * q2 + q3);
        speed = 1 / (2 * knot_span_) * (q3 - q1);

        double next_speed = speed + gradient_discrete_step_;

        double virtual_time = tunnel_center_line_->getTimeFromLength(knot);
        double virtual_yaw_dot = tunnel_center_yaw_->evaluateDeBoorT(virtual_time)(0) / VIRTUAL_FLIGHT_PROGRESS_SPEED;

        double yaw_dot = virtual_yaw_dot * speed;
        double next_yaw_dot = virtual_yaw_dot * next_speed;

        // cout<<"knot: "<<knot<<endl;

        unsigned int cs_idx_after = 0;
        while(cs_idx_after < cross_sections_ptr_->size() && knot > cross_sections_ptr_->at(cs_idx_after).curve_length_){
            cs_idx_after++;
        }


        if(cs_idx_after == 0){
            if(w_disturbance_ > 0.0){
                float pitch = asin(cross_sections_ptr_->front().w_R_cs.col(2).z());
                double cur_disturbance = compute_cross_section_disturbance(speed, pitch, cross_sections_ptr_->front());
                double next_disturbance = compute_cross_section_disturbance(next_speed, pitch, cross_sections_ptr_->front());
                disturbance_cost += cur_disturbance * cur_disturbance;

                double gradient_squared_disturbance = cur_disturbance * (next_disturbance - cur_disturbance) / gradient_discrete_step_;
                gradient_disturbance[knot_idx] -= gradient_squared_disturbance;
                gradient_disturbance[knot_idx+2] += gradient_squared_disturbance;
            }


            if(w_vision_ > 0.0){

                double cur_optical_flow = optical_flow_estimator_->cal_total_mean_optical_flow(0, speed, yaw_dot);
                vision_cost += cur_optical_flow * cur_optical_flow;

                double next_optical_flow = optical_flow_estimator_->cal_total_mean_optical_flow(0, next_speed, next_yaw_dot);

                double gradient_squared_optical_flow = cur_optical_flow * (next_optical_flow - cur_optical_flow) / gradient_discrete_step_;
                gradient_vision[knot_idx] -= gradient_squared_optical_flow;
                gradient_vision[knot_idx+2] += gradient_squared_optical_flow;
            }
            
        }
        else if(cs_idx_after == cross_sections_ptr_->size()){
            if(w_disturbance_ > 0.0){
                float pitch = asin(cross_sections_ptr_->back().w_R_cs.col(2).z());
                double cur_disturbance = compute_cross_section_disturbance(speed, pitch, cross_sections_ptr_->back());
                double next_disturbance = compute_cross_section_disturbance(next_speed, pitch, cross_sections_ptr_->back());
                disturbance_cost += cur_disturbance * cur_disturbance;

                double gradient_squared_disturbance = cur_disturbance * (next_disturbance - cur_disturbance) / gradient_discrete_step_;
                gradient_disturbance[knot_idx] -= gradient_squared_disturbance;
                gradient_disturbance[knot_idx+2] += gradient_squared_disturbance;
            }

            if(w_vision_ > 0.0){

                double cur_optical_flow = optical_flow_estimator_->cal_total_mean_optical_flow(cross_sections_ptr_->size()-1, speed, yaw_dot);
                vision_cost += cur_optical_flow * cur_optical_flow;

                double next_optical_flow = optical_flow_estimator_->cal_total_mean_optical_flow(cross_sections_ptr_->size()-1, next_speed, next_yaw_dot);

                double gradient_squared_optical_flow = cur_optical_flow * (next_optical_flow - cur_optical_flow) / gradient_discrete_step_;
                gradient_vision[knot_idx] -= gradient_squared_optical_flow;
                gradient_vision[knot_idx+2] += gradient_squared_optical_flow;
            }
        }
        else{

            auto cross_section_1 = &(cross_sections_ptr_->at(cs_idx_after-1));
            auto cross_section_2 = &(cross_sections_ptr_->at(cs_idx_after));

            double curve_length_1 = cross_section_1->curve_length_;
            double curve_length_2 = cross_section_2->curve_length_;

            float pitch1 = asin(cross_section_1->w_R_cs.col(2).z()), pitch2 = asin(cross_section_2->w_R_cs.col(2).z());

            double cross_section_1_ratio = (curve_length_2 - knot) / (curve_length_2 - curve_length_1);

            if(w_disturbance_ > 0.0){

                double cross_section_1_disturbance = compute_cross_section_disturbance(speed, pitch1, *cross_section_1);
                double cross_section_2_disturbance = compute_cross_section_disturbance(speed, pitch2, *cross_section_2);

                double cur_disturbance = cross_section_1_ratio * cross_section_1_disturbance + (1.0 - cross_section_1_ratio) * cross_section_2_disturbance;

                disturbance_cost += cur_disturbance * cur_disturbance;

                double next_cross_section_1_disturbance = compute_cross_section_disturbance(next_speed, pitch1, *cross_section_1);
                double next_cross_section_2_disturbance = compute_cross_section_disturbance(next_speed, pitch1, *cross_section_2);

                double next_disturbance = cross_section_1_ratio * next_cross_section_1_disturbance + (1.0 - cross_section_1_ratio) * next_cross_section_2_disturbance;

                double gradient_squared_disturbance = cur_disturbance * (next_disturbance - cur_disturbance) / gradient_discrete_step_;
                gradient_disturbance[knot_idx] -= gradient_squared_disturbance;
                gradient_disturbance[knot_idx+2] += gradient_squared_disturbance;

                
            }

            if(w_vision_ > 0.0){

                double cross_section_1_mean_optical_flow = optical_flow_estimator_->cal_total_mean_optical_flow(cs_idx_after-1, speed, yaw_dot);
                double cross_section_2_mean_optical_flow = optical_flow_estimator_->cal_total_mean_optical_flow(cs_idx_after, speed, yaw_dot);

                double cur_optical_flow = cross_section_1_ratio * cross_section_1_mean_optical_flow + (1.0 - cross_section_1_ratio) * cross_section_2_mean_optical_flow;           
                
                vision_cost += cur_optical_flow * cur_optical_flow;

                double next_cross_section_1_mean_optical_flow = optical_flow_estimator_->cal_total_mean_optical_flow(cs_idx_after-1, next_speed, next_yaw_dot);
                double next_cross_section_2_mean_optical_flow = optical_flow_estimator_->cal_total_mean_optical_flow(cs_idx_after, next_speed, next_yaw_dot);

                double next_optical_flow = cross_section_1_ratio * next_cross_section_1_mean_optical_flow + (1.0 - cross_section_1_ratio) * next_cross_section_2_mean_optical_flow;

                double gradient_squared_optical_flow = cur_optical_flow * (next_optical_flow - cur_optical_flow) / gradient_discrete_step_;
                gradient_vision[knot_idx] -= gradient_squared_optical_flow;
                gradient_vision[knot_idx+2] += gradient_squared_optical_flow;
            }
        }
    }

    if(hard_start_constraint_){
        for(int i = 0; i < 3; i++){
            gradient_vision[i] = 0.0;
            gradient_disturbance[i] = 0.0;
        }
    }

    vision_cost *= knot_span_;
    disturbance_cost *= knot_span_;

}


void optimizer_1d::combine_cost(const std::vector<double> &x, std::vector<double> &grad, double &f_combine)
{
    // Combine all cost functions
    g_q_ = x;

    const double dt = knot_span_;

    f_combine = 0.0;
    grad.resize(variable_num_);
    fill(grad.begin(), grad.end(), 0.0);

    double smooth_jerk_cost = 0.0;
    double smooth_acc_cost = 0.0;
    double smooth_speed_cost = 0.0;
    double start_cost = 0.0;
    double end_cost = 0.0;
    double endhard_cost = 0.0;
    double waypt_cost = 0.0;
    double feasi_cost = 0.0;
    double vision_cost = 0.0;
    double disturbance_cost = 0.0;

    

    if (cost_function_ & SMOOTHNESS_JERK)
    {
        double f_smoothness = 0.0;
        calc_smoothness_jerk_cost(g_q_, dt, f_smoothness, g_smoothness_jerk_);

        f_combine += w_smooth_jerk_ * f_smoothness;

        for (int i = 0; i < point_num_; i++)
            grad[i] += w_smooth_jerk_ * g_smoothness_jerk_[i];

        smooth_jerk_cost = f_smoothness;

    }

    if (cost_function_ & SMOOTHNESS_ACC)
    {
        double f_smoothness = 0.0;
        calc_smoothness_acc_cost(g_q_, dt, f_smoothness, g_smoothness_acc_);

        f_combine += w_smooth_acc_ * f_smoothness;

        for (int i = 0; i < point_num_; i++)
            grad[i] += w_smooth_acc_ * g_smoothness_acc_[i];

        smooth_acc_cost = f_smoothness;

    }

    if (cost_function_ & SMOOTHNESS_YAW_JERK)
    {
        double f_smoothness = 0.0;
        calc_smoothness_jerk_cost(g_q_, dt, f_smoothness, g_smoothness_jerk_);

        f_combine += w_smooth_yaw_ * f_smoothness;

        for (int i = 0; i < point_num_; i++)
            grad[i] += w_smooth_yaw_ * g_smoothness_jerk_[i];

        smooth_jerk_cost = f_smoothness;

    }

    if (cost_function_ & SMOOTHNESS_YAW_ACC)
    {
        double f_smoothness = 0.0;
        calc_smoothness_acc_cost(g_q_, dt, f_smoothness, g_smoothness_acc_);

        f_combine += w_smooth_yaw_ * f_smoothness;

        for (int i = 0; i < point_num_; i++)
            grad[i] += w_smooth_yaw_ * g_smoothness_acc_[i];

        smooth_acc_cost = f_smoothness;

    }

    if (cost_function_ & SMOOTHNESS_YAW_SPEED)
    {
        double f_smoothness = 0.0;
        calc_smoothness_speed_cost(g_q_, dt, f_smoothness, g_smoothness_speed_);

        f_combine += w_smooth_yaw_ * f_smoothness;

        for (int i = 0; i < point_num_; i++)
            grad[i] += w_smooth_yaw_ * g_smoothness_speed_[i];

        smooth_speed_cost = f_smoothness;

    }
    
    if (cost_function_ & START)
    {
        double f_start = 0.0;
        calc_start_cost(g_q_, dt, f_start, g_start_);
        f_combine += w_start_ * f_start;

        for (int i = 0; i < 3; i++)
            grad[i] += w_start_ * g_start_[i];

        start_cost = f_start;

    }

    if (cost_function_ & END)
    {
        double f_end = 0.0;
        calc_end_cost(g_q_, dt, f_end, g_end_);       
        f_combine += w_end_ * f_end;

        for (int i = point_num_ - 3; i < point_num_; i++)
            grad[i] += w_end_ * g_end_[i];

        end_cost = f_end;
    }

    if (cost_function_ & END_HARD)
    {
        double f_end = 0.0;
        calc_end_cost(g_q_, dt, f_end, g_end_);       
        f_combine += w_end_hard_ * f_end;

        for (int i = point_num_ - 3; i < point_num_; i++){
            grad[i] += w_end_hard_ * g_end_[i];
        }

        endhard_cost = f_end;
    }

    if (cost_function_ & WAYPOINTS)
    {
        double f_waypoints = 0.0;
        calc_waypoints_cost(g_q_, f_waypoints, g_waypoints_);
        f_combine += w_waypt_ * f_waypoints;

        for (int i = 0; i < point_num_; i++)
            grad[i] += w_waypt_ * g_waypoints_[i];

        waypt_cost = f_waypoints;
    }

    if (cost_function_ & FEASIBILITY)
    {
        double f_feasibility = 0.0;
        calc_feasibility_cost(g_q_, dt, f_feasibility, g_feasibility_);
        f_combine += w_feasi_ * f_feasibility;

        for (int i = 0; i < point_num_; i++)
            grad[i] += w_feasi_ * g_feasibility_[i];

        feasi_cost = f_feasibility;
    }

    if (cost_function_ & VISION || cost_function_ & AIR_FLOW_DISTURBANCE)
    {
        double f_vision = 0.0, f_disturbance = 0.0;
        calc_vision_disturbance_cost(g_q_, f_vision, g_vision_, f_disturbance, g_disturbance_);
        f_combine += (w_vision_ * f_vision + w_disturbance_ * f_disturbance);


        for (int i = 0; i < point_num_; i++){
            grad[i] += (w_vision_ * g_vision_[i] + w_disturbance_ * g_disturbance_[i]);
        }

        vision_cost = f_vision;
        disturbance_cost = f_disturbance;
    }

}

double optimizer_1d::cost_function(const std::vector<double> &x, std::vector<double> &grad, void *func_data)
{
    optimizer_1d *opt = reinterpret_cast<optimizer_1d*>(func_data);
    double cost;
    opt->combine_cost(x, grad, cost);
    opt->iter_num_++;

    /* save the min cost result */
    if (cost < opt->min_cost_)
    {
        opt->min_cost_ = cost;
        opt->best_variable_ = x;
        // std::cout << cost << ", ";
    }
    return cost;

}

}