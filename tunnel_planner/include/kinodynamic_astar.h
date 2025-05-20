/*******************************************************
Copyright (C) 2025, Aerial Robotics Group, Hong Kong University of Science and Technology
This file is part of FINT.
Licensed under the GNU General Public License v3.0;
you may not use this file except in compliance with the License.
*******************************************************/

#ifndef _KINODYNAMIC_ASTAR_H
#define _KINODYNAMIC_ASTAR_H

#include <Eigen/Eigen>
#include <iostream>
#include <map>
#include <set>
#include <unordered_set>
#include <ros/console.h>
#include <ros/ros.h>
#include <string>
#include <unordered_map>
#include <boost/functional/hash.hpp>
#include <queue>
#include <parameters.h>
#include <nlopt.hpp>

#include "tunnel_data_type.hpp"
#include "disturbance_estimator/regression_net.hpp"
#include "optical_flow_estimator/optical_flow_estimator.hpp"
#include "non_uniform_bspline.h"

using namespace std;
using namespace Eigen;

namespace tunnel_planner{

enum path_search_state{IN_OPEN, IN_CLOSE, NOT_EXPAND};

class path_node{

public:

    path_node(): parent_(nullptr), node_search_state_(NOT_EXPAND){

    }

    double f_score_, g_score_;

    double disturbance_cost, vision_cost;

    path_node* parent_;
    Vector2d state_;
    double prev_input_;
    // double next_input_;
    unsigned int path_step_;

    double time_from_parent_;
    double total_time_up_to_node_;

    int node_search_state_;

    private:
};


// class node_compare{
// public:
//   bool operator()(path_node* node1, path_node* node2) {
//     return node1->f_score_ > node2->f_score_;
//   }
// };

class path_node_compare{
public:
  bool operator()(path_node* node1, path_node* node2) {
    return node1->f_score_ < node2->f_score_;
  }
};

class optimizer_1d{
public:
    enum search_result_state{NO_PATH, SUCCESS};

    static const int SMOOTHNESS_JERK = (1 << 0);
    static const int SMOOTHNESS_ACC = (1 << 1);
    static const int SMOOTHNESS_YAW_JERK = (1 << 2);
    static const int SMOOTHNESS_YAW_ACC = (1 << 3);
    static const int SMOOTHNESS_YAW_SPEED = (1 << 4);
    static const int START = (1 << 5);
    static const int END = (1 << 6);
    static const int END_HARD = (1 << 7);
    static const int WAYPOINTS = (1 << 8);
    static const int FEASIBILITY = (1 << 9);
    static const int VISION = (1 << 10);
    static const int AIR_FLOW_DISTURBANCE = (1 << 11);
    static const int CONST_SPEED = (1 << 12);

    static const int WAY_PT_JERK_FEASI_PHASE = SMOOTHNESS_JERK | 
                                                   WAYPOINTS | 
                                                   FEASIBILITY |
                                                  //  START |
                                                   END_HARD;

    static const int WAY_PT_ACC_JERK_FEASI_PHASE = SMOOTHNESS_JERK | SMOOTHNESS_ACC |
                                                   WAYPOINTS | 
                                                   FEASIBILITY |
                                                  //  START |
                                                   END_HARD;                                              

    static const int WAY_PT_JERK_PHASE = SMOOTHNESS_JERK | 
                                                   WAYPOINTS | 
                                                  //  START |
                                                   END_HARD;

    static const int WAY_PT_YAW_JERK_PHASE = SMOOTHNESS_YAW_JERK | 
                                                   WAYPOINTS | 
                                                  //  START |
                                                   END_HARD;

    static const int WAY_PT_YAW_ACC_PHASE = SMOOTHNESS_YAW_ACC | 
                                                   WAYPOINTS | 
                                                  //  START |
                                                   END;

    static const int WAY_PT_YAW_SPEED_PHASE = SMOOTHNESS_YAW_SPEED | 
                                                   WAYPOINTS | 
                                                  //  START |
                                                   END;

    static const int WAY_PT_JERK_OF_AD_PHASE = SMOOTHNESS_JERK | WAYPOINTS |
                                                   VISION | AIR_FLOW_DISTURBANCE |
                                                   START|
                                                   END_HARD;

    static const int JERK_OF_AD_PHASE = SMOOTHNESS_JERK |
                                                   VISION | AIR_FLOW_DISTURBANCE |
                                                   START|
                                                   END_HARD;
    
    static const int JERK_OF_AD_START_END_HARD_PHASE = SMOOTHNESS_JERK |
                                                   VISION | AIR_FLOW_DISTURBANCE |
                                                   END_HARD;

    static const int ACC_JERK_OF_AD_START_END_HARD_PHASE = SMOOTHNESS_JERK | SMOOTHNESS_ACC |
                                                   VISION | AIR_FLOW_DISTURBANCE |
                                                   END_HARD;

    static const int JERK_FEASI_SPEED_PHASE = SMOOTHNESS_JERK |
                                                   FEASIBILITY |
                                                   CONST_SPEED |
                                                  //  START |
                                                   END_HARD;


    optimizer_1d(const shared_ptr<vector<cross_section>>& cross_sections, const shared_ptr<vector<vert_section>>& vert_sections, 
      const shared_ptr<NonUniformBspline>& tunnel_center_line, const shared_ptr<NonUniformBspline> tunnel_center_vel, const shared_ptr<NonUniformBspline> tunnel_center_acc, 
      const shared_ptr<NonUniformBspline> tunnel_center_yaw, const shared_ptr<NonUniformBspline> tunnel_center_yaw_dot, 
      const double& max_speed, const double& max_acc, 
      const int discrete_acc_size = 7, const double& time_res = 0.25, 
      const double& w_disturbance = 0.0, const double& w_vision = 0.0, const double& w_heuristic = 1.0, const double& w_time = 5.0, 
      const int max_node_num = 100000, const double max_search_time = 0.04,
      const shared_ptr<regression_net> circle_net = shared_ptr<regression_net>(nullptr), const shared_ptr<regression_net> rect_net = shared_ptr<regression_net>(nullptr), 
      const shared_ptr<optical_flow_estimator> of_estimator = shared_ptr<optical_flow_estimator>(nullptr)):
    cross_sections_ptr_(cross_sections), vert_sections_ptr_(vert_sections), 
    tunnel_center_line_(tunnel_center_line), tunnel_center_vel_(tunnel_center_vel), tunnel_center_acc_(tunnel_center_acc),
    tunnel_center_yaw_(tunnel_center_yaw), tunnel_center_yaw_dot_(tunnel_center_yaw_dot),
    max_speed_(max_speed), max_acc_(max_acc), 
    discrete_acc_size_(discrete_acc_size), time_res_(time_res), 
    w_disturbance_(w_disturbance), w_vision_(w_vision), w_heuristic_(w_heuristic), w_time_(w_time), 
    max_node_num_(max_node_num), max_search_time_(max_search_time),
    circle_net_(circle_net), rect_net_(rect_net), 
    optical_flow_estimator_(of_estimator)
    {
        inputs_ = VectorXd::LinSpaced(discrete_acc_size, -max_acc, max_acc);
        path_node_pool_.resize(max_node_num);
        close_states_.reserve(max_node_num);
        reset();

        if(w_vision <= 0.0){
          free_end_v_ = true;
          cout<<"FREE END SPEED!"<<endl;
        }
        else{
          free_end_v_ = false;
        }

            g_q_.reserve(100);
            g_smoothness_jerk_.reserve(100);
            g_smoothness_acc_.reserve(100);
            g_start_.reserve(100);
            g_feasibility_.reserve(100);
            g_end_.reserve(100);
            g_waypoints_.reserve(100);
            g_vision_.reserve(100);
            g_disturbance_.reserve(100);

        cout<<"max_speed: "<<max_speed_<<endl;
        cout<<"max_acc: "<<max_acc_<<endl;
        cout<<"w_disturbance: "<<w_disturbance<<endl;
    }

    void reset(){
        use_node_num_ = 0;
        iter_num_ = 0;
        open_set_.clear();
        close_states_.clear();
        search_result_.clear();
    }


    int astar_search(const double& start_v);
    int const_speed_search(const double& start_v, double& end_v);
    int const_dcc_search(const double& start_v);

    vector<double> get_init_sample(double& delta_t, double& end_v);

    void set_param(ros::NodeHandle &nh);
    void optimize_1d(Eigen::VectorXd &points, double &dt, const int &cost_function);
    void optimize_1d();
    void set_cost_function(const int &cost_function);
    void set_boundary_states(const vector<double> &start, const vector<double> &end);

    void set_des_speed(const double &des_speed){
      des_speed_ = des_speed;
    }

    void set_waypoints(const vector<double> &waypts, const vector<int> &waypt_idx); // N-2 constraints at most
    void set_waypoint_weight(const double &w_waypt){
      w_waypt_ = w_waypt;
    }

    double comb_time_;

private:

  // state: x,v
    double calculate_vision_disturbance_cost(const Eigen::Vector2d& x1, const Eigen::Vector2d& x2, const double& acc, double& vision_cost, double& disturbance_cost);
    double estimate_heuristic(Eigen::Vector2d& x1, Eigen::Vector2d& x2, double& optimal_time);
    vector<double> solve_cubic(double a, double b, double c, double d);
    vector<double> solve_quartic(double a, double b, double c, double d, double e);

    double compute_cross_section_disturbance(float speed, float pitch, const cross_section& cross_section);

    double get_curvature(const double& x);

    void retrieve_path(path_node* end_node_ptr);
    void state_transit(Vector2d& state0, Vector2d& state1, double um, double tau);


    static double cost_function(const std::vector<double> &x, std::vector<double> &grad, void *func_data);
    void combine_cost(const std::vector<double> &x, vector<double> &grad, double &cost);
    void add_gradient(vector<double> &grad, const double& weight, const vector<double>& g);

    bool isQuadratic();


    void calc_smoothness_jerk_cost(const vector<double> &q, const double &dt, double &cost, vector<double> &gradient_q);
    void calc_smoothness_acc_cost(const vector<double> &q, const double &dt, double &cost, vector<double> &gradient_q);
    void calc_smoothness_speed_cost(const vector<double> &q, const double &dt, double &cost, vector<double> &gradient_q);
    void calc_const_speed_cost(const vector<double> &q, const double &dt, double &cost, vector<double> &gradient_q);
    void calc_start_cost(const vector<double> &q, const double &dt, double &cost, vector<double> &gradient_q);
    void calc_end_cost(const vector<double> &q, const double &dt, double &cost, vector<double> &gradient_q);
    void calc_waypoints_cost(const vector<double> &q, double &cost, vector<double> &gradient_q);
    void calc_feasibility_cost(const vector<double> &q, const double &dt, double &cost, vector<double> &gradient_q);
    void calc_vision_disturbance_cost(const vector<double> &q, double &vision_cost, vector<double> &gradient_vision, double &disturbance_cost, vector<double> &gradient_disturbance);

    vector<path_node> path_node_pool_;
    int use_node_num_, max_node_num_, iter_num_;
    double max_search_time_;

    multiset<path_node*, path_node_compare> open_set_;
    vector<Vector2d> close_states_;


    const shared_ptr<NonUniformBspline> tunnel_center_line_;
    const shared_ptr<NonUniformBspline> tunnel_center_vel_; 
    const shared_ptr<NonUniformBspline> tunnel_center_acc_;

    const shared_ptr<NonUniformBspline> tunnel_center_yaw_; 
    const shared_ptr<NonUniformBspline> tunnel_center_yaw_dot_; 

    const shared_ptr<vector<cross_section>> cross_sections_ptr_;
    const shared_ptr<vector<vert_section>> vert_sections_ptr_;

    double w_time_;
    double tie_breaker_ = 1.0 + 1.0 / 10000;
    double w_smooth_jerk_, w_smooth_acc_, w_smooth_yaw_;
    double w_start_, w_end_, w_end_hard_;
    double w_waypt_;
    double w_feasi_;
    double w_disturbance_;
    double w_vision_;
    double w_heuristic_;

    double max_acc_;
    double max_speed_;

    bool free_end_v_;

    VectorXd inputs_;

    int discrete_acc_size_;
    double time_res_;

    vector<path_node*> search_result_;

    vector<double> start_state_, end_state_;

    vector<double> waypoints_; // waypts constraints
    vector<int> waypt_idx_;

    shared_ptr<regression_net> circle_net_, rect_net_;
    shared_ptr<optical_flow_estimator> optical_flow_estimator_;

    // Optimized variables
    Eigen::VectorXd control_points_; // B-spline control points, N x dim
    double knot_span_;               // B-spline knot span

    int cost_function_;

    int algorithm1_;               // optimization algorithms for quadratic cost
    int algorithm2_;               // optimization algorithms for general cost
    int max_iteration_num_;     // stopping criteria that can be used
    double max_iteration_time_; // stopping criteria that can be used

    bool hard_start_constraint_;

    // Data of opt
    vector<double> g_q_;
    vector<double> g_smoothness_jerk_, g_smoothness_acc_, g_smoothness_speed_, g_start_, g_end_, g_waypoints_, g_feasibility_, g_vision_, g_disturbance_;

    double gradient_discrete_step_ = 0.1;

    double pt_dist_;

    double des_speed_;

    unsigned int variable_num_; // optimization variables
    int point_num_;
    bool optimize_time_;
    
    int opt_iter_num_;                      // iteration of the solver
    std::vector<double> best_variable_; //
    double min_cost_;                   //

    unique_ptr<nlopt::opt> optimizer_;
};


}

#endif