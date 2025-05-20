/*******************************************************
Copyright (C) 2025, Aerial Robotics Group, Hong Kong University of Science and Technology
This file is part of FINT.
Licensed under the GNU General Public License v3.0;
you may not use this file except in compliance with the License.
*******************************************************/

#ifndef _BSPLINE_OPTIMIZER_H_
#define _BSPLINE_OPTIMIZER_H_

#include <Eigen/Eigen>
#include <nlopt.hpp>
#include <edf_map/edf_map_generator.h>

// Gradient and elasitc band optimization

// Input: a signed distance field and a sequence of points
// Output: the optimized sequence of points
// The format of points: N x 3 matrix, each row is a point

class BsplineOptimizer
{
public:
    static const int SMOOTHNESS_ACC  = (1 << 0);
    static const int SMOOTHNESS_JERK = (1 << 1);
    static const int DISTANCE = (1 << 2);
    static const int INTERVAL = (1 << 3);
    static const int START = (1 << 4);
    static const int END = (1 << 5);
    static const int END_HARD = (1 << 6);
    static const int WAYPOINTS = (1 << 7);

    static const int WAY_PT_JERK_PHASE = SMOOTHNESS_JERK | WAYPOINTS |
                                                   START|
                                                   END_HARD;

    static const int WAY_PT_JERK_VEL_PHASE = SMOOTHNESS_JERK | WAYPOINTS |
                                                   INTERVAL |
                                                   START|
                                                   END;

    static const int WAY_PT_JERK_VEL_START_HARD_PHASE = SMOOTHNESS_JERK | WAYPOINTS |
                                                   INTERVAL | DISTANCE |
                                                   END;

    static const int WAY_PT_ACC_VEL_START_HARD_PHASE = SMOOTHNESS_ACC | WAYPOINTS |
                                                   INTERVAL |
                                                   END;

    static const int WAY_PT_ACC_VEL_PHASE = SMOOTHNESS_ACC | WAYPOINTS |
                                                  INTERVAL |
                                               START | 
                                               END;
    static const int WAY_PT_PHASE;
    static const int NORMAL_PHASE = SMOOTHNESS_JERK | DISTANCE |
                                           INTERVAL | START | END;
    static const int WAY_PT_ACC_PHASE = SMOOTHNESS_ACC | WAYPOINTS;;

    BsplineOptimizer()
    {
    }
    ~BsplineOptimizer()
    {
    }

    /* main API */
    void setEnvironment(const shared_ptr<tunnel_planner::edf_map_generator> &edf_map_ptr);
    void setDist0(const double &dist0);
    void setParam(ros::NodeHandle &nh);
    void optimize(Eigen::MatrixX3d &points, double &dt, const int &cost_function, const int &max_num_id,
                  const int &max_time_id);

    /* helper function */

    // required inputs
    void setCostFunction(const int &cost_function);
    void setBoundaryStates(const vector<Vector3d> &start, const vector<Vector3d> &end);
    void setTimeLowerBound(const double &lb);

    // optional inputs
    void setGuidePath(const vector<Vector3d> &guide_pt);
    void setWaypoints(const vector<Vector3d> &waypts,
                      const vector<int> &waypt_idx); // N-2 constraints at most

    void optimize();

    Eigen::MatrixX3d getControlPoints();
    vector<Vector3d> matrixToVectors(const Eigen::MatrixXd &ctrl_pts);

private:
    // Wrapper of cost function
    static double costFunction(const std::vector<double> &x, std::vector<double> &grad, void *func_data);
    void combineCost(const std::vector<double> &x, vector<double> &grad, double &cost);

    // Cost functions, q: control points, dt: knot span
    void calcSmoothnessAccCost(const vector<Vector3d> &q, const double &dt, double &cost,
                            vector<Vector3d> &gradient_q, double &gt);
    void calcSmoothnessJerkCost(const vector<Vector3d> &q, const double &dt, double &cost,
                               vector<Vector3d> &gradient_q, double &gt);
    void calcDistanceCost(const vector<Vector3d> &q, double &cost,
                          vector<Vector3d> &gradient_q);
    void calcIntervalCost(const vector<Vector3d> &q, const double &dt, double &cost,
                             vector<Vector3d> &gradient_q, double &gt);
    void calcStartCost(const vector<Vector3d> &q, const double &dt, double &cost,
                       vector<Vector3d> &gradient_q, double &gt);
    void calcEndCost(const vector<Vector3d> &q, const double &dt, double &cost,
                     vector<Vector3d> &gradient_q, double &gt);
    void calcWaypointsCost(const vector<Vector3d> &q, double &cost,
                           vector<Vector3d> &gradient_q);

    bool isQuadratic();

    void retrieveCtrlPts(vector<Vector3d> &g_q, const std::vector<double> &x);

    void addGradient(vector<double> &grad, const double& weight, const vector<Vector3d>& g);

    shared_ptr<tunnel_planner::edf_map_generator> edf_map_ptr_;

    unique_ptr<nlopt::opt> opt_;

    // Optimized variables
    Eigen::MatrixX3d control_points_; // B-spline control points, N x dim
    double knot_span_;               // B-spline knot span

    // Input to solver
    int dim_; // dimension of the B-spline
    vector<Vector3d> start_state_, end_state_;
    vector<Vector3d> guide_pts_; // geometric guiding path points, N-6
    vector<Vector3d> waypoints_; // waypts constraints
    vector<int> waypt_idx_;
    int max_num_id_, max_time_id_; // stopping criteria
    int cost_function_;
    double time_lb_;

    /* Parameters of optimization  */
    bool hard_start_constraint_;
    int order_; // bspline degree
    int bspline_degree_;
    double ld_smooth_acc_, ld_smooth_jerk_, ld_dist_, ld_interval_, ld_start_, ld_end_, ld_end_hard_, ld_guide_, ld_waypt_, ld_time_;
    double dist0_;             // safe distance
    double des_speed_, max_acc_; // dynamic limits
    double wnl_, dlmin_;
    int algorithm1_;               // optimization algorithms for quadratic cost
    int algorithm2_;               // optimization algorithms for general cost
    int max_iteration_num_[4];     // stopping criteria that can be used
    double max_iteration_time_[4]; // stopping criteria that can be used

    // Data of opt
    vector<Vector3d> g_q_, g_smoothness_acc_, g_smoothness_jerk_, g_distance_, g_interval_, g_start_, g_end_, g_guide_,
        g_waypoints_, g_view_, g_time_;

    unsigned int variable_num_; // optimization variables
    int point_num_;
    bool optimize_time_;
    int iter_num_;                      // iteration of the solver
    std::vector<double> best_variable_; //
    double min_cost_;                   //
    double pt_dist_;

    /* for benckmark evaluation only */
public:
    vector<double> vec_cost_;
    vector<double> vec_time_;
    ros::Time time_start_;

    void getCostCurve(vector<double> &cost, vector<double> &time)
    {
        cost = vec_cost_;
        time = vec_time_;
    }

    double comb_time;

    typedef unique_ptr<BsplineOptimizer> Ptr;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif
