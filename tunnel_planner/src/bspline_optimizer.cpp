/*******************************************************
Copyright (C) 2025, Aerial Robotics Group, Hong Kong University of Science and Technology
This file is part of FINT.
Licensed under the GNU General Public License v3.0;
you may not use this file except in compliance with the License.
*******************************************************/

#include "bspline_optimizer.h"
#include <thread>

// using namespace std;

void BsplineOptimizer::setParam(ros::NodeHandle &nh)
{

    algorithm1_ = QUAD_ALGORITHM_ID;
    algorithm2_ = NON_QUAD_ALGORITHM_ID;
    bspline_degree_ = BSPLINE_DEGREE;

    ld_smooth_acc_ = W_SMOOTH_3d;
    ld_smooth_jerk_ = W_SMOOTH_3d;
    ld_interval_ = W_FEASI;
    ld_dist_ = W_DIST;
    ld_start_ = W_START;
    ld_end_ = W_END;
    ld_end_hard_ = W_END_HARD;
    ld_guide_ = W_GUIDE;
    ld_waypt_ = W_WAYPT;
    ld_time_ = W_TIME;

    max_iteration_num_[0] = MAX_ITERATION_NUM1;
    max_iteration_num_[1] = MAX_ITERATION_NUM2;
    max_iteration_num_[2] = MAX_ITERATION_NUM3;
    max_iteration_num_[3] = MAX_ITERATION_NUM4;

    max_iteration_time_[0] = MAX_ITERATION_TIME1;
    max_iteration_time_[1] = MAX_ITERATION_TIME2;
    max_iteration_time_[2] = MAX_ITERATION_TIME3;
    max_iteration_time_[3] = MAX_ITERATION_TIME4;

    time_lb_ = -1; // Not used by in most case

    des_speed_ = VIRTUAL_FLIGHT_PROGRESS_SPEED;
}


void BsplineOptimizer::setEnvironment(const shared_ptr<tunnel_planner::edf_map_generator> &edf_map_ptr)
{
    this->edf_map_ptr_ = edf_map_ptr;
}

void BsplineOptimizer::setDist0(const double &dist0){
    this->dist0_ = dist0;
}

void BsplineOptimizer::setCostFunction(const int &cost_code)
{
    cost_function_ = cost_code;

    // print optimized cost function
    string cost_str;
    if (cost_function_ & SMOOTHNESS_ACC)
        cost_str += "smooth acc |";
    if (cost_function_ & SMOOTHNESS_JERK)
        cost_str += "smooth jerk |";
    if (cost_function_ & DISTANCE)
        cost_str += " dist  |";
    if (cost_function_ & INTERVAL)
        cost_str += " interval |";
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

    ROS_DEBUG_STREAM("cost func: " << cost_str);
}

void BsplineOptimizer::setGuidePath(const vector<Vector3d> &guide_pt)
{
    // guide_pts_ = guide_pt;
}

void BsplineOptimizer::setWaypoints(const vector<Vector3d> &waypts,
                                    const vector<int> &waypt_idx)
{
    waypoints_ = waypts;
    waypt_idx_ = waypt_idx;
}


void BsplineOptimizer::setBoundaryStates(const vector<Vector3d> &start,
                                         const vector<Vector3d> &end)
{
    start_state_ = start;
    end_state_ = end;
}

void BsplineOptimizer::setTimeLowerBound(const double &lb)
{
    time_lb_ = lb;
}

void BsplineOptimizer::optimize(Eigen::MatrixX3d &points, double &dt, const int &cost_function,
                                const int &max_num_id, const int &max_time_id)
{
    if (start_state_.empty())
    {
        ROS_ERROR("Initial state undefined!");
        return;
    }
    control_points_ = points;
    knot_span_ = dt;
    max_num_id_ = max_num_id;
    max_time_id_ = max_time_id;

    // Set necessary data and flag
    dim_ = 3;
    order_ = bspline_degree_;

    point_num_ = control_points_.rows();
    optimize_time_ = false;
    variable_num_ = optimize_time_ ? dim_ * point_num_ + 1 : dim_ * point_num_;
    if (variable_num_ <= 0)
    {
        ROS_ERROR("Empty varibale to optimization solver.");
        return;
    }

    setCostFunction(cost_function);


    pt_dist_ = 0.0;
    for (int i = 0; i < control_points_.rows() - 1; ++i)
    {
        pt_dist_ += (control_points_.row(i + 1) - control_points_.row(i)).norm();
    }
    pt_dist_ /= double(point_num_);

    iter_num_ = 0;
    min_cost_ = std::numeric_limits<double>::max();
    g_q_.resize(point_num_);
    g_smoothness_acc_.resize(point_num_);
    g_smoothness_jerk_.resize(point_num_);
    g_distance_.resize(point_num_);
    g_interval_.resize(point_num_);
    g_start_.resize(point_num_);
    g_end_.resize(point_num_);
    g_guide_.resize(point_num_);
    g_waypoints_.resize(point_num_);
    g_view_.resize(point_num_);
    g_time_.resize(point_num_);

    comb_time = 0.0;

    optimize();

    points = control_points_;
    dt = knot_span_;
    start_state_.clear();
    time_lb_ = -1;
}

void BsplineOptimizer::optimize()
{
    // Optimize all control points and maybe knot span dt
    // Use NLopt solver

    opt_ = make_unique<nlopt::opt>(nlopt::algorithm(isQuadratic() ? algorithm1_ : algorithm2_), variable_num_);

    opt_->set_min_objective(BsplineOptimizer::costFunction, this);

    opt_->set_maxeval(max_iteration_num_[max_num_id_]);
    opt_->set_maxtime(max_iteration_time_[max_time_id_]);

    opt_->set_xtol_rel(1e-6);


    vector<double> q(variable_num_);
    // Variables for control points
    for (int i = 0; i < point_num_; ++i)
        for (int j = 0; j < dim_; ++j)
        {
            double cij = control_points_(i, j);
            // if (dim_ != 1)
            //     cij = max(min(cij, bmax[j % 3]), bmin[j % 3]);
            q[dim_ * i + j] = cij;
        }
    // Variables for knot span
    if (optimize_time_)
        q[variable_num_ - 1] = knot_span_;


    vector<double> grad_temp;
    best_variable_ = q;
    combineCost(q, grad_temp, min_cost_);


    if (dim_ != 1)
    {

        vector<double> lb(variable_num_), ub(variable_num_);
        const double bound = 10.0;
        for (int i = 0; i < dim_ * point_num_; ++i)
        {
            lb[i] = q[i] - bound;
            ub[i] = q[i] + bound;
        }
        if (optimize_time_)
        {
            lb[variable_num_ - 1] = 0.0;
            ub[variable_num_ - 1] = 5.0;
        }
        opt_->set_lower_bounds(lb);
        opt_->set_upper_bounds(ub);

    }


    auto t1 = ros::Time::now();
    try
    {
        double final_cost;
        nlopt::result result = opt_->optimize(q, final_cost);
    }
    catch (std::exception &e)
    {
        // cout << e.what() << endl;
    }
    for (int i = 0; i < point_num_; ++i)
        for (int j = 0; j < dim_; ++j)
            control_points_(i, j) = best_variable_[dim_ * i + j];
    if (optimize_time_)
        knot_span_ = best_variable_[variable_num_ - 1];
}

void BsplineOptimizer::calcSmoothnessAccCost(const vector<Vector3d> &q, const double &dt,
                                          double &cost, vector<Vector3d> &gradient_q,
                                          double &gt)
{
    cost = 0.0;
    std::fill(gradient_q.begin(), gradient_q.end(), Vector3d::Zero());
    Vector3d temp_a;

    for (int i = 0; i < q.size() - 2; i++)
    {
        if(hard_start_constraint_ && i + 2 < order_){
            continue;
        }
        // acc cost
        Vector3d ai = (q[i + 2] - 2 * q[i + 1] + q[i]) / pt_dist_;
        cost += ai.squaredNorm();
        temp_a = 2 * ai / pt_dist_;

        gradient_q[i + 0] += temp_a;
        gradient_q[i + 1] += -2.0 * temp_a;
        gradient_q[i + 2] += temp_a;
    }

    if(hard_start_constraint_){
        for(int i = 0; i < order_; i++){
           gradient_q[i].setZero();
        }
    }
}

void BsplineOptimizer::calcSmoothnessJerkCost(const vector<Vector3d> &q, const double &dt,
                                          double &cost, vector<Vector3d> &gradient_q,
                                          double &gt)
{
    cost = 0.0;
    std::fill(gradient_q.begin(), gradient_q.end(), Vector3d::Zero());
    Vector3d temp_j;

    for (int i = 0; i < q.size() - 3; i++)
    {

        if(hard_start_constraint_ && i + 3 < order_){
            continue;
        }

        Vector3d ji = (q[i + 3] - 3 * q[i + 2] + 3 * q[i + 1] - q[i]) / pt_dist_;
        cost += ji.squaredNorm();
        temp_j = 2 * ji / pt_dist_;

        gradient_q[i + 0] += -temp_j;
        gradient_q[i + 1] += 3.0 * temp_j;
        gradient_q[i + 2] += -3.0 * temp_j;
        gradient_q[i + 3] += temp_j;

    }

    if(hard_start_constraint_){
        for(int i = 0; i < order_; i++){
           gradient_q[i].setZero();
        }
    }

}

void BsplineOptimizer::calcDistanceCost(const vector<Vector3d> &q, double &cost,
                                        vector<Vector3d> &gradient_q)
{
    cost = 0.0;
    Vector3d zero(0, 0, 0);
    std::fill(gradient_q.begin(), gradient_q.end(), zero);

    double dist;
    Vector3d dist_grad, g_zero(0, 0, 0);
    for (int i = 0; i < q.size(); i++)
    {

        dist = edf_map_ptr_->get_dist_grad(q[i], dist_grad);
        if (dist_grad.norm() > 1e-4)
            dist_grad.normalize();
        
        if (dist < dist0_)
        {
            cost += pow(dist - dist0_, 2);
            gradient_q[i] += 2.0 * (dist - dist0_) * dist_grad;
        }
    }

    if(hard_start_constraint_){
        for(int i = 0; i < order_; i++){
           gradient_q[i].setZero();
        }
    }
}

void BsplineOptimizer::calcIntervalCost(const vector<Vector3d> &q, const double &dt,
                                           double &cost, vector<Vector3d> &gradient_q,
                                           double &gt)
{
    cost = 0.0;

    std::fill(gradient_q.begin(), gradient_q.end(), Vector3d::Zero());
    gt = 0.0;

    // Abbreviation of params
    const double dt_inv = 1.0 / dt;
    const double dt_inv2 = dt_inv * dt_inv;

    for (int i = 0; i < q.size() - 1; ++i)
    {
        if(hard_start_constraint_ && i + 1 < order_){
            continue;
        }
        // Control point of velocity
        Vector3d vi = (q[i + 1] - q[i]) * dt_inv;

        double v_norm = vi.norm();
        double vd = v_norm - des_speed_;
        cost += pow(vd, 2);
        Vector3d tmp = 2.0 * dt_inv2 * (1.0 - des_speed_ / v_norm) * (q[i + 1] - q[i]);
        gradient_q[i] += (-tmp);
        gradient_q[i+1] += tmp;
    }

    if(hard_start_constraint_){
        for(int i = 0; i < order_; i++){
           gradient_q[i].setZero();
        }
    }

}

void BsplineOptimizer::calcStartCost(const vector<Vector3d> &q, const double &dt, double &cost,
                                     vector<Vector3d> &gradient_q, double &gt)
{
    cost = 0.0;
    std::fill(gradient_q.begin(), gradient_q.end(), Vector3d::Zero());
    gt = 0.0;

    Vector3d q1, q2, q3, dq;
    q1 = q[0];
    q2 = q[1];
    q3 = q[2];

    // Start position
    static const double w_pos = 10.0;
    dq = 1 / 6.0 * (q1 + 4 * q2 + q3) - start_state_[0];
    cost += w_pos * dq.squaredNorm();
    gradient_q[0] += w_pos * 2 * dq * (1 / 6.0);
    gradient_q[1] += w_pos * 2 * dq * (4 / 6.0);
    gradient_q[2] += w_pos * 2 * dq * (1 / 6.0);

    // Start velocity
    dq = 1 / (2 * dt) * (q3 - q1) - start_state_[1];
    cost += dq.squaredNorm();
    gradient_q[0] += 2 * dq * (-1.0) / (2 * dt);
    gradient_q[2] += 2 * dq * 1.0 / (2 * dt);
    if (optimize_time_)
        gt += dq.dot(q3 - q1) / (-dt * dt);

    // Start acceleration
    dq = 1 / (dt * dt) * (q1 - 2 * q2 + q3) - start_state_[2];
    cost += dq.squaredNorm();
    gradient_q[0] += 2 * dq * 1.0 / (dt * dt);
    gradient_q[1] += 2 * dq * (-2.0) / (dt * dt);
    gradient_q[2] += 2 * dq * 1.0 / (dt * dt);
    if (optimize_time_)
        gt += dq.dot(q1 - 2 * q2 + q3) / (-dt * dt * dt);
}

void BsplineOptimizer::calcEndCost(const vector<Vector3d> &q, const double &dt, double &cost,
                                   vector<Vector3d> &gradient_q, double &gt)
{
    cost = 0.0;
    std::fill(gradient_q.begin(), gradient_q.end(), Vector3d::Zero());
    gt = 0.0;

    Vector3d q_3, q_2, q_1, dq;
    q_3 = q[q.size() - 3];
    q_2 = q[q.size() - 2];
    q_1 = q[q.size() - 1];

    // End position
    dq = 1 / 6.0 * (q_1 + 4 * q_2 + q_3) - end_state_[0];
    cost += dq.squaredNorm();
    gradient_q[q.size() - 1] += 2 * dq * (1 / 6.0);
    gradient_q[q.size() - 2] += 2 * dq * (4 / 6.0);
    gradient_q[q.size() - 3] += 2 * dq * (1 / 6.0);

    if (end_state_.size() >= 2)
    {
        // End velocity
        dq = 1 / (2 * dt) * (q_1 - q_3) - end_state_[1];
        cost += dq.squaredNorm();
        gradient_q[q.size() - 1] += 2 * dq * 1.0 / (2 * dt);
        gradient_q[q.size() - 3] += 2 * dq * (-1.0) / (2 * dt);
        if (optimize_time_)
            gt += dq.dot(q_1 - q_3) / (-dt * dt);
    }
    if (end_state_.size() == 3)
    {
        // End acceleration
        dq = 1 / (dt * dt) * (q_1 - 2 * q_2 + q_3) - end_state_[2];
        cost += dq.squaredNorm();
        gradient_q[q.size() - 1] += 2 * dq * 1.0 / (dt * dt);
        gradient_q[q.size() - 2] += 2 * dq * (-2.0) / (dt * dt);
        gradient_q[q.size() - 3] += 2 * dq * 1.0 / (dt * dt);
        if (optimize_time_)
            gt += dq.dot(q_1 - 2 * q_2 + q_3) / (-dt * dt * dt);
    }

    if(hard_start_constraint_){
        for(int i = 0; i < order_; i++){
           gradient_q[i].setZero();
        }
    }
}

void BsplineOptimizer::calcWaypointsCost(const vector<Vector3d> &q, double &cost,
                                         vector<Vector3d> &gradient_q)
{
    cost = 0.0;
    std::fill(gradient_q.begin(), gradient_q.end(), Vector3d::Zero());

    Vector3d q1, q2, q3, dq;

    for (int i = 0; i < waypoints_.size(); ++i)
    {

        if(hard_start_constraint_ && i + 2 < order_){
            continue;
        }

        Vector3d waypt = waypoints_[i];
        int idx = waypt_idx_[i];

        q1 = q[idx];
        q2 = q[idx + 1];
        q3 = q[idx + 2];

        dq = 1 / 6.0 * (q1 + 4 * q2 + q3) - waypt;
        cost += dq.squaredNorm();

        gradient_q[idx] += dq * (2.0 / 6.0);     // 2*dq*(1/6)
        gradient_q[idx + 1] += dq * (8.0 / 6.0); // 2*dq*(4/6)
        gradient_q[idx + 2] += dq * (2.0 / 6.0);
    }

    if(hard_start_constraint_){
        for(int i = 0; i < order_; i++){
           gradient_q[i].setZero();
        }
    }
}


void BsplineOptimizer::combineCost(const std::vector<double> &x, std::vector<double> &grad,
                                   double &f_combine)
{
    /* Convert the NLopt format vector to control points. */


    retrieveCtrlPts(g_q_, x);

    const double dt = optimize_time_ ? x[variable_num_ - 1] : knot_span_;

    f_combine = 0.0;
    grad.resize(variable_num_);
    fill(grad.begin(), grad.end(), 0.0);
    

    if (cost_function_ & SMOOTHNESS_ACC)
    {

        double f_smoothness = 0.0, gt_smoothness = 0.0;
        calcSmoothnessAccCost(g_q_, dt, f_smoothness, g_smoothness_acc_, gt_smoothness);

        f_combine += ld_smooth_acc_ * f_smoothness;
        addGradient(grad, ld_smooth_acc_, g_smoothness_acc_);

        if (optimize_time_)
            grad[variable_num_ - 1] += ld_smooth_acc_ * gt_smoothness;
        
    }
    if (cost_function_ & SMOOTHNESS_JERK)
    {
        double f_smoothness = 0.0, gt_smoothness = 0.0;
        calcSmoothnessJerkCost(g_q_, dt, f_smoothness, g_smoothness_jerk_, gt_smoothness);
        f_combine += ld_smooth_jerk_ * f_smoothness;
        addGradient(grad, ld_smooth_jerk_, g_smoothness_jerk_);

        if (optimize_time_)
            grad[variable_num_ - 1] += ld_smooth_jerk_ * gt_smoothness;
    }
    if (cost_function_ & DISTANCE)
    {
        double f_distance = 0.0;
        calcDistanceCost(g_q_, f_distance, g_distance_);
        f_combine += ld_dist_ * f_distance;
        addGradient(grad, ld_dist_, g_distance_);

    }
    if (cost_function_ & INTERVAL)
    {
        double f_interval = 0.0, gt_interval = 0.0;
        calcIntervalCost(g_q_, dt, f_interval, g_interval_, gt_interval);
        f_combine += ld_interval_ * f_interval;
        addGradient(grad, ld_interval_, g_interval_);

        if (optimize_time_)
            grad[variable_num_ - 1] += ld_interval_ * gt_interval;
    }
    if (cost_function_ & START)
    {
        double f_start = 0.0, gt_start = 0.0;
        calcStartCost(g_q_, dt, f_start, g_start_, gt_start);
        f_combine += ld_start_ * f_start;
        addGradient(grad, ld_start_, g_start_);

        if (optimize_time_)
            grad[variable_num_ - 1] += ld_start_ * gt_start;
    }
    if (cost_function_ & END)
    {
        double f_end = 0.0, gt_end = 0.0;
        calcEndCost(g_q_, dt, f_end, g_end_, gt_end);       
        f_combine += ld_end_ * f_end;
        addGradient(grad, ld_end_, g_end_);

        if (optimize_time_)
            grad[variable_num_ - 1] += ld_end_ * gt_end;
    }
    if (cost_function_ & END_HARD)
    {
        double f_end = 0.0, gt_end = 0.0;
        calcEndCost(g_q_, dt, f_end, g_end_, gt_end);
        f_combine += ld_end_hard_ * f_end;
        addGradient(grad, ld_end_hard_, g_end_);

        if (optimize_time_)
            grad[variable_num_ - 1] += ld_end_hard_ * gt_end;
    }

    if (cost_function_ & WAYPOINTS)
    {
        double f_waypoints = 0.0;
        calcWaypointsCost(g_q_, f_waypoints, g_waypoints_);
        f_combine += ld_waypt_ * f_waypoints;
        addGradient(grad, ld_waypt_, g_waypoints_);

    }

}

double BsplineOptimizer::costFunction(const std::vector<double> &x, std::vector<double> &grad,
                                      void *func_data)
{
    BsplineOptimizer *opt = reinterpret_cast<BsplineOptimizer *>(func_data);
    double cost;
    opt->combineCost(x, grad, cost);
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

vector<Vector3d> BsplineOptimizer::matrixToVectors(const Eigen::MatrixXd &ctrl_pts)
{
    vector<Vector3d> ctrl_q;
    for (int i = 0; i < ctrl_pts.rows(); ++i)
    {
        ctrl_q.push_back(ctrl_pts.row(i));
    }
    return ctrl_q;
}

Eigen::MatrixX3d BsplineOptimizer::getControlPoints()
{
    return this->control_points_;
}

bool BsplineOptimizer::isQuadratic()
{
    if (cost_function_ == WAY_PT_JERK_PHASE || cost_function_ == WAY_PT_JERK_VEL_PHASE || cost_function_ == WAY_PT_JERK_VEL_START_HARD_PHASE)
    {
        return true;
    }
    else if (cost_function_ == WAY_PT_ACC_VEL_PHASE)
    {
        return true;
    }
    else if (cost_function_ == SMOOTHNESS_ACC || cost_function_ == SMOOTHNESS_JERK)
    {
        return true;
    }
    else if (cost_function_ == (SMOOTHNESS_JERK | WAYPOINTS))
    {
        return true;
    }

    return false;
}

void BsplineOptimizer::retrieveCtrlPts(vector<Vector3d> &g_q, const std::vector<double> &x){
    
    for (int i = 0; i < point_num_; ++i)
    {
        for (int j = 0; j < dim_; ++j)
            g_q_[i][j] = x[dim_ * i + j];
        for (int j = dim_; j < 3; ++j)
            g_q_[i][j] = 0.0;
    }
}

 void BsplineOptimizer::addGradient(vector<double> &grad, const double& weight, const vector<Vector3d>& g){
    for (int i = 0; i < point_num_; i++){
        for (int j = 0; j < dim_; j++){
            grad[dim_ * i + j] += weight * g[i](j);       
        }
    }
}

