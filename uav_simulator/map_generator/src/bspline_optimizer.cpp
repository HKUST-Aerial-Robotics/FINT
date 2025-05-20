#include "bspline_optimizer.h"
#include <thread>

// using namespace std;

void BsplineOptimizer::setParam(double des_speed){

    algorithm1_ = 11;
    algorithm2_ = 11;
    bspline_degree_ = 3;

    ld_smooth_acc_ = 1.0;
    ld_smooth_jerk_ = 0.5;
    ld_interval_ = 3.0;
    ld_start_ = 5.0;
    ld_end_ = 5.0;
    ld_end_hard_ = 5.0;
    ld_guide_ = 1.0;
    ld_waypt_ = 20.0;
    ld_time_ = 1.0;

    max_iteration_num_[0] = 1000;
    max_iteration_num_[1] = 1000;
    max_iteration_num_[2] = 1000;
    max_iteration_num_[3] = 1000;

    max_iteration_time_[0] = 0.5;
    max_iteration_time_[1] = 0.5;
    max_iteration_time_[2] = 0.5;
    max_iteration_time_[3] = 0.5;

    time_lb_ = -1; // Not used by in most case

    des_speed_ = des_speed;
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
    if (cost_function_ & MINTIME)
        cost_str += " time  |";

    ROS_INFO_STREAM("cost func: " << cost_str);
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
    // dim_ = control_points_.cols();
    dim_ = 3;
    order_ = bspline_degree_;
    // if (dim_ == 1)
    //     order_ = 3;
    // else
    //     order_ = bspline_degree_;
    point_num_ = control_points_.rows();
    optimize_time_ = cost_function_ & MINTIME;
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
    // cout<<"prepare to opt"<<endl;
    // cout<<"variable_num:"<<variable_num_<<endl;

    opt_.reset(new nlopt::opt(nlopt::algorithm(isQuadratic() ? algorithm1_ : algorithm2_), variable_num_));
    // nlopt::opt opt(nlopt::algorithm(11), variable_num_);
    // cout<<"nlopt"<<endl;

    opt_->set_min_objective(BsplineOptimizer::costFunction, this);

    opt_->set_maxeval(max_iteration_num_[max_num_id_]);
    opt_->set_maxtime(max_iteration_time_[max_time_id_]);
    // cout<<"nlopt stop criteria"<<endl;

    opt_->set_xtol_rel(1e-6);

    // cout<<"prepare opt finish"<<endl;


    // Set axis aligned bounding box for optimization
    // Vector3d bmin, bmax;
    // edt_environment_->sdf_map_->getBox(bmin, bmax);
    // for (int k = 0; k < 3; ++k)
    // {
    //     bmin[k] += 0.1;
    //     bmax[k] -= 0.1;
    // }


    // Deprecated: does not optimize start and end control points
    // for (int i = order_; i < pt_num; ++i)
    // {
    //   if (!(cost_function_ & BOUNDARY) && i >= pt_num - order_)
    //     continue;
    //   for (int j = 0; j < dim_; j++)
    //   {
    //     double cij = control_points_(i, j);
    //     if (dim_ != 1)
    //       cij = max(min(cij, bmax[j % 3]), bmin[j % 3]);
    //     q[dim_ * (i - order_) + j] = cij;
    //   }
    // }

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


    if (dim_ != 1)
    {

        vector<double> lb(variable_num_), ub(variable_num_);
        const double bound = 10.0;
        for (int i = 0; i < dim_ * point_num_; ++i)
        {
            lb[i] = q[i] - bound;
            ub[i] = q[i] + bound;
            // lb[i] = max(lb[i], bmin[i % 3]);
            // ub[i] = min(ub[i], bmax[i % 3]);
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
        ROS_WARN("final cost: %lf", final_cost);
    }
    catch (std::exception &e)
    {
        cout << e.what() << endl;
    }
    for (int i = 0; i < point_num_; ++i)
        for (int j = 0; j < dim_; ++j)
            control_points_(i, j) = best_variable_[dim_ * i + j];
    if (optimize_time_)
        knot_span_ = best_variable_[variable_num_ - 1];

    if (cost_function_ & MINTIME)
    {
        std::cout << "Iter num: " << iter_num_ << ", time: " << (ros::Time::now() - t1).toSec()
                  << ", point num: " << point_num_ << ", comb time: " << comb_time << std::endl;
    }

    // Deprecated
    // for (int i = order_; i < control_points_.rows(); ++i)
    // {
    //   if (!(cost_function_ & BOUNDARY) && i >= pt_num - order_)
    //     continue;
    //   for (int j = 0; j < dim_; j++)
    //   {
    //     control_points_(i, j) = best_variable_[dim_ * (i - order_) + j];
    //   }
    // }

    // if (!(cost_function_ & GUIDE))
    //   ROS_INFO_STREAM("iter num: " << iter_num_);
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
        // Test jerk cost
        Vector3d ji = (q[i + 3] - 3 * q[i + 2] + 3 * q[i + 1] - q[i]) / pt_dist_;
        cost += ji.squaredNorm();
        temp_j = 2 * ji / pt_dist_;

        // /* evaluate jerk */
        // jerk = q[i + 3] - 3 * q[i + 2] + 3 * q[i + 1] - q[i];
        // cost += jerk.squaredNorm();
        // temp_j = 2.0 * jerk;
        // /* jerk gradient_q */

        gradient_q[i + 0] += -temp_j;
        gradient_q[i + 1] += 3.0 * temp_j;
        gradient_q[i + 2] += -3.0 * temp_j;
        gradient_q[i + 3] += temp_j;
        // if (optimize_time_)
        //   gt += -6 * ji.dot(ji) / dt;
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
    // cost = 0.0;
    // Vector3d zero(0, 0, 0);
    // std::fill(gradient_q.begin(), gradient_q.end(), zero);

    // double dist;
    // Vector3d dist_grad, g_zero(0, 0, 0);
    // for (int i = 0; i < q.size(); i++)
    // {
    //     if (!dynamic_)
    //     {
    //         edt_environment_->evaluateEDTWithGrad(q[i], -1.0, dist, dist_grad);
    //         if (dist_grad.norm() > 1e-4)
    //             dist_grad.normalize();
    //     }
    //     else
    //     {
    //         double time = double(i + 2 - order_) * knot_span_ + start_time_;
    //         edt_environment_->evaluateEDTWithGrad(q[i], time, dist, dist_grad);
    //     }

    //     if (dist < dist0_)
    //     {
    //         cost += pow(dist - dist0_, 2);
    //         gradient_q[i] += 2.0 * (dist - dist0_) * dist_grad;
    //     }
    // }
}

void BsplineOptimizer::calcIntervalCost(const vector<Vector3d> &q, const double &dt,
                                           double &cost, vector<Vector3d> &gradient_q,
                                           double &gt)
{
    cost = 0.0;
    // std::fill(gradient_q.begin(), gradient_q.end(), Vector3d::Zero());
    std::fill(gradient_q.begin(), gradient_q.end(), Vector3d::Zero());
    gt = 0.0;

    // Abbreviation of params
    const double dt_inv = 1 / dt;
    const double dt_inv2 = dt_inv * dt_inv;

    for (int i = 0; i < q.size() - 1; ++i)
    {
        // Control point of velocity
        Vector3d vi = (q[i + 1] - q[i]) * dt_inv;
        // for (int k = 0; k < 3; ++k)
        // {
        //     // Calculate cost for each axis
        //     double vd = fabs(vi[k]) - des_vel_;
        //     if (vd > 0.0)
        //     {
        //         cost += pow(vd, 2);
        //         double sign = vi[k] > 0 ? 1.0 : -1.0;
        //         double tmp = 2 * vd * sign * dt_inv;
        //         gradient_q[i][k] += -tmp;
        //         gradient_q[i + 1][k] += tmp;
        //         if (optimize_time_)
        //             gt += tmp * (-vi[k]);
        //     }
        // }

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

    // Acc feasibility cost
    // for (int i = 0; i < q.size() - 2; ++i)
    // {
    //     Vector3d ai = (q[i + 2] - 2 * q[i + 1] + q[i]) * dt_inv2;
    //     for (int k = 0; k < 3; ++k)
    //     {
    //         double ad = fabs(ai[k]) - max_acc_;
    //         if (ad > 0.0)
    //         {
    //             cost += pow(ad, 2);
    //             double sign = ai[k] > 0 ? 1.0 : -1.0;
    //             double tmp = 2 * ad * sign * dt_inv2;
    //             gradient_q[i][k] += tmp;
    //             gradient_q[i + 1][k] += -2 * tmp;
    //             gradient_q[i + 2][k] += tmp;
    //             if (optimize_time_)
    //                 gt += tmp * ai[k] * (-2) * dt;
    //         }
    //     }
    // }
}

void BsplineOptimizer::calcStartCost(const vector<Vector3d> &q, const double &dt, double &cost,
                                     vector<Vector3d> &gradient_q, double &gt)
{
    cost = 0.0;
    std::fill(gradient_q.begin(), gradient_q.begin()+3, Vector3d::Zero());
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
    std::fill(gradient_q.end()-3, gradient_q.end(), Vector3d::Zero());
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

    // for (auto wp : waypoints_) {
    for (int i = 0; i < waypoints_.size(); ++i)
    {
        Vector3d waypt = waypoints_[i];
        int idx = waypt_idx_[i];

        // if(idx < order_) continue;

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

/* use the uniformly sampled points on a geomertic path to guide the
 * trajectory. For each control points to be optimized, it is assigned a
 * guiding point on the path and the distance between them is penalized */
void BsplineOptimizer::calcGuideCost(const vector<Vector3d> &q, double &cost,
                                     vector<Vector3d> &gradient_q)
{
    cost = 0.0;
    std::fill(gradient_q.begin(), gradient_q.end(), Vector3d::Zero());

    int end_idx = q.size() - order_;

    for (int i = order_; i < end_idx; i++)
    {
        Vector3d gpt = guide_pts_[i - order_];
        cost += (q[i] - gpt).squaredNorm();
        gradient_q[i] += 2 * (q[i] - gpt);
    }

    if(hard_start_constraint_){
        for(int i = 0; i < order_; i++){
           gradient_q[i].setZero();
        }
    }
}

void BsplineOptimizer::calcViewCost(const vector<Vector3d> &q, double &cost,
                                    vector<Vector3d> &gradient_q)
{
    // cost = 0.0;
    // Vector3d zero(0, 0, 0);
    // std::fill(gradient_q.begin(), gradient_q.end(), zero);
    // Vector3d p = view_cons_.pt_;
    // Vector3d v = view_cons_.dir_.normalized();
    // Eigen::Matrix3d vvT = v * v.transpose();
    // Eigen::Matrix3d I_vvT = Eigen::Matrix3d::Identity() - vvT;

    // // prependicular cost, increase visibility of points before blocked point
    // int i = view_cons_.idx_;
    // Vector3d dn = (q[i] - p) - ((q[i] - p).dot(v)) * v;
    // cost += dn.squaredNorm();
    // gradient_q[i] += 2 * I_vvT * dn;
    // double norm_dn = dn.norm();

    // // parallel cost, increase projection along view direction
    // Vector3d dl = ((q[i] - p).dot(v)) * v;
    // double norm_dl = dl.norm();
    // double safe_dist = view_cons_.dir_.norm();
    // if (norm_dl < safe_dist)
    // {
    //     cost += wnl_ * pow(norm_dl - safe_dist, 2);
    //     gradient_q[i] += wnl_ * 2 * (norm_dl - safe_dist) * vvT * dl / norm_dl;
    // }
}

void BsplineOptimizer::calcTimeCost(const double &dt, double &cost, double &gt)
{
    // Min time
    double duration = (point_num_ - order_) * dt;
    cost = duration;
    gt = double(point_num_ - order_);

    // Time lower bound
    if (time_lb_ > 0 && duration < time_lb_)
    {
        static const double w_lb = 10;
        cost += w_lb * pow(duration - time_lb_, 2);
        gt += w_lb * 2 * (duration - time_lb_) * (point_num_ - order_);
    }
}


void BsplineOptimizer::combineCost(const std::vector<double> &x, std::vector<double> &grad,
                                   double &f_combine)
{
    {
        /* Convert the NLopt format vector to control points. */

        // This solver can support 1D-3D B-spline optimization, but we use Vector3d to store
        // each control point
        // For 1D case, the second and third elements are zero, and similar for the 2D case.

        // Deprecated
        // for (int i = 0; i < order_; i++)
        // {
        //   for (int j = 0; j < dim_; ++j)
        //     g_q_[i][j] = control_points_(i, j);
        //   for (int j = dim_; j < 3; ++j)
        //     g_q_[i][j] = 0.0;
        // }

        // for (int i = 0; i < variable_num_ / dim_; i++)
        // {
        //   for (int j = 0; j < dim_; ++j)
        //     g_q_[i + order_][j] = x[dim_ * i + j];
        //   for (int j = dim_; j < 3; ++j)
        //     g_q_[i + order_][j] = 0.0;
        // }

        // if (!(cost_function_ & BOUNDARY))
        // {
        //   for (int i = 0; i < order_; i++)
        //   {
        //     for (int j = 0; j < dim_; ++j)
        //       g_q_[order_ + variable_num_ / dim_ + i][j] = control_points_(control_points_.rows() - order_
        //       + i, j);
        //     for (int j = dim_; j < 3; ++j)
        //       g_q_[order_ + variable_num_ / dim_ + i][j] = 0.0;
        //   }
        // }
    }

    ros::Time t1 = ros::Time::now();

    // for (int i = 0; i < point_num_; ++i)
    // {
    //     for (int j = 0; j < dim_; ++j)
    //         g_q_[i][j] = x[dim_ * i + j];
    //     for (int j = dim_; j < 3; ++j)
    //         g_q_[i][j] = 0.0;
    // }


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
        // for (int i = 0; i < point_num_; i++)
        //     for (int j = 0; j < dim_; j++)
        //         grad[dim_ * i + j] += ld_smooth_acc_ * g_smoothness_acc_[i](j);
        addGradient(grad, ld_smooth_acc_, g_smoothness_acc_);

        if (optimize_time_)
            grad[variable_num_ - 1] += ld_smooth_acc_ * gt_smoothness;
        
    }
    if (cost_function_ & SMOOTHNESS_JERK)
    {
        double f_smoothness = 0.0, gt_smoothness = 0.0;
        calcSmoothnessJerkCost(g_q_, dt, f_smoothness, g_smoothness_jerk_, gt_smoothness);
        f_combine += ld_smooth_jerk_ * f_smoothness;
        // for (int i = 0; i < point_num_; i++)
        //     for (int j = 0; j < dim_; j++)
        //         grad[dim_ * i + j] += ld_smooth_jerk_ * g_smoothness_jerk_[i](j);
        addGradient(grad, ld_smooth_jerk_, g_smoothness_jerk_);

        if (optimize_time_)
            grad[variable_num_ - 1] += ld_smooth_jerk_ * gt_smoothness;
    }
    if (cost_function_ & DISTANCE)
    {
        double f_distance = 0.0;
        calcDistanceCost(g_q_, f_distance, g_distance_);
        f_combine += ld_dist_ * f_distance;
        // for (int i = 0; i < point_num_; i++)
        //     for (int j = 0; j < dim_; j++)
        //         grad[dim_ * i + j] += ld_dist_ * g_distance_[i](j);
        addGradient(grad, ld_dist_, g_distance_);

    }
    if (cost_function_ & INTERVAL)
    {
        double f_interval = 0.0, gt_interval = 0.0;
        calcIntervalCost(g_q_, dt, f_interval, g_interval_, gt_interval);
        f_combine += ld_interval_ * f_interval;
        // for (int i = 0; i < point_num_; i++)
        //     for (int j = 0; j < dim_; j++)
        //         grad[dim_ * i + j] += ld_feasi_ * g_feasibility_[i](j);
        addGradient(grad, ld_interval_, g_interval_);

        if (optimize_time_)
            grad[variable_num_ - 1] += ld_interval_ * gt_interval;
    }
    if (cost_function_ & START)
    {
        double f_start = 0.0, gt_start = 0.0;
        calcStartCost(g_q_, dt, f_start, g_start_, gt_start);
        f_combine += ld_start_ * f_start;
        // for (int i = 0; i < 3; i++)
        //     for (int j = 0; j < dim_; j++)
        //         grad[dim_ * i + j] += ld_start_ * g_start_[i](j);
        addGradient(grad, ld_start_, g_start_);

        if (optimize_time_)
            grad[variable_num_ - 1] += ld_start_ * gt_start;
    }
    if (cost_function_ & END)
    {
        double f_end = 0.0, gt_end = 0.0;
        calcEndCost(g_q_, dt, f_end, g_end_, gt_end);       
        f_combine += ld_end_ * f_end;
        // for (int i = point_num_ - 3; i < point_num_; i++)
        //     for (int j = 0; j < dim_; j++)
        //         grad[dim_ * i + j] += ld_end_ * g_end_[i](j);
        addGradient(grad, ld_end_, g_end_);

        if (optimize_time_)
            grad[variable_num_ - 1] += ld_end_ * gt_end;
    }
    if (cost_function_ & END_HARD)
    {
        double f_end = 0.0, gt_end = 0.0;
        calcEndCost(g_q_, dt, f_end, g_end_, gt_end);
        f_combine += ld_end_hard_ * f_end;
        // for (int i = point_num_ - 3; i < point_num_; i++)
        //     for (int j = 0; j < dim_; j++)
        //         grad[dim_ * i + j] += ld_end_hard_ * g_end_[i](j);
        addGradient(grad, ld_end_hard_, g_end_);

        if (optimize_time_)
            grad[variable_num_ - 1] += ld_end_hard_ * gt_end;
    }
    // if (cost_function_ & GUIDE)
    // {
    //     double f_guide = 0.0;
    //     calcGuideCost(g_q_, f_guide, g_guide_);
    //     f_combine += ld_guide_ * f_guide;
    //     for (int i = 0; i < point_num_; i++)
    //         for (int j = 0; j < dim_; j++)
    //             grad[dim_ * i + j] += ld_guide_ * g_guide_[i](j);
    // }
    if (cost_function_ & WAYPOINTS)
    {
        double f_waypoints = 0.0;
        calcWaypointsCost(g_q_, f_waypoints, g_waypoints_);
        f_combine += ld_waypt_ * f_waypoints;
        // for (int i = 0; i < point_num_; i++)
        //     for (int j = 0; j < dim_; j++)
        //         grad[dim_ * i + j] += ld_waypt_ * g_waypoints_[i](j);
        addGradient(grad, ld_waypt_, g_waypoints_);

    }
    if (cost_function_ & MINTIME)
    {
        double f_time = 0.0, gt_time = 0.0;
        calcTimeCost(dt, f_time, gt_time);
        f_combine += ld_time_ * f_time;
        grad[variable_num_ - 1] += ld_time_ * gt_time;
    }

    comb_time += (ros::Time::now() - t1).toSec();

    // // Join thread and retrive cost/gradient
    // for (int i = 0; i < cost_thread.size(); ++i)
    //   cost_thread[i].join();

    // if (cost_function_ & SMOOTHNESS)
    // {
    //   f_combine += ld_smooth_ * f_smoothness;
    //   for (int i = 0; i < point_num_; i++)
    //     for (int j = 0; j < dim_; j++)
    //       grad[dim_ * i + j] += ld_smooth_ * g_smoothness_[i](j);
    //   if (optimize_time_)
    //     grad[variable_num_ - 1] += ld_smooth_ * gt_smoothness;
    // }
    // if (cost_function_ & DISTANCE)
    // {
    //   f_combine += ld_dist_ * f_distance;
    //   for (int i = 0; i < point_num_; i++)
    //     for (int j = 0; j < dim_; j++)
    //       grad[dim_ * i + j] += ld_dist_ * g_distance_[i](j);
    // }
    // if (cost_function_ & FEASIBILITY)
    // {
    //   f_combine += ld_feasi_ * f_feasibility;
    //   for (int i = 0; i < point_num_; i++)
    //     for (int j = 0; j < dim_; j++)
    //       grad[dim_ * i + j] += ld_feasi_ * g_feasibility_[i](j);
    //   if (optimize_time_)
    //     grad[variable_num_ - 1] += ld_feasi_ * gt_feasibility;
    // }

    /*  print cost  */
    // if ((cost_function_ & WAYPOINTS) && iter_num_ % 10 == 0) {
    //   cout << iter_num_ << ", total: " << f_combine << ", acc: " << ld_view_ * f_view
    //        << ", waypt: " << ld_waypt_ * f_waypoints << endl;
    // }

    // if (optimization_phase_ == SECOND_PHASE) {
    //  << ", smooth: " << ld_smooth_ * f_smoothness
    //  << " , dist:" << ld_dist_ * f_distance
    //  << ", fea: " << ld_feasi_ * f_feasibility << endl;
    // << ", end: " << ld_bound_ * f_endpoint
    // << ", guide: " << ld_guide_ * f_guide
    // }
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
        std::cout << cost << ", ";
    }
    return cost;

    // /* evaluation */
    // ros::Time te1 = ros::Time::now();
    // double time_now = (te1 - opt->time_start_).toSec();
    // opt->vec_time_.push_back(time_now);
    // if (opt->vec_cost_.size() == 0)
    // {
    //   opt->vec_cost_.push_back(f_combine);
    // }
    // else if (opt->vec_cost_.back() > f_combine)
    // {
    //   opt->vec_cost_.push_back(f_combine);
    // }
    // else
    // {
    //   opt->vec_cost_.push_back(opt->vec_cost_.back());
    // }
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
    for (int i = 0; i < point_num_; i++)
        for (int j = 0; j < dim_; j++)
            grad[dim_ * i + j] += weight * g[i](j);
}

