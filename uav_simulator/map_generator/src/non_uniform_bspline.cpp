#include "non_uniform_bspline.h"

NonUniformBspline::NonUniformBspline(const Eigen::MatrixXd &points, const int &order,
                                     const double &interval)
{
    setUniformBspline(points, order, interval);
}

NonUniformBspline::NonUniformBspline()
{
}

NonUniformBspline::~NonUniformBspline()
{
}

void NonUniformBspline::setUniformBspline(const Eigen::MatrixXd &points, const int &order,
                                          const double &interval)
{
    control_points_ = points;
    p_ = order;
    knot_span_ = interval;

    n_ = points.rows() - 1;
    m_ = n_ + p_ + 1;

    u_ = Eigen::VectorXd::Zero(m_ + 1);
    for (int i = 0; i <= m_; ++i)
    {
        if (i <= p_)
            u_(i) = double(-p_ + i) * knot_span_;
        else
            u_[i] = u_[i - 1] + knot_span_;
    }

    knot_length_.setZero(m_ + 1);
}

void NonUniformBspline::setKnot(const Eigen::VectorXd &knot)
{
    this->u_ = knot;
}

int NonUniformBspline::getOrder(){
    return this->p_;
}

Eigen::VectorXd NonUniformBspline::getKnot()
{
    return this->u_;
}

Eigen::VectorXd& NonUniformBspline::getKnotRef()
{
    return this->u_;
}

Eigen::VectorXd& NonUniformBspline::getKnotLength(){
    return this->knot_length_;
}

void NonUniformBspline::computeKnotLength(const double &res){
    double cal_res = min(this->getKnotSpan() * 0.5, res);

    knot_length_.setZero(m_ + 1);

    double time_offset = u_(p_);
    double max_time = u_(m_ - p_);

    double dur = getTimeSum();

    double length_s = 0.0;
    NonUniformBspline derivative = this->getDerivative();
    double va = derivative.evaluateDeBoorT(0.0).norm(), vb, v_half_ab;
    double computing_knot_idx = p_+1;
    double knot_t_offset = u_(p_);
    for (double t = cal_res, computing_knot = u_(computing_knot_idx) - knot_t_offset; t <= dur; t += cal_res, va = vb)
    {
        vb = derivative.evaluateDeBoorT(t).norm();
        v_half_ab = derivative.evaluateDeBoorT(t-0.5*cal_res).norm();

        length_s += cal_res / 6.0 * (va+ 4.0*v_half_ab+ vb);

        if(t+cal_res >= computing_knot){
            va = vb;
            vb = derivative.evaluateDeBoorT(computing_knot).norm();
            v_half_ab = derivative.evaluateDeBoorT(0.5*(t+computing_knot)).norm();
            length_s += (computing_knot-t) / 6.0 * (va+ 4.0*v_half_ab+ vb);

            t = computing_knot;
            knot_length_(computing_knot_idx) = length_s;
            computing_knot_idx++;
            computing_knot = u_(computing_knot_idx) - knot_t_offset;

        }
    }

    for(; computing_knot_idx < m_+1; computing_knot_idx++){
        knot_length_(computing_knot_idx) = length_s;
    }


}

void NonUniformBspline::getTimeSpan(double &um, double &um_p)
{
    um = u_(p_);
    um_p = u_(m_ - p_);
}

Eigen::MatrixXd NonUniformBspline::getControlPoint()
{
    return control_points_;
}

Eigen::VectorXd NonUniformBspline::evaluateDeBoor(const double &u)
{
    double ub = min(max(u_(p_), u), u_(m_ - p_));

    // Determine which [uk,uk+1] does u lay in
    int k = p_;
    while (u_(k + 1) < ub)
        ++k;

    /* deBoor's algorithm */
    // [uk,uk+1] is controlled by q[k-p]...q[k], retrieve the associated points
    vector<Eigen::VectorXd> d;
    for (int i = 0; i <= p_; ++i)
        d.emplace_back(control_points_.row(k - p_ + i));

    for (int r = 1; r <= p_; ++r)
        for (int i = p_; i >= r; --i)
        {
            double alpha = (ub - u_[i + k - p_]) / (u_[i + 1 + k - r] - u_[i + k - p_]);
            d[i] = (1 - alpha) * d[i - 1] + alpha * d[i];
        }
    return d[p_];
}

Eigen::VectorXd NonUniformBspline::evaluateDeBoorT(const double &t)
{
    return evaluateDeBoor(t + u_(p_));
}

Eigen::MatrixXd NonUniformBspline::getDerivativeControlPoints()
{
    // The derivative of a b-spline is also a b-spline, its degree become p_-1
    Eigen::MatrixXd ctp = Eigen::MatrixXd::Zero(control_points_.rows() - 1, control_points_.cols());

    // Control point Qi = p_*(Pi+1-Pi)/(ui+p_+1-ui+1)
    for (int i = 0; i < ctp.rows(); ++i)
        ctp.row(i) =
            p_ * (control_points_.row(i + 1) - control_points_.row(i)) / (u_(i + p_ + 1) - u_(i + 1));
    return ctp;
}

// Compute up to k order derivatives
void NonUniformBspline::computeDerivatives(const int &k, vector<NonUniformBspline> &ders)
{
    ders.clear();
    ders.emplace_back(getDerivative());

    for (int i = 2; i <= k; ++i)
        ders.emplace_back(ders.back().getDerivative());
}

NonUniformBspline NonUniformBspline::getDerivative()
{
    Eigen::MatrixXd ctp = getDerivativeControlPoints();
    NonUniformBspline derivative(ctp, p_ - 1, knot_span_);

    // Remove cut the first and last knot
    Eigen::VectorXd knot(u_.rows() - 2);
    knot = u_.segment(1, u_.rows() - 2);
    derivative.setKnot(knot);
    return derivative;
}

void NonUniformBspline::getBoundaryStates(const int &ks, const int &ke, vector<Eigen::Vector3d> &start,
                                          vector<Eigen::Vector3d> &end)
{
    vector<NonUniformBspline> ders;
    computeDerivatives(max(ks, ke), ders);
    double duration = getTimeSum();

    start.clear();
    end.clear();
    start.emplace_back(evaluateDeBoorT(0));
    for (int i = 0; i < ks; ++i)
        start.emplace_back(ders[i].evaluateDeBoorT(0));

    end.emplace_back(evaluateDeBoorT(duration));
    for (int i = 0; i < ke; ++i)
        end.emplace_back(ders[i].evaluateDeBoorT(duration));
}

double NonUniformBspline::getKnotSpan()
{
    return knot_span_;
}

double NonUniformBspline::getTimeFromLength(const double& length, const double& res){
    if(knot_length_.isZero()){
        computeKnotLength();
    }

    if(length >= knot_length_(m_)){
        return u_(m_ - p_) - u_(p_);
    }
    else if(length <= knot_length_(p_)){
        return 0.0;
    }

    // cout<<"knot_length:\n"<<knot_length_<<endl;

    for(int knot_idx = m_ - p_; knot_idx > p_; knot_idx--){
        
        if(knot_length_(knot_idx) < length){
            double cur_length = knot_length_(knot_idx), tmp_length;
            double t = u_(knot_idx);
            NonUniformBspline derivative = this->getDerivative();
            double va = derivative.evaluateDeBoorT(t).norm(), vb, v_half_ab;
            
            for (double cal_res = res; t <= u_(knot_idx+1); cur_length = tmp_length)
            {
                double tb = t + cal_res;
                vb = derivative.evaluateDeBoorT(tb).norm();
                v_half_ab = derivative.evaluateDeBoorT((t + tb)*0.5).norm();

                tmp_length = cur_length + cal_res / 6.0 * (va+ 4.0*v_half_ab+ vb);

                if(tmp_length == length){
                    return tb;
                }
                else if(tmp_length < length){
                    va = vb;
                    t = tb;
                }
                
                while(tmp_length > length){
                    if(cal_res < 1e-4){
                        return t;
                    }
                    cal_res *= 0.5;
                    tb = t + cal_res;
                    vb = derivative.evaluateDeBoorT(tb).norm();
                    v_half_ab = derivative.evaluateDeBoorT((t + tb)*0.5).norm();
                    tmp_length = cur_length + cal_res / 6.0 * (va+ 4.0*v_half_ab+ vb);
                }
                
            }
        
        }
        else if(knot_length_(knot_idx) == length){
            return u_(knot_idx) - u_(p_);
        }
    }
}

void NonUniformBspline::setPhysicalLimits(const double &vel, const double &acc)
{
    limit_vel_ = vel;
    limit_acc_ = acc;
    limit_ratio_ = 1.1;
}

double NonUniformBspline::checkRatio()
{
    Eigen::MatrixXd P = control_points_;
    int dimension = control_points_.cols();

    // Find max vel
    double max_vel = -1.0;
    for (int i = 0; i < P.rows() - 1; ++i)
    {
        Eigen::VectorXd vel = p_ * (P.row(i + 1) - P.row(i)) / (u_(i + p_ + 1) - u_(i + 1));
        for (int j = 0; j < dimension; ++j)
            max_vel = max(max_vel, fabs(vel(j)));
    }
    // Find max acc
    double max_acc = -1.0;
    for (int i = 0; i < P.rows() - 2; ++i)
    {
        Eigen::VectorXd acc = p_ * (p_ - 1) * ((P.row(i + 2) - P.row(i + 1)) / (u_(i + p_ + 2) - u_(i + 2)) - (P.row(i + 1) - P.row(i)) / (u_(i + p_ + 1) - u_(i + 1))) /
                              (u_(i + p_ + 1) - u_(i + 2));
        for (int j = 0; j < dimension; ++j)
            max_acc = max(max_acc, fabs(acc(j)));
    }
    double ratio = max(max_vel / limit_vel_, sqrt(fabs(max_acc) / limit_acc_));
    // ROS_INFO("max vel: %lf, max acc: %lf, ratio: %lf", max_vel, max_acc, ratio);
    // ROS_ERROR_COND(ratio > 2.0, "max vel: %lf, max acc: %lf, ratio: %lf", max_vel, max_acc, ratio);

    return ratio;
}

void NonUniformBspline::lengthenTime(const double &ratio)
{
    // To ensure that the boundary vel/acc remain the same, some of the knot should not be changed
    // The start derivatives are not influenced from u_(2p), end derivatives are not influenced until
    // u_(m-2p+1)=
    int num1 = 2 * p_ - 1;
    int num2 = (getKnot().rows() - 1) - 2 * p_ + 1;
    if (num1 >= num2)
        return;

    double delta_t = (ratio - 1.0) * (u_(num2) - u_(num1));
    double t_inc = delta_t / double(num2 - num1);
    for (int i = num1 + 1; i <= num2; ++i)
        u_(i) += double(i - num1) * t_inc;
    for (int i = num2 + 1; i < u_.rows(); ++i)
        u_(i) += delta_t;
}

void NonUniformBspline::parameterizeToBspline(const double &ts, const vector<Eigen::Vector3d> &point_set,
                                              const vector<Eigen::Vector3d> &start_end_derivative,
                                              const int &degree, Eigen::MatrixX3d &ctrl_pts)
{
    if (ts <= 0)
    {
        cout << "[B-spline]:time step error." << endl;
        return;
    }
    if (point_set.size() < 2)
    {
        cout << "[B-spline]:point set have only " << point_set.size() << " points." << endl;
        return;
    }
    if (start_end_derivative.size() != 4)
    {
        cout << "[B-spline]:derivatives error." << endl;
    }

    // Number of waypoint constraints
    int K = point_set.size();

    // Solve control points of B-spline (of different degree)
    // The matrix representation is detailed in "General matrix representations for B-splines, Qin,
    // Kaihuai"
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(K + 4, K + degree - 1);
    // Eigen::VectorXd bx(K + 4), by(K + 4), bz(K + 4);
    Eigen::MatrixX3d b(K + 4, 3);
    ctrl_pts.resize(K + degree - 1, 3);

    if (degree == 3)
    {
        // Matrix mapping control points to waypoints and boundary derivatives
        Eigen::Vector3d pt_to_pos = 1 / 6.0 * Eigen::Vector3d(1, 4, 1);
        Eigen::Vector3d pt_to_vel = 1 / (2 * ts) * Eigen::Vector3d(-1, 0, 1);
        Eigen::Vector3d pt_to_acc = 1 / (ts * ts) * Eigen::Vector3d(1, -2, 1);

        for (int i = 0; i < K; ++i)
            A.block<1, 3>(i, i) = pt_to_pos.transpose();
        A.block<1, 3>(K, 0) = pt_to_vel.transpose();
        A.block<1, 3>(K + 1, 0) = pt_to_acc.transpose();
        A.block<1, 3>(K + 2, K - 1) = pt_to_vel.transpose();
        A.block<1, 3>(K + 3, K - 1) = pt_to_acc.transpose();
    }
    else if (degree == 4)
    {
        // Repeat the same thing, but for 4 degree B-spline
        Eigen::Vector4d pt_to_pos = 1 / 24.0 * Eigen::Vector4d(1, 11, 11, 1);
        Eigen::Vector4d pt_to_vel = 1 / (6 * ts) * Eigen::Vector4d(-1, -3, 3, 1);
        Eigen::Vector4d pt_to_acc = 1 / (2 * ts * ts) * Eigen::Vector4d(1, -1, -1, 1);

        for (int i = 0; i < K; ++i)
            A.block<1, 4>(i, i) = pt_to_pos.transpose();
        A.block<1, 4>(K, 0) = pt_to_vel.transpose();
        A.block<1, 4>(K + 1, 0) = pt_to_acc.transpose();
        A.block<1, 4>(K + 2, K - 1) = pt_to_vel.transpose();
        A.block<1, 4>(K + 3, K - 1) = pt_to_acc.transpose();
    }
    else if (degree == 5)
    {
        Eigen::Matrix<double, 5, 1> pt_to_pos, pt_to_vel, pt_to_acc;
        pt_to_pos << 1, 26, 66, 26, 1;
        pt_to_pos /= 120.0;
        pt_to_vel << -1, -10, 0, 10, 1;
        pt_to_vel /= (24 * ts);
        pt_to_acc << 1, 2, -6, 2, 1;
        pt_to_acc /= (6 * ts * ts);

        for (int i = 0; i < K; ++i)
            A.block<1, 5>(i, i) = pt_to_pos.transpose();
        A.block<1, 5>(K, 0) = pt_to_vel.transpose();
        A.block<1, 5>(K + 1, 0) = pt_to_acc.transpose();
        A.block<1, 5>(K + 2, K - 1) = pt_to_vel.transpose();
        A.block<1, 5>(K + 3, K - 1) = pt_to_acc.transpose();
    }

    // cout << fixed << setprecision(2) << endl;
    // std::cout << "A:" << std::endl;
    // std::cout << A << std::endl;

    // K Waypoints and 4 boundary derivative
    // for (int i = 0; i < K; ++i)
    // {
    //     bx(i) = point_set[i][0];
    //     by(i) = point_set[i][1];
    //     bz(i) = point_set[i][2];
    // }
    // for (int i = 0; i < 4; ++i)
    // {
    //     bx(K + i) = start_end_derivative[i][0];
    //     by(K + i) = start_end_derivative[i][1];
    //     bz(K + i) = start_end_derivative[i][2];
    // }

    for (int i = 0; i < K; ++i)
    {
        for(int j = 0; j < 3; j++){
            b(i,j) = point_set[i][j];
        }

    }
    for (int i = 0; i < 4; ++i)
    {
        for(int j = 0; j < 3; j++){
            b(K+i,j) = start_end_derivative[i][j];
        }

    }

    for(int j = 0; j < 3; j++){
        ctrl_pts.col(j) = A.colPivHouseholderQr().solve(b.col(j));
    }

    // Solve Ax = b to find control points
    // ctrl_pts.col(0) = A.colPivHouseholderQr().solve(bx);
    // ctrl_pts.col(1) = A.colPivHouseholderQr().solve(by);
    // ctrl_pts.col(2) = A.colPivHouseholderQr().solve(bz);
    // cout << "[B-spline]: parameterization ok." << endl;
}

void NonUniformBspline::parameterizeToBspline(const double &ts, const vector<double> &point_set,
                                              const vector<double> &start_end_derivative,
                                              const int &degree, Eigen::VectorXd &ctrl_pts)
{
    if (ts <= 0)
    {
        cout << "[B-spline]:time step error." << endl;
        return;
    }
    if (point_set.size() < 2)
    {
        cout << "[B-spline]:point set have only " << point_set.size() << " points." << endl;
        return;
    }
    if (start_end_derivative.size() != 4)
    {
        cout << "[B-spline]:derivatives error." << endl;
    }

    // Number of waypoint constraints
    int K = point_set.size();

    // Solve control points of B-spline (of different degree)
    // The matrix representation is detailed in "General matrix representations for B-splines, Qin,
    // Kaihuai"
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(K + 4, K + degree - 1);
    // Eigen::VectorXd bx(K + 4), by(K + 4), bz(K + 4);
    Eigen::VectorXd b(K + 4);
    ctrl_pts.resize(K + degree - 1);

    if (degree == 3)
    {
        // Matrix mapping control points to waypoints and boundary derivatives
        Eigen::Vector3d pt_to_pos = 1 / 6.0 * Eigen::Vector3d(1, 4, 1);
        Eigen::Vector3d pt_to_vel = 1 / (2 * ts) * Eigen::Vector3d(-1, 0, 1);
        Eigen::Vector3d pt_to_acc = 1 / (ts * ts) * Eigen::Vector3d(1, -2, 1);

        for (int i = 0; i < K; ++i)
            A.block<1, 3>(i, i) = pt_to_pos.transpose();
        A.block<1, 3>(K, 0) = pt_to_vel.transpose();
        A.block<1, 3>(K + 1, 0) = pt_to_acc.transpose();
        A.block<1, 3>(K + 2, K - 1) = pt_to_vel.transpose();
        A.block<1, 3>(K + 3, K - 1) = pt_to_acc.transpose();
    }
    else if (degree == 4)
    {
        // Repeat the same thing, but for 4 degree B-spline
        Eigen::Vector4d pt_to_pos = 1 / 24.0 * Eigen::Vector4d(1, 11, 11, 1);
        Eigen::Vector4d pt_to_vel = 1 / (6 * ts) * Eigen::Vector4d(-1, -3, 3, 1);
        Eigen::Vector4d pt_to_acc = 1 / (2 * ts * ts) * Eigen::Vector4d(1, -1, -1, 1);

        for (int i = 0; i < K; ++i)
            A.block<1, 4>(i, i) = pt_to_pos.transpose();
        A.block<1, 4>(K, 0) = pt_to_vel.transpose();
        A.block<1, 4>(K + 1, 0) = pt_to_acc.transpose();
        A.block<1, 4>(K + 2, K - 1) = pt_to_vel.transpose();
        A.block<1, 4>(K + 3, K - 1) = pt_to_acc.transpose();
    }
    else if (degree == 5)
    {
        Eigen::Matrix<double, 5, 1> pt_to_pos, pt_to_vel, pt_to_acc;
        pt_to_pos << 1, 26, 66, 26, 1;
        pt_to_pos /= 120.0;
        pt_to_vel << -1, -10, 0, 10, 1;
        pt_to_vel /= (24 * ts);
        pt_to_acc << 1, 2, -6, 2, 1;
        pt_to_acc /= (6 * ts * ts);

        for (int i = 0; i < K; ++i)
            A.block<1, 5>(i, i) = pt_to_pos.transpose();
        A.block<1, 5>(K, 0) = pt_to_vel.transpose();
        A.block<1, 5>(K + 1, 0) = pt_to_acc.transpose();
        A.block<1, 5>(K + 2, K - 1) = pt_to_vel.transpose();
        A.block<1, 5>(K + 3, K - 1) = pt_to_acc.transpose();
    }

    // cout << fixed << setprecision(2) << endl;
    // std::cout << "A:" << std::endl;
    // std::cout << A << std::endl;

    // K Waypoints and 4 boundary derivative
    // for (int i = 0; i < K; ++i)
    // {
    //     bx(i) = point_set[i][0];
    //     by(i) = point_set[i][1];
    //     bz(i) = point_set[i][2];
    // }
    // for (int i = 0; i < 4; ++i)
    // {
    //     bx(K + i) = start_end_derivative[i][0];
    //     by(K + i) = start_end_derivative[i][1];
    //     bz(K + i) = start_end_derivative[i][2];
    // }

    for (int i = 0; i < K; ++i)
    {       
        b(i) = point_set[i];       
    }
    for (int i = 0; i < 4; ++i)
    {
        b(K+i) = start_end_derivative[i];

    }

    ctrl_pts = A.colPivHouseholderQr().solve(b);

    // Solve Ax = b to find control points
    // ctrl_pts.col(0) = A.colPivHouseholderQr().solve(bx);
    // ctrl_pts.col(1) = A.colPivHouseholderQr().solve(by);
    // ctrl_pts.col(2) = A.colPivHouseholderQr().solve(bz);
    // cout << "[B-spline]: parameterization ok." << endl;
}

void NonUniformBspline::solveFirst3CtrlPts(const double &ts, const vector<Eigen::Vector3d> &start, Eigen::Matrix3d &ctrl_pts){
    
    if(start.size() != 3){
        cout << "[B-spline]:start constraints error." << endl;
    }

    Eigen::RowVector3d pt_to_pos = 1 / 6.0 * Eigen::RowVector3d(1, 4, 1);
    Eigen::RowVector3d pt_to_vel = 1 / (2 * ts) * Eigen::RowVector3d(-1, 0, 1);
    Eigen::RowVector3d pt_to_acc = 1 / (ts * ts) * Eigen::RowVector3d(1, -2, 1);

    Eigen::Matrix3d A, b;

    A << pt_to_pos, pt_to_vel, pt_to_acc;

    for(int i = 0; i < 3; i++)
        b.row(i) = start[i].transpose();
    

    // cout<<"A mat\n"<<A<<endl;

    // cout<<"b:\n"<<b<<endl;

    for(int i = 0; i < 3; i++)
        ctrl_pts.col(i) = A.colPivHouseholderQr().solve(b.col(i));

    // cout<<"ctrl pts\n"<<ctrl_pts<<endl;

}

void NonUniformBspline::solveFirst3CtrlPts(const double &ts, const vector<double> &start, Eigen::Vector3d &ctrl_pts){
    
    if(start.size() != 3){
        cout << "[B-spline]:start constraints error." << endl;
    }

    Eigen::RowVector3d pt_to_pos = 1 / 6.0 * Eigen::RowVector3d(1, 4, 1);
    Eigen::RowVector3d pt_to_vel = 1 / (2 * ts) * Eigen::RowVector3d(-1, 0, 1);
    Eigen::RowVector3d pt_to_acc = 1 / (ts * ts) * Eigen::RowVector3d(1, -2, 1);

    Eigen::Matrix3d A;
    Eigen::Vector3d b;

    A << pt_to_pos, pt_to_vel, pt_to_acc;

    for(int i = 0; i < 3; i++)
        b(i) = start[i];
    

    // cout<<"A mat\n"<<A<<endl;
    // cout<<"b: "<<b.transpose()<<endl;

    ctrl_pts = A.colPivHouseholderQr().solve(b);

    // cout<<"ctrl pts\n"<<ctrl_pts<<endl;

}

double NonUniformBspline::getTimeSum()
{
    return u_(m_ - p_) - u_(p_);
}

double NonUniformBspline::getLength(const double &dur, const double &res)
{
    // double length = 0.0;
    // double dur = getTimeSum();
    // Eigen::VectorXd p_l = evaluateDeBoorT(0.0), p_n;
    // for (double t = res; t <= dur + 1e-4; t += res)
    // {
    //     p_n = evaluateDeBoorT(t);
    //     length += (p_n - p_l).norm();
    //     p_l = p_n;
    // }

    double length_s = 0.0;
    NonUniformBspline derivative = this->getDerivative();
    double va = derivative.evaluateDeBoorT(0.0).norm(), vb, v_half_ab;
    for (double t = res; t <= dur; t += res, va = vb)
    {
        vb = derivative.evaluateDeBoorT(t).norm();
        v_half_ab = derivative.evaluateDeBoorT(t-0.5*res).norm();

        length_s += res / 6.0 * (va+ 4.0*v_half_ab+ vb);

        if(t+res > dur){
            va = vb;
            vb = derivative.evaluateDeBoorT(dur).norm();
            v_half_ab = derivative.evaluateDeBoorT(0.5*(t+dur)).norm();
            length_s += (dur-t) / 6.0 * (va+ 4.0*v_half_ab+ vb);
        }
    }

    return length_s;
}

// double NonUniformBspline::getLength(const double &res)
// {
//     return getLength(getTimeSum(), res);
// }

double NonUniformBspline::getJerk()
{
    NonUniformBspline jerk_traj = getDerivative().getDerivative().getDerivative();

    Eigen::VectorXd times = jerk_traj.getKnot();
    Eigen::MatrixXd ctrl_pts = jerk_traj.getControlPoint();
    int dimension = ctrl_pts.cols();

    double jerk = 0.0;
    for (int i = 0; i < ctrl_pts.rows(); ++i)
    {
        for (int j = 0; j < dimension; ++j)
        {
            jerk += (times(i + 1) - times(i)) * ctrl_pts(i, j) * ctrl_pts(i, j);
        }
    }

    return jerk;
}

void NonUniformBspline::getMeanAndMaxVel(double &mean_v, double &max_v)
{
    NonUniformBspline vel = getDerivative();
    double tm, tmp;
    vel.getTimeSpan(tm, tmp);

    double max_vel = -1.0, mean_vel = 0.0;
    int num = 0;
    for (double t = tm; t <= tmp; t += 0.01)
    {
        Eigen::VectorXd vxd = vel.evaluateDeBoor(t);
        double vn = vxd.norm();

        mean_vel += vn;
        ++num;
        if (vn > max_vel)
        {
            max_vel = vn;
        }
    }

    mean_vel = mean_vel / double(num);
    mean_v = mean_vel;
    max_v = max_vel;
}

void NonUniformBspline::getMeanAndMaxAcc(double &mean_a, double &max_a)
{
    NonUniformBspline acc = getDerivative().getDerivative();
    double tm, tmp;
    acc.getTimeSpan(tm, tmp);

    double max_acc = -1.0, mean_acc = 0.0;
    int num = 0;
    for (double t = tm; t <= tmp; t += 0.01)
    {
        Eigen::VectorXd axd = acc.evaluateDeBoor(t);
        double an = axd.norm();

        mean_acc += an;
        ++num;
        if (an > max_acc)
        {
            max_acc = an;
        }
    }

    mean_acc = mean_acc / double(num);
    mean_a = mean_acc;
    max_a = max_acc;
}

bool NonUniformBspline::reallocateTime(bool show)
{
    // SETY << "[Bspline]: total points size: " << control_points_.rows() << endl;
    // cout << "origin knots:\n" << u_.transpose() << endl;
    bool fea = true;

    Eigen::MatrixXd P = control_points_;
    int dimension = control_points_.cols();

    double max_vel, max_acc;

    /* check vel feasibility and insert points */
    for (int i = 0; i < P.rows() - 1; ++i)
    {
        Eigen::VectorXd vel = p_ * (P.row(i + 1) - P.row(i)) / (u_(i + p_ + 1) - u_(i + 1));

        if (fabs(vel(0)) > limit_vel_ + 1e-4 || fabs(vel(1)) > limit_vel_ + 1e-4 ||
            fabs(vel(2)) > limit_vel_ + 1e-4)
        {
            fea = false;
            if (show)
                cout << "[Realloc]: Infeasible vel " << i << " :" << vel.transpose() << endl;

            max_vel = -1.0;
            for (int j = 0; j < dimension; ++j)
            {
                max_vel = max(max_vel, fabs(vel(j)));
            }

            double ratio = max_vel / limit_vel_ + 1e-4;
            if (ratio > limit_ratio_)
                ratio = limit_ratio_;

            double time_ori = u_(i + p_ + 1) - u_(i + 1);
            double time_new = ratio * time_ori;
            double delta_t = time_new - time_ori;
            double t_inc = delta_t / double(p_);

            for (int j = i + 2; j <= i + p_ + 1; ++j)
            {
                u_(j) += double(j - i - 1) * t_inc;
                if (j <= 5 && j >= 1)
                {
                    // cout << "vel j: " << j << endl;
                }
            }

            for (int j = i + p_ + 2; j < u_.rows(); ++j)
            {
                u_(j) += delta_t;
            }
        }
    }

    /* acc feasibility */
    for (int i = 0; i < P.rows() - 2; ++i)
    {
        Eigen::VectorXd acc = p_ * (p_ - 1) * ((P.row(i + 2) - P.row(i + 1)) / (u_(i + p_ + 2) - u_(i + 2)) - (P.row(i + 1) - P.row(i)) / (u_(i + p_ + 1) - u_(i + 1))) /
                              (u_(i + p_ + 1) - u_(i + 2));

        if (fabs(acc(0)) > limit_acc_ + 1e-4 || fabs(acc(1)) > limit_acc_ + 1e-4 ||
            fabs(acc(2)) > limit_acc_ + 1e-4)
        {
            fea = false;
            if (show)
                cout << "[Realloc]: Infeasible acc " << i << " :" << acc.transpose() << endl;

            max_acc = -1.0;
            for (int j = 0; j < dimension; ++j)
            {
                max_acc = max(max_acc, fabs(acc(j)));
            }

            double ratio = sqrt(max_acc / limit_acc_) + 1e-4;
            if (ratio > limit_ratio_)
                ratio = limit_ratio_;
            // cout << "ratio: " << ratio << endl;

            double time_ori = u_(i + p_ + 1) - u_(i + 2);
            double time_new = ratio * time_ori;
            double delta_t = time_new - time_ori;
            double t_inc = delta_t / double(p_ - 1);

            if (i == 1 || i == 2)
            {
                // cout << "acc i: " << i << endl;
                for (int j = 2; j <= 5; ++j)
                {
                    u_(j) += double(j - 1) * t_inc;
                }

                for (int j = 6; j < u_.rows(); ++j)
                {
                    u_(j) += 4.0 * t_inc;
                }
            }
            else
            {
                for (int j = i + 3; j <= i + p_ + 1; ++j)
                {
                    u_(j) += double(j - i - 2) * t_inc;
                    if (j <= 5 && j >= 1)
                    {
                        // cout << "acc j: " << j << endl;
                    }
                }

                for (int j = i + p_ + 2; j < u_.rows(); ++j)
                {
                    u_(j) += delta_t;
                }
            }
        }
    }

    return fea;
}

bool NonUniformBspline::checkFeasibility(bool show)
{
    bool fea = true;
    // SETY << "[Bspline]: total points size: " << control_points_.rows() << endl;

    Eigen::MatrixXd P = control_points_;
    int dimension = control_points_.cols();

    /* check vel feasibility and insert points */
    double max_vel = -1.0;
    for (int i = 0; i < P.rows() - 1; ++i)
    {
        Eigen::VectorXd vel = p_ * (P.row(i + 1) - P.row(i)) / (u_(i + p_ + 1) - u_(i + 1));

        if (fabs(vel(0)) > limit_vel_ + 1e-4 || fabs(vel(1)) > limit_vel_ + 1e-4 ||
            fabs(vel(2)) > limit_vel_ + 1e-4)
        {
            if (show)
                cout << "[Check]: Infeasible vel " << i << " :" << vel.transpose() << endl;
            fea = false;

            for (int j = 0; j < dimension; ++j)
            {
                max_vel = max(max_vel, fabs(vel(j)));
            }
        }
    }

    /* acc feasibility */
    double max_acc = -1.0;
    for (int i = 0; i < P.rows() - 2; ++i)
    {
        Eigen::VectorXd acc = p_ * (p_ - 1) * ((P.row(i + 2) - P.row(i + 1)) / (u_(i + p_ + 2) - u_(i + 2)) - (P.row(i + 1) - P.row(i)) / (u_(i + p_ + 1) - u_(i + 1))) /
                              (u_(i + p_ + 1) - u_(i + 2));

        if (fabs(acc(0)) > limit_acc_ + 1e-4 || fabs(acc(1)) > limit_acc_ + 1e-4 ||
            fabs(acc(2)) > limit_acc_ + 1e-4)
        {
            if (show)
                cout << "[Check]: Infeasible acc " << i << " :" << acc.transpose() << endl;
            fea = false;

            for (int j = 0; j < dimension; ++j)
            {
                max_acc = max(max_acc, fabs(acc(j)));
            }
        }
    }

    double ratio = max(max_vel / limit_vel_, sqrt(fabs(max_acc) / limit_acc_));

    return fea;
}
