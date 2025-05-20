/*******************************************************
Copyright (C) 2025, Aerial Robotics Group, Hong Kong University of Science and Technology
This file is part of FINT.
Licensed under the GNU General Public License v3.0;
you may not use this file except in compliance with the License.
*******************************************************/

#include "non_uniform_bspline.h"
#include "nav_msgs/Odometry.h"
#include "geometry_msgs/PoseStamped.h"
#include "tunnel_planner/Bspline_with_retiming.h"
#include "tunnel_data_type.hpp"
#include "quadrotor_msgs/PositionCommand.h"
#include "std_msgs/Empty.h"
#include "visualization_msgs/Marker.h"
#include <ros/ros.h>
#include <queue>


struct traj_info{
    int traj_id_;
    double duration_;
    ros::Time start_time_;
    NonUniformBspline position_traj_, velocity_traj_, acceleration_traj_;
    NonUniformBspline yaw_traj_;
    NonUniformBspline position_traj_1d_, velocity_traj_1d_, acceleration_traj_1d_;

    bool have_retiming_;
    int yaw_strategy_;
    Eigen::Vector3d yaw_dir_pt_;
};

queue<traj_info> traj_queue;

ros::Publisher cmd_vis_pub, pos_cmd_pub, traj_pub;
nav_msgs::Odometry odom;
quadrotor_msgs::PositionCommand cmd;

double last_yaw_cmd_ = 0.0;

// Info of generated traj
int pub_traj_id_;

// Info of replan
bool receive_traj_ = false;
bool is_init_ = false;
bool is_traj_ = false;
bool traj_finish_ = true;
double replan_time_;

// Executed traj, commanded and real ones
deque<Eigen::Vector3d> traj_real_;
vector<Eigen::Vector3d> traj_cmd_;

// Data for benchmark comparison
ros::Time start_time, end_time, last_time;
double energy;

// Loop correction
Eigen::Matrix3d R_loop;
Eigen::Vector3d T_loop;
bool isLoopCorrection = false;

bool use_vel_yaw = false;

double calcPathLength(const vector<Eigen::Vector3d> &path)
{
    if (path.empty())
        return 0;
    double len = 0.0;
    for (int i = 0; i < path.size() - 1; ++i)
    {
        len += (path[i + 1] - path[i]).norm();
    }
    return len;
}

void displayTrajWithColor(vector<Eigen::Vector3d> path, double resolution, Eigen::Vector4d color,
                          int id)
{
    visualization_msgs::Marker mk;
    mk.header.frame_id = "world";
    mk.header.stamp = ros::Time::now();
    mk.type = visualization_msgs::Marker::SPHERE_LIST;
    mk.action = visualization_msgs::Marker::DELETE;
    mk.id = id;
    traj_pub.publish(mk);

    mk.action = visualization_msgs::Marker::ADD;
    mk.pose.orientation.x = 0.0;
    mk.pose.orientation.y = 0.0;
    mk.pose.orientation.z = 0.0;
    mk.pose.orientation.w = 1.0;
    mk.color.r = color(0);
    mk.color.g = color(1);
    mk.color.b = color(2);
    mk.color.a = color(3);
    mk.scale.x = resolution;
    mk.scale.y = resolution;
    mk.scale.z = resolution;
    geometry_msgs::Point pt;
    for (int i = 0; i < int(path.size()); i++)
    {
        pt.x = path[i](0);
        pt.y = path[i](1);
        pt.z = path[i](2);
        mk.points.push_back(pt);
    }
    traj_pub.publish(mk);
    ros::Duration(0.001).sleep();
}

void drawFOV(const vector<Eigen::Vector3d> &list1, const vector<Eigen::Vector3d> &list2)
{
    visualization_msgs::Marker mk;
    mk.header.frame_id = "world";
    mk.header.stamp = ros::Time::now();
    mk.id = 0;
    mk.ns = "current_pose";
    mk.type = visualization_msgs::Marker::LINE_LIST;
    mk.pose.orientation.x = 0.0;
    mk.pose.orientation.y = 0.0;
    mk.pose.orientation.z = 0.0;
    mk.pose.orientation.w = 1.0;
    mk.color.r = 1.0;
    mk.color.g = 0.0;
    mk.color.b = 0.0;
    mk.color.a = 1.0;
    mk.scale.x = 0.04;
    mk.scale.y = 0.04;
    mk.scale.z = 0.04;

    // Clean old marker
    mk.action = visualization_msgs::Marker::DELETE;
    cmd_vis_pub.publish(mk);

    if (list1.size() == 0)
        return;

    // Pub new marker
    geometry_msgs::Point pt;
    for (int i = 0; i < int(list1.size()); ++i)
    {
        pt.x = list1[i](0);
        pt.y = list1[i](1);
        pt.z = list1[i](2);
        mk.points.push_back(pt);

        pt.x = list2[i](0);
        pt.y = list2[i](1);
        pt.z = list2[i](2);
        mk.points.push_back(pt);
    }
    mk.action = visualization_msgs::Marker::ADD;
    cmd_vis_pub.publish(mk);
}

void drawCmd(const Eigen::Vector3d &pos, const Eigen::Vector3d &vec, const int &id,
             const Eigen::Vector4d &color)
{
    visualization_msgs::Marker mk_state;
    mk_state.header.frame_id = "world";
    mk_state.header.stamp = ros::Time::now();
    mk_state.id = id;
    mk_state.type = visualization_msgs::Marker::ARROW;
    mk_state.action = visualization_msgs::Marker::ADD;

    mk_state.pose.orientation.w = 1.0;
    mk_state.scale.x = 0.1;
    mk_state.scale.y = 0.2;
    mk_state.scale.z = 0.3;

    geometry_msgs::Point pt;
    pt.x = pos(0);
    pt.y = pos(1);
    pt.z = pos(2);
    mk_state.points.push_back(pt);

    pt.x = pos(0) + vec(0);
    pt.y = pos(1) + vec(1);
    pt.z = pos(2) + vec(2);
    mk_state.points.push_back(pt);

    mk_state.color.r = color(0);
    mk_state.color.g = color(1);
    mk_state.color.b = color(2);
    mk_state.color.a = color(3);

    cmd_vis_pub.publish(mk_state);
}

void newCallback(std_msgs::Empty msg)
{
    // Clear the executed traj data
    traj_cmd_.clear();
    traj_real_.clear();
}

void odomCallbck(const nav_msgs::Odometry &msg)
{
    if (msg.child_frame_id == "X" || msg.child_frame_id == "O")
        return;

    odom = msg;
    if(is_init_){
        if(is_traj_) {
            Eigen::Vector3d cur_pos(odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z);
            Eigen::Vector3d cur_vel(odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z);

            traj_real_.emplace_back(cur_pos);

            if (traj_real_.size() > 10000)
                traj_real_.erase(traj_real_.begin(), traj_real_.begin() + 1000);

            if (!receive_traj_)
                return;

            if(traj_queue.empty()){
                // should hover
                return;
            }


            // pop old traj
            for(double t = msg.header.stamp.toSec(); traj_queue.size() > 1 && t > traj_queue.front().start_time_.toSec() + traj_queue.front().duration_; traj_queue.pop());
            
            
            traj_info& cur_traj = traj_queue.front();
            double t_cur = (msg.header.stamp - cur_traj.start_time_).toSec();

            Eigen::Vector3d pos = Eigen::Vector3d::Zero();
            Eigen::Vector3d vel = Eigen::Vector3d::Zero();
            Eigen::Vector3d acc = Eigen::Vector3d::Zero();

            double yaw = atan2(2 * (msg.pose.pose.orientation.w * msg.pose.pose.orientation.z + msg.pose.pose.orientation.x * msg.pose.pose.orientation.y), 1 - 2 * (msg.pose.pose.orientation.y * msg.pose.pose.orientation.y + msg.pose.pose.orientation.z * msg.pose.pose.orientation.z));
            double yawdot = 0.0;

            if (t_cur < cur_traj.duration_ && t_cur >= 0.0)
            {
                // Current time within range of planned traj
                double process = 0.0;
                if(cur_traj.have_retiming_){
                    double curve_length = cur_traj.position_traj_1d_.evaluateDeBoorT(t_cur)(0);

                    double tangent_speed = cur_traj.velocity_traj_1d_.evaluateDeBoorT(t_cur)(0);
                    double tangent_acceleration = cur_traj.acceleration_traj_1d_.evaluateDeBoorT(t_cur)(0);

                    process = cur_traj.position_traj_.getTimeFromLength(curve_length, 2e-4);

                    pos = cur_traj.position_traj_.evaluateDeBoorT(process);

                    Eigen::Vector3d center_line_vel = cur_traj.velocity_traj_.evaluateDeBoorT(process);
                    Eigen::Vector3d center_line_dir = center_line_vel.normalized();
                    vel = center_line_dir * tangent_speed;
                    
                    Eigen::Vector3d center_line_acc = cur_traj.acceleration_traj_.evaluateDeBoorT(process);
                    double squared_center_line_v = center_line_vel.squaredNorm();

                    Eigen::Vector3d center_line_central_acc = center_line_acc - center_line_acc.dot(center_line_dir) * center_line_dir;
                    double center_line_central_a = center_line_central_acc.norm();

                    // Eigen::Vector3d curvature = center_line_central_acc / squared_center_line_v;

                    double curvature_norm = center_line_central_a / squared_center_line_v;

                    Eigen::Vector3d centripetal_acc = center_line_central_acc / center_line_central_a * tangent_speed * tangent_speed * curvature_norm;

                    acc = centripetal_acc + tangent_acceleration * center_line_dir;
                }
                else{
                    pos = cur_traj.position_traj_.evaluateDeBoorT(t_cur);
                    vel = cur_traj.velocity_traj_.evaluateDeBoorT(t_cur);
                    acc = cur_traj.acceleration_traj_.evaluateDeBoorT(t_cur);
                }


                switch(cur_traj.yaw_strategy_){

                    case tunnel_planner::yaw_stragety::TANGENT:{
                        if(vel.norm() > 0.1){
                            yaw = atan2(vel.y(), vel.x());
                        }
                        else{
                            yaw = last_yaw_cmd_;
                        }
                        break;
                    }

                    case tunnel_planner::yaw_stragety::CONSTANT_PT:{
                        Eigen::Vector3d diff = cur_traj.yaw_dir_pt_ - cur_pos;
                        if(diff.head(2).squaredNorm() > 1e-4){
                            yaw = atan2(diff.y(), diff.x());
                        }
                        else{
                            yaw = last_yaw_cmd_;
                        }
                        break;
                    }
                    
                    case tunnel_planner::yaw_stragety::CONSTANT_TRAJ_DIST:{
                        yaw = last_yaw_cmd_;
                        break;
                    }

                    case tunnel_planner::yaw_stragety::CONSTANT_CHORD_DIST:{
                        yaw = last_yaw_cmd_;
                        break;
                    }

                    case tunnel_planner::yaw_stragety::PLAN:{
                        if(cur_traj.have_retiming_){
                            yaw = cur_traj.yaw_traj_.evaluateDeBoorT(process)(0);
                        }
                        else{
                            yaw = cur_traj.yaw_traj_.evaluateDeBoorT(t_cur)(0);
                        }
                        break;
                    }
                    default:{
                        yaw = last_yaw_cmd_;
                        break;
                    }
                }

                cmd.trajectory_flag = quadrotor_msgs::PositionCommand::TRAJECTORY_STATUS_READY;

                if(traj_finish_){
                    if((pos - cur_pos).squaredNorm() > 0.01){
                        is_traj_ = false;
                        ROS_ERROR("large pos error!!!");
                        return;
                    }
                    if((vel - cur_vel).squaredNorm() > 0.01){
                        is_traj_ = false;
                        ROS_ERROR("large vel error!!!");
                        return;
                    }
                    traj_finish_ = false;
                    ROS_INFO("continue traj!!!");
                }

            }
            else if (t_cur >= cur_traj.duration_)
            {
                // Current time exceed range of planned traj
                // keep publishing the final position and yaw
                double total_traj_virtual_duration = cur_traj.position_traj_.getTimeSum();
                pos = cur_traj.position_traj_.evaluateDeBoorT(total_traj_virtual_duration);

                yaw = last_yaw_cmd_;

                cmd.trajectory_flag = quadrotor_msgs::PositionCommand::TRAJECTORY_STATUS_COMPLETED;

                // Report info of the whole flight
                double len = calcPathLength(traj_cmd_);
                double flight_t = (end_time - start_time).toSec();
                ROS_WARN_THROTTLE(2, "flight time: %lf, path length: %lf, mean vel: %lf, energy is: % lf ", flight_t,
                                len, len / flight_t, energy);
                
                traj_finish_ = true;
                // ROS_ERROR("traj finish!!!");

                if(t_cur > cur_traj.duration_ + 10.0){
                    is_traj_ = false;
                }
            }
            else
            {
                // cout << "[Traj server]: invalid time." << endl;
                cmd.trajectory_flag = quadrotor_msgs::PositionCommand::TRAJECTORY_STATUS_EMPTY;
                pos = cur_traj.position_traj_.evaluateDeBoorT(0.0);
                yaw = last_yaw_cmd_;

                if((pos-cur_pos).squaredNorm() > 0.01){
                    pos = cur_pos;
                }
            }

            

            if (isLoopCorrection)
            {
                pos = R_loop.transpose() * (pos - T_loop);
                vel = R_loop.transpose() * vel;
                acc = R_loop.transpose() * acc;

                Eigen::Vector3d yaw_dir(cos(yaw), sin(yaw), 0);
                yaw_dir = R_loop.transpose() * yaw_dir;
                yaw = atan2(yaw_dir[1], yaw_dir[0]);
            }

            while(yaw > M_PI){
                yaw -= (2 * M_PI);
            }
            while(yaw < -M_PI){
                yaw += (2 * M_PI);
            }

            cmd.header.stamp = msg.header.stamp;
            cmd.trajectory_id = cur_traj.traj_id_;
            cmd.position.x = pos(0);
            cmd.position.y = pos(1);
            cmd.position.z = pos(2);
            cmd.velocity.x = vel(0);
            cmd.velocity.y = vel(1);
            cmd.velocity.z = vel(2);
            cmd.acceleration.x = acc(0);
            cmd.acceleration.y = acc(1);
            cmd.acceleration.z = acc(2);
            cmd.yaw = yaw;
            cmd.yaw_dot = yawdot;
            pos_cmd_pub.publish(cmd);

            last_yaw_cmd_ = yaw;

            // Draw cmd
            Eigen::Vector3d dir(cos(yaw), sin(yaw), 0.0);
            drawCmd(pos, 2 * dir, 2, Eigen::Vector4d(1, 1, 0, 0.7));
            drawCmd(pos, vel, 0, Eigen::Vector4d(0, 1, 0, 1));
            drawCmd(pos, acc, 1, Eigen::Vector4d(0, 0, 1, 1));
            // drawCmd(pos, pos_err, 3, Eigen::Vector4d(1, 1, 0, 0.7));

            // Record info of the executed traj
            if (traj_cmd_.size() == 0)
            {
                // Add the first position
                traj_cmd_.emplace_back(pos);
            }
            else if ((pos - traj_cmd_.back()).norm() > 1e-6)
            {
                // Add new different commanded position
                traj_cmd_.push_back(pos);
                double dt = (msg.header.stamp - last_time).toSec();
                // energy += jer.squaredNorm() * dt;
                end_time = ros::Time::now();
            }
            // last_pos_cmd_ = pos;
            // last_vel_cmd_ = vel;
            // last_acc_cmd_ = acc;
            last_time = msg.header.stamp;
        }
    }
    else{
        is_init_ = true;
    }
   
}

void pgTVioCallback(geometry_msgs::Pose msg)
{
    // World to odom
    Eigen::Quaterniond q =
        Eigen::Quaterniond(msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z);
    R_loop = q.toRotationMatrix();
    T_loop << msg.position.x, msg.position.y, msg.position.z;

    // cout << "R_loop: " << R_loop << endl;
    // cout << "T_loop: " << T_loop << endl;
}

void visCallback(const ros::TimerEvent &e)
{
    // Draw the executed traj (desired state)
    displayTrajWithColor(traj_cmd_, 0.05, Eigen::Vector4d(0, 0, 1, 1), pub_traj_id_);

}

void trajCallback(const tunnel_planner::Bspline_with_retimingConstPtr &msg)
{
    // Received traj should have ascending traj_id
    if(!traj_queue.empty()){
        if (msg->traj_id <= traj_queue.back().traj_id_ || msg->start_time <= traj_queue.back().start_time_)
        {
            // ROS_ERROR("out of order traj.");
            return;
        }
    }

    traj_info traj_input;

    // Parse the msg
    Eigen::MatrixXd pos_pts(msg->pos_pts_3d.size(), 3);
    Eigen::VectorXd knots(msg->knots_3d.size());
    for (int i = 0; i < msg->knots_3d.size(); ++i)
    {
        knots(i) = msg->knots_3d[i];
    }

    for (int i = 0; i < msg->pos_pts_3d.size(); ++i)
    {
        pos_pts(i, 0) = msg->pos_pts_3d[i].x;
        pos_pts(i, 1) = msg->pos_pts_3d[i].y;
        pos_pts(i, 2) = msg->pos_pts_3d[i].z;
    }

    traj_input.start_time_ = msg->start_time;
    traj_input.traj_id_ = msg->traj_id;

    traj_input.position_traj_ = NonUniformBspline(pos_pts, msg->order_3d, 0.1);
    traj_input.position_traj_.setKnot(knots);
    traj_input.velocity_traj_ = traj_input.position_traj_.getDerivative();
    traj_input.acceleration_traj_ = traj_input.velocity_traj_.getDerivative();

    traj_input.duration_ = traj_input.position_traj_.getTimeSum();

    traj_input.yaw_strategy_ = msg->yaw_strategy;
    
    switch(msg->yaw_strategy){

        case tunnel_planner::yaw_stragety::CONSTANT_PT:{
            traj_input.yaw_dir_pt_ << msg->yaw_dir_pt.x, msg->yaw_dir_pt.y, msg->yaw_dir_pt.z;
            break;
        }

        case tunnel_planner::yaw_stragety::PLAN:{
            Eigen::VectorXd yaw_pts(msg->yaw_pts.size());
            Eigen::VectorXd knots_yaw(msg->knots_yaw.size());
            for (int i = 0; i < msg->knots_yaw.size(); ++i)
            {
                knots_yaw(i) = msg->knots_yaw[i];
            }
            for (int i = 0; i < msg->yaw_pts.size(); ++i)
            {
                yaw_pts(i) = msg->yaw_pts[i];
            }

            traj_input.yaw_traj_ = NonUniformBspline(yaw_pts, msg->order_yaw, 0.1);
            traj_input.yaw_traj_.setKnot(knots_yaw);
            break;
        }

        default:
            break;
    }


    traj_input.have_retiming_ = msg->have_retiming;

    if(msg->have_retiming){
        Eigen::VectorXd pos_pts_1d(msg->pos_pts_1d.size());
        Eigen::VectorXd knots_1d(msg->knots_1d.size());
        for (int i = 0; i < msg->knots_1d.size(); ++i)
        {
            knots_1d(i) = msg->knots_1d[i];
        }
        for (int i = 0; i < msg->pos_pts_1d.size(); ++i)
        {
            pos_pts_1d(i) = msg->pos_pts_1d[i];
        }

        traj_input.position_traj_1d_ = NonUniformBspline(pos_pts_1d, msg->order_1d, 0.1);
        traj_input.position_traj_1d_.setKnot(knots_1d);
        traj_input.velocity_traj_1d_ = traj_input.position_traj_1d_.getDerivative();
        traj_input.acceleration_traj_1d_ = traj_input.velocity_traj_1d_.getDerivative();

        traj_input.duration_ = traj_input.position_traj_1d_.getTimeSum();
    }

    receive_traj_ = true;

    if(!traj_queue.empty()){
        auto& last_traj = traj_queue.back();
        last_traj.duration_ = min(last_traj.duration_, (traj_input.start_time_ - last_traj.start_time_).toSec());
    }
    traj_queue.emplace(traj_input);

    // Record the start time of flight
    if (start_time.isZero())
    {
        ROS_WARN("start flight");
        start_time = ros::Time::now();
    }
}

void trigger_callback( const geometry_msgs::PoseStamped::ConstPtr& trigger_msg ){
    if ( is_init_ ){
        std::cout << "[#INFO] get traj trigger info." << std::endl;
        is_traj_    = true;
        traj_finish_ = false;
        last_yaw_cmd_ = atan2(2 * (odom.pose.pose.orientation.w * odom.pose.pose.orientation.z + odom.pose.pose.orientation.x * odom.pose.pose.orientation.y), 1 - 2 * (odom.pose.pose.orientation.y * odom.pose.pose.orientation.y + odom.pose.pose.orientation.z * odom.pose.pose.orientation.z));
    }
    receive_traj_ = false;
    traj_queue = {};

}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "traj_server");
    ros::NodeHandle node;
    ros::NodeHandle nh("~");

    ros::Subscriber traj_sub = node.subscribe("bspline_traj", 10, trajCallback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber new_sub = node.subscribe("new", 10, newCallback);
    ros::Subscriber odom_sub = node.subscribe("odom", 50, odomCallbck, ros::TransportHints().tcpNoDelay());
    ros::Subscriber pg_T_vio_sub = node.subscribe("pg_T_vio", 10, pgTVioCallback);

    ros::Subscriber trigger_sub = node.subscribe( "traj_start_trigger", 100, trigger_callback, ros::TransportHints().tcpNoDelay());

    cmd_vis_pub = node.advertise<visualization_msgs::Marker>("position_cmd_vis", 10);
    pos_cmd_pub = node.advertise<quadrotor_msgs::PositionCommand>("position_cmd", 50);
    traj_pub = node.advertise<visualization_msgs::Marker>("travel_traj", 10);

    ros::Timer vis_timer = node.createTimer(ros::Duration(0.25), visCallback);

    nh.param("traj_server/pub_traj_id", pub_traj_id_, -1);
    nh.param("traj_server/isLoopCorrection", isLoopCorrection, false);
    nh.param("traj_server/use_vel_yaw", use_vel_yaw, false);

    ROS_INFO("[Traj server]: init...");
    ros::Duration(1.0).sleep();

    R_loop = Eigen::Quaterniond(1, 0, 0, 0).toRotationMatrix();
    T_loop = Eigen::Vector3d(0, 0, 0);

    ROS_INFO("[Traj server]: ready.");
    ros::spin();

    return 0;
}