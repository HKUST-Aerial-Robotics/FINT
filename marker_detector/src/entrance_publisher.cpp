#include <iostream>
#include <cmath>
#include <memory>
#include <ros/ros.h>
#include <ros/console.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Int16.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Vector3.h>
#include <Eigen/Eigen>

ros::Publisher mean_pose_pub;

double pos_x, pos_y, pos_z;
double ori_w, ori_x, ori_y, ori_z;

void detect_trigger_callback(const geometry_msgs::PoseStamped::ConstPtr &trigger)
{
    ROS_WARN("rcv trigger");

    geometry_msgs::PoseStamped mean_pose;
    mean_pose.header.stamp = trigger->header.stamp;
    mean_pose.header.frame_id = "world";

    mean_pose.pose.position.x = pos_x;
    mean_pose.pose.position.y = pos_y;
    mean_pose.pose.position.z = pos_z;

    Eigen::Quaterniond q(ori_w, ori_x, ori_y, ori_z);
    q.normalize();

    mean_pose.pose.orientation.w = q.w();
    mean_pose.pose.orientation.x = q.x();
    mean_pose.pose.orientation.y = q.y();
    mean_pose.pose.orientation.z = q.z();

    mean_pose_pub.publish(mean_pose);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "entrance_publisher");
    ros::NodeHandle nh("~");

    nh.getParam("pos_x", pos_x);
    nh.getParam("pos_y", pos_y);
    nh.getParam("pos_z", pos_z);

    nh.getParam("ori_w", ori_w);
    nh.getParam("ori_x", ori_x);
    nh.getParam("ori_y", ori_y);
    nh.getParam("ori_z", ori_z);


    mean_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("mean_pose", 10);
    //pub_ar_odom = nh.advertise<nav_msgs::Odometry>("/detected_markers", 10);
    // ros::Subscriber sub_image = nh.subscribe("/camera/color/image_raw", 1, image_callback);
    //ros::Subscriber sub_cam_pose = nh.subscribe("camera_pose", 1, img_callback);
    ros::Subscriber detect_trigger_sub = nh.subscribe<geometry_msgs::PoseStamped>("detect_trigger", 10, detect_trigger_callback);
    
    ros::spin();
}