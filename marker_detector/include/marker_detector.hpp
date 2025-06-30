#include <iostream>
#include <cmath>
#include <memory>
#include <ros/ros.h>
#include <ros/console.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Int16.h>
#include <cv_bridge/cv_bridge.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Vector3.h>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <Eigen/Eigen>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>

using namespace std;
using namespace cv;
using namespace Eigen;


class marker_detector{
public:
    marker_detector(ros::NodeHandle& nh): nh_(nh){
        mean_pose_pub_ = nh.advertise<geometry_msgs::PoseStamped>("mean_pose", 10);
    }

    void read_param(std::string config_file);

    // typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image, geometry_msgs::PoseStamped> ColorDepthPoseSyncPolicy;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, geometry_msgs::PoseStamped> ColorDepthPoseSyncPolicy;

    class cam_module{
    public:

        cam_module(marker_detector* detector_ptr): detector_ptr_(detector_ptr), K_(cv::Mat::eye(3,3,CV_64F)), D_(cv::Mat::zeros(4,1,CV_64F)){}

        void color_depth_pose_callback(const sensor_msgs::ImageConstPtr &color_img_ptr, const sensor_msgs::ImageConstPtr &depth_img_ptr, const geometry_msgs::PoseStampedConstPtr &cam_pose_ptr);
        cv::Mat K_, D_;

        std::string img0_topic_;
        std::string img1_topic_;
        std::string cam_pose_topic_;
    
        unique_ptr<message_filters::Subscriber<sensor_msgs::Image>> sub_color_ptr_;
        unique_ptr<message_filters::Subscriber<sensor_msgs::Image>> sub_depth_ptr_;
        unique_ptr<message_filters::Subscriber<geometry_msgs::PoseStamped>> sub_cam_pose_ptr_;

        unique_ptr<message_filters::Synchronizer<ColorDepthPoseSyncPolicy>> sync_color_depth_pose_ptr_;

        marker_detector* detector_ptr_;

    };

private:

    void register_sub();

    Vector3d plane_dir_fitting(MatrixXd &pts_mat);
    bool get_mean_pos();

    void detect_trigger_callback(const geometry_msgs::PoseStamped::ConstPtr &trigger);

    void publish_mean_pose(const Vector3d& cam_dir ,const ros::Time& stamp);

    ros::Subscriber detect_trigger_sub_;

    ros::Publisher mean_pose_pub_;

    double marker_size_;
    double required_sample_;
    double output_min_sample_;
    double pos_tol_;
    double inlayer_tol_;

    MatrixXd mean_marker_pos_;

    vector<int> marker_id_;
    vector<list<Vector3d>> marker_pos_buffer_;

    vector<cam_module> cam_modules_;

    ros::NodeHandle& nh_;

    bool detecting_ = false;
};
