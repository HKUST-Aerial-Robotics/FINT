#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <iostream>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Vector3.h>
#include <math.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/console.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <Eigen/Eigen>
#include <random>

#include "bspline_optimizer.h"
#include "non_uniform_bspline.h"
#include "voxel_map.h"

using namespace std;
using namespace Eigen;

ros::Publisher tunnel_center_line_pub, map2_pub;

string way_pt_file_path;
string out_file_path;

void save_map(pcl::PointCloud<pcl::PointXYZ>& cloud) {
  
  pcl::io::savePCDFileASCII(out_file_path + std::string("_tmp.pcd"), cloud);

  cout << "map saved." << endl;
}

vector<Vector3d> read_way_pt_file(const string& fname){
 
	vector<Vector3d> way_pts;
	vector<double> row;
	string line, word;
 
	fstream file (fname, ios::in);
	if(file.is_open())
	{
		while(getline(file, line))
		{
			row.clear();
 
			stringstream str(line);
 
			while(getline(str, word, ','))
				row.emplace_back(stod(word));
			way_pts.emplace_back(row[0], row[1], row[2]);
		}
	}
	else
		cout<<"Could not open the file\n";
 
    cout<<"way pts:\n";
	for(int i=0;i<way_pts.size();i++)
	{		
		cout<<way_pts[i].transpose()<<"\n";
	}

    return way_pts;
}

NonUniformBspline generate_tunnel_center_line(const vector<Vector3d>& way_pts){

    double speed = 1.0;

    vector<Vector3d> start(3, Vector3d::Zero());
    start[0] = way_pts.front();
    start[1] = (way_pts[1] - way_pts[0]).normalized() * speed;

    vector<Vector3d> end(3, Vector3d::Zero());
    end[0] = way_pts.back();
    end[1] = (way_pts[way_pts.size() - 1] - way_pts[way_pts.size() - 2]).normalized() * speed;

    vector<Vector3d> start_end_derivative(4, Vector3d::Zero());
    start_end_derivative[0] = start[1];
    start_end_derivative[2] = end[1];

    double total_dist = 0.0;
    for(unsigned int i = 1; i < way_pts.size(); i++){
        total_dist += (way_pts[i] - way_pts[i - 1]).norm();
    }
    double ts = total_dist / speed / (way_pts.size() - 1);

    MatrixX3d ctrl_pts;
    NonUniformBspline::parameterizeToBspline(ts, way_pts, start_end_derivative, 3, ctrl_pts);


    Matrix3d start_ctrl_pts;
    NonUniformBspline::solveFirst3CtrlPts(ts, start, start_ctrl_pts);

    ctrl_pts.topRows(3) = start_ctrl_pts.topRows(3);

    cout<<"ctrl pts:\n"<<ctrl_pts<<endl;


    NonUniformBspline pos_init = NonUniformBspline(ctrl_pts, 3, ts);


    vector<int> way_pt_idx;
    for (int i = 0; i < way_pts.size(); way_pt_idx.emplace_back(i), i++);

    // int cost_function = bspline_optimizer_->WAY_PT_ACC_VEL_PHASE;
    BsplineOptimizer bopt;
    bopt.setParam(speed);
    int cost_function = bopt.WAY_PT_JERK_VEL_START_HARD_PHASE;
    bopt.setBoundaryStates(start, end);
    bopt.setWaypoints(way_pts, way_pt_idx);

    bopt.optimize(ctrl_pts, ts, cost_function, 0, 0);

    return NonUniformBspline(ctrl_pts, 3, ts);
}

Matrix3d cal_w_R_plane(const Vector3d &dir)
{

    Vector3d normal_dir = dir.normalized();

    Vector3d plane_x_dir = Vector3d::UnitX();
    Vector3d plane_y_dir = Vector3d::UnitY();

    // LUF coord
    // if normal ~ up/down, X?F coord

    if(abs(normal_dir.z()) > 0.99){
        // approx z dir
        plane_y_dir =  (normal_dir.cross(plane_x_dir)).normalized();
        plane_x_dir = (plane_y_dir.cross(normal_dir)).normalized();

        // Y?F coord
        plane_x_dir = Vector3d::UnitY();
        plane_y_dir =  (normal_dir.cross(plane_x_dir)).normalized();
        plane_x_dir = (plane_y_dir.cross(normal_dir)).normalized();

    }
    else{
        plane_x_dir = (Vector3d::UnitZ().cross(normal_dir)).normalized();
        // if(plane_x_dir.x() < 0.0){
        //     plane_x_dir *= -1.0;
        // }
        plane_y_dir = (normal_dir.cross(plane_x_dir)).normalized();
    }

    Matrix3d w_R_plane;

    w_R_plane.col(0) = plane_x_dir;
    w_R_plane.col(1) = plane_y_dir;
    w_R_plane.col(2) = normal_dir;

    return w_R_plane;
}

void pub_tunnel_center_line_vis(NonUniformBspline &bspline)
{
    if (bspline.getControlPoint().size() == 0)
        return;

    nav_msgs::Path traj_path;
    traj_path.header.stamp = ros::Time::now();
    traj_path.header.frame_id = "world";
    geometry_msgs::PoseStamped traj_pt_pose;
    traj_pt_pose.header.frame_id = "world";
    traj_pt_pose.header.seq = 0;
    traj_pt_pose.pose.orientation.w = 1.0;
    traj_pt_pose.pose.orientation.x = 0.0;
    traj_pt_pose.pose.orientation.y = 0.0;
    traj_pt_pose.pose.orientation.z = 0.0;

    //   vector<Eigen::Vector3d> traj_pts;
    double dur = bspline.getTimeSum();

    NonUniformBspline vel = bspline.getDerivative();

    for (double t = 0.0; t <= dur; t += 0.01)
    {
        Eigen::Vector3d pt = bspline.evaluateDeBoorT(t);
        Eigen::Quaterniond ori = Quaterniond(cal_w_R_plane(vel.evaluateDeBoorT(t)));

        traj_pt_pose.header.stamp = ros::Time(t);
        traj_pt_pose.pose.position.x = pt.x();
        traj_pt_pose.pose.position.y = pt.y();
        traj_pt_pose.pose.position.z = pt.z();

        traj_pt_pose.pose.orientation.w = ori.w();
        traj_pt_pose.pose.orientation.x = ori.x();
        traj_pt_pose.pose.orientation.y = ori.y();
        traj_pt_pose.pose.orientation.z = ori.z();

        traj_pt_pose.header.seq++;

        traj_path.poses.push_back(traj_pt_pose);



        // traj_pts.push_back(pt);
    }

    tunnel_center_line_pub.publish(traj_path);
}


void pub_occ_map(voxel_map<bool>& vm){
    
    pcl::PointXYZ pt;
    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.clear();

    for(unsigned int idx = 0; idx < vm.map_data.size(); idx++){
        if(vm.map_data[idx] > 0){
            Vector3d pos = vm.idx2pos(idx);
            pt.x = pos.x();
            pt.y = pos.y();
            pt.z = pos.z();
            cloud.push_back(pt);
        }
    }

    cloud.width = cloud.points.size();
    cloud.height = 1;
    cloud.is_dense = true;
    cloud.header.frame_id = "world";
    sensor_msgs::PointCloud2 cloud_msg;

    pcl::toROSMsg(cloud, cloud_msg);
    cloud_msg.header.stamp = ros::Time::now();

    map2_pub.publish(cloud_msg);

    save_map(cloud);
}

void fill_map(NonUniformBspline& tunnel_center_line, const MatrixXi& cs, voxel_map<bool>& vm){
    double dur = tunnel_center_line.getTimeSum();
    double t_res = 0.01;

    NonUniformBspline tunnel_vel = tunnel_center_line.getDerivative();

    int cs_row_center = cs.rows() / 2;
    int cs_col_center = cs.cols() / 2;
    
    double res = vm.map_res;

    for(double t = 0.0; t <= dur-0.5; t += t_res){
        Vector3d tunnel_center_pos = tunnel_center_line.evaluateDeBoorT(t);
        Vector3d tunnel_dir = tunnel_vel.evaluateDeBoorT(t).normalized();
        Matrix3d w_R_plane = cal_w_R_plane(tunnel_dir);


        for(int r = 0; r < cs.rows(); r++){
            for(int c = 0; c < cs.cols(); c++){
                Vector3d pt_in_cs = Vector3d(r - cs_row_center, c - cs_col_center, 0.0) * res;
                Vector3d pt_in_w = w_R_plane * pt_in_cs + tunnel_center_pos;

                if(vm.in_map(pt_in_w)){
                    vm.map_data[vm.pos2idx(pt_in_w)] = (cs(r, c) > 0);
                }
                
            }
        }

    }

    pub_occ_map(vm);

}


int main(int argc, char** argv) {
    ros::init(argc, argv, "tunnel_generator");
    ros::NodeHandle node("~");

    if (argc < 3) {
        std::cout << "File path not specified" << std::endl;
        return 0;
    }

    way_pt_file_path = argv[1];
    out_file_path = argv[2];

    // ros::Subscriber cloud_sub = node.subscribe("/map_generator/global_cloud",
    // 10, cloudCallback);
    // ros::Subscriber cloud_sub =
    // node.subscribe("/firefly/nbvPlanner/octomap_pcl", 10, cloudCallback);

    // Generate map by clicking
    // ros::Subscriber cloud_sub = node.subscribe("/map_generator/click_map", 10, cloudCallback);
    tunnel_center_line_pub = node.advertise<nav_msgs::Path>("generated_tunnel_center_line", 10, true);
    map2_pub = node.advertise<sensor_msgs::PointCloud2>("generated_tunnel_map", 10, true);
    //   ros::Subscriber cloud2_sub = node.subscribe("/local_occ_map/occ_map2", 10, cloud2Callback);
    ros::Duration(1.0).sleep();

    vector<Vector3d> way_pts = read_way_pt_file(way_pt_file_path);
    NonUniformBspline tunnel_center_line = generate_tunnel_center_line(way_pts);
    pub_tunnel_center_line_vis(tunnel_center_line);

    double map_res = 0.02;
    Matrix<double,3,2> MAP_LIM;
    MAP_LIM << -10.0, 10.0,
                -3.0, 8.0,
                -1.0, 4.0;
    voxel_map<bool> vm(map_res, MAP_LIM);

    // double tunnel_radius = 0.30;
    // int tunnel_radius_pixel = static_cast<int>(round(tunnel_radius / map_res));
    // int cs_size = 2 * tunnel_radius_pixel + 3;
    
    // MatrixXi cs = MatrixXi::Zero(cs_size, cs_size);
    // int cs_center = tunnel_radius_pixel + 1;
    // for(double x = 0.0; x <= tunnel_radius; x += map_res){
    //     double y = sqrt(tunnel_radius * tunnel_radius - x * x);
    //     int x_pixel = static_cast<int>(round(x / map_res));
    //     int y_pixel = static_cast<int>(round(y / map_res));

    //     cs(cs_center + x_pixel, cs_center + y_pixel) = 1;
    //     cs(cs_center + x_pixel, cs_center - y_pixel) = 1;
    //     cs(cs_center - x_pixel, cs_center + y_pixel) = 1;
    //     cs(cs_center - x_pixel, cs_center - y_pixel) = 1;

    //     cs(cs_center + y_pixel, cs_center + x_pixel) = 1;
    //     cs(cs_center + y_pixel, cs_center - x_pixel) = 1;
    //     cs(cs_center - y_pixel, cs_center + x_pixel) = 1;
    //     cs(cs_center - y_pixel, cs_center - x_pixel) = 1;
    // }

    double tunnel_width = 0.70;
    double tunnel_height = 0.50;

    int tunnel_width_pixel = static_cast<int>(round(tunnel_width / map_res));
    int tunnel_height_pixel = static_cast<int>(round(tunnel_height / map_res));
    int cs_size = 2 * max(tunnel_width_pixel, tunnel_height_pixel) + 3;
    
    MatrixXi cs = MatrixXi::Zero(cs_size, cs_size);
    int cs_center = cs_size / 2;

    for(double x = -0.5 * tunnel_width; x <= 0.5 * tunnel_width; x += map_res){
        double y = 0.5 * tunnel_height;

        int x_pixel = static_cast<int>(round(x / map_res));
        int y_pixel = static_cast<int>(round(y / map_res));

        cs(cs_center + x_pixel, cs_center + y_pixel) = 1;
        cs(cs_center + x_pixel, cs_center - y_pixel) = 1;
    }

    for(double y = -0.5 * tunnel_height; y <= 0.5 * tunnel_height; y += map_res){
        double x = 0.5 * tunnel_width;

        int x_pixel = static_cast<int>(round(x / map_res));
        int y_pixel = static_cast<int>(round(y / map_res));

        cs(cs_center + x_pixel, cs_center + y_pixel) = 1;
        cs(cs_center - x_pixel, cs_center + y_pixel) = 1;
    }

    fill_map(tunnel_center_line, cs, vm);

    cout<<"\ncs:\n"<<cs<<endl;

    while (ros::ok()) {
        ros::spinOnce();
        ros::Duration(0.1).sleep();
    }

    // cout << "finish record map." << endl;
    ROS_WARN("[Map Recorder]: finish record map.");
    return 0;
}