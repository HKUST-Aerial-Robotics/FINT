/*******************************************************
Copyright (C) 2025, Aerial Robotics Group, Hong Kong University of Science and Technology
This file is part of FINT.
Licensed under the GNU General Public License v3.0;
you may not use this file except in compliance with the License.
*******************************************************/

#include <tunnel_planner.h>


int main(int argc, char** argv){
    
    ros::init(argc, argv, "tunnel_planner_node");
    ros::NodeHandle n("~");

    if (argc < 2) {
        printf("please intput: rosrun tunnel_planner tunnel_planner [config file] \n");
        return 1;
    }
    string config_file = argv[1];
    std::cout << "cf:" << config_file << std::endl;

    readParameters(config_file);
    
    tunnel_planner::tunnel_planner tp(n);

    ros::Rate r(100.0);

    while(ros::ok()){

        r.sleep();
        ros::spinOnce();
    
    }
    return 1;
}