#include "marker_detector.hpp"


int main(int argc, char **argv)
{
    ros::init(argc, argv, "marker_detector");
    ros::NodeHandle nh("~");
    string config_file;
    nh.getParam("config_file", config_file);
    
    marker_detector mkd(nh);
    mkd.read_param(config_file);

    
    ros::spin();
}
