/*******************************************************
Copyright (C) 2025, Aerial Robotics Group, Hong Kong University of Science and Technology
This file is part of FINT.
Licensed under the GNU General Public License v3.0;
you may not use this file except in compliance with the License.
*******************************************************/

#include "parameters.h"

double MAP_RES;
Matrix<double,3,2> MAP_LIM;
Matrix<double,3,2> LOCAL_EDF_LIM;

double xmin, xmax, ymin, ymax, zmin, zmax;
double local_edf_xmin, local_edf_xmax, local_edf_ymin, local_edf_ymax, local_edf_zmin, local_edf_zmax;

double PROB_HIT;
double PROB_MISS;
double CLAMP_MIN;
double CLAMP_MAX;
double MIN_OCCUPANCY;

double UPDATE_FREQ;

int DEPTH_MARGIN;
int NUM_PIXEL_SKIP;
double DEPTH_SCALE;
double MIN_DEPTH;
double MAX_DEPTH;
double MAX_RAY_LENGTH;

double DRONE_DIM;

double HOUGH_CIRCLE_THRESHOLD, HOUGH_RECTANGLE_THRESHOLD;

int ADAPTIVE_SPEED = 0;

Vector3d TUNNEL_ENTRANCE_POS;
Vector3d TUNNEL_ENTRANCE_DIR;
double TUNNEL_DIM;
double TUNNEL_STEP_RES;
double CROSS_SECTION_STEP_RES;
double TUNNEL_WAY_PT_MIN_INTERVAL;
double GRAD_MAX_RES;
double PLAN_RANGE;
double FLIGHT_SPEED;
double VIRTUAL_FLIGHT_PROGRESS_SPEED;
double MAX_SPEED, MAX_ACC;
double MAX_ANGULAR_SPEED;
double MAX_YAW_DIR_CURVATURE_RATIO;
double YAW_AHEAD_LENGTH;

double MAX_YAW_CHANGE_OVER_DISTANCE;
double MAX_YAW_CENTER_LINE_DIR_DIFF;

double VERT_SECTION_COS_THRESHOLD;

vector<pair<Eigen::MatrixXf, Eigen::VectorXf>> CIRCLE_LINEAR_LAYERS;
vector<pair<Eigen::MatrixXf, Eigen::VectorXf>> RECT_LINEAR_LAYERS;

nn SHAPE_CLASSIFIER_NET;

std::vector<camera_module_info> CAM_INFO_VEC;
double OPTICAL_FLOW_CAL_RES;
double MAX_RAYCAST_LENGTH;

double tunnel_entrance_x, tunnel_entrance_y, tunnel_entrance_z;
double tunnel_entrance_dir_x, tunnel_entrance_dir_y, tunnel_entrance_dir_z;

int QUAD_ALGORITHM_ID;
int NON_QUAD_ALGORITHM_ID;
int BSPLINE_DEGREE;

double W_DISTURBANCE;
double W_VISION;
double W_SMOOTH_1d_JERK;
double W_SMOOTH_1d_ACC;
double W_SMOOTH_YAW;
double W_SMOOTH_3d;
double W_FEASI;
double W_INTERVAL;
double W_DIST;
double W_START;
double W_END;
double W_END_HARD;
double W_GUIDE;
double W_WAYPT;
double W_TIME;
double W_YAW_WAYPT;

double W_HEURISTIC;

double DISTANCE_COST_ORIGIN;

int MAX_ITERATION_NUM1;
int MAX_ITERATION_NUM2;
int MAX_ITERATION_NUM3;
int MAX_ITERATION_NUM4;

double MAX_ITERATION_TIME1;
double MAX_ITERATION_TIME2;
double MAX_ITERATION_TIME3;
double MAX_ITERATION_TIME4;

double REPLAN_FREQ;

int USE_EXACT_TIME_SYNC;

double TIME_COMMIT;

template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

void readParametersNet(std::string config_file, vector<pair<Eigen::MatrixXf, Eigen::VectorXf>>& linear_layers)
{
    FILE *fh = fopen(config_file.c_str(),"r");
    if(fh == NULL){
        printf("config_file dosen't exist; wrong config_file path");
        return;
    }
    fclose(fh);
    
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to net settings" << std::endl;
    }
    
    int num_layers = fsSettings["num_layers"];
    cout<<"num_layers: "<<num_layers<<endl;
    
    linear_layers.clear();
    
    for(int layer_idx = 0; layer_idx < num_layers; layer_idx += 2){
        cv::Mat cv_weight, cv_bias;
        fsSettings[string("layers_")+to_string(layer_idx)+string("_weight")] >> cv_weight;
        fsSettings[string("layers_")+to_string(layer_idx)+string("_bias")] >> cv_bias;
        
        Eigen::MatrixXf weight(static_cast<int>(cv_weight.rows), static_cast<int>(cv_weight.cols));
        Eigen::VectorXf bias(static_cast<int>(cv_bias.rows));
        cv::cv2eigen(cv_weight, weight);
        cv::cv2eigen(cv_bias, bias);
        
        linear_layers.emplace_back(make_pair(weight, bias));
    }
    
    fsSettings.release();
}


int retrieve_layer_type(const string& layer_type_string){

    if(layer_type_string == string("RELU")) return RELU;
    if(layer_type_string == string("SIGMOID")) return SIGMOID;
    if(layer_type_string == string("SOFTMAX")) return SOFTMAX;
    if(layer_type_string == string("FLATTEN")) return FLATTEN;
    if(layer_type_string == string("LINEAR")) return LINEAR;
    if(layer_type_string == string("CONV2D")) return CONV2D;
    if(layer_type_string == string("MAXPOOL2D")) return MAXPOOL2D;
    
    return UNKNOWN;
}

linear_layer* read_linear_layer_parameter(const cv::FileStorage& fsSettings, int layer_idx){
    
    cv::Mat cv_weight, cv_bias;
    fsSettings[string("layers_")+to_string(layer_idx)+string("_weight")] >> cv_weight;
    fsSettings[string("layers_")+to_string(layer_idx)+string("_bias")] >> cv_bias;

    
    Eigen::MatrixXf weight(static_cast<int>(cv_weight.rows), static_cast<int>(cv_weight.cols));
    Eigen::VectorXf bias(static_cast<int>(cv_bias.rows));
    cv::cv2eigen(cv_weight, weight);
    cv::cv2eigen(cv_bias, bias);
    
    return new linear_layer(weight, bias);
}

conv2d_layer* read_conv2d_layer_parameter(const cv::FileStorage& fsSettings, int layer_idx){
    
    cv::Mat cv_weight, cv_bias;
    fsSettings[string("layers_")+to_string(layer_idx)+string("_weight")] >> cv_weight;
    fsSettings[string("layers_")+to_string(layer_idx)+string("_bias")] >> cv_bias;
    
    auto mat_size = cv_weight.size;
//    cout<<cv_weight.size<<endl;
    
    Eigen::Matrix<Eigen::MatrixXf, Eigen::Dynamic, Eigen::Dynamic> weight(mat_size[0], mat_size[1]);
    
    float* data_ptr = reinterpret_cast<float*>(cv_weight.data);
    for(int i = 0; i < mat_size[0]; i++){
        for(int j = 0; j < mat_size[1]; j++){
            auto& kernel = weight(i,j);
            kernel.resize(mat_size[2], mat_size[3]);
            
            for(int k = 0; k < mat_size[2]; k++){
                for(int l = 0; l < mat_size[3]; l++){
                    kernel(k,l) = *(data_ptr);
                    data_ptr++;
                }
            }
            
        }
    }

//    for(int i = 0; i < weight.rows(); i++){
//        for(int j = 0; j < weight.cols(); j++){
//            cout<<"kernel "<<i<<"\t"<<j<<endl;
//            cout<<weight(i,j)<<endl<<endl;
//
//        }
//    }
    
//    Eigen::MatrixX<Eigen::MatrixXf> weight(static_cast<int>(cv_weight.rows), static_cast<int>(cv_weight.cols));
    Eigen::VectorXf bias(static_cast<int>(cv_bias.rows));
//    cv::cv2eigen(cv_weight, weight);
    cv::cv2eigen(cv_bias, bias);
    
//    cout<<"bias:\n"<<bias<<endl;
    
    Eigen::Vector2i strides(1,1);
    
    return new conv2d_layer(weight, bias, strides);
}

maxpool2d_layer* read_maxpool2d_layer_parameter(const cv::FileStorage& fsSettings, int layer_idx){
    
    int kernel_size_row = 1, kernel_size_col = 1;
    int stride_row = 1, stride_col = 1;
    int padding = 0;
    fsSettings[string("layers_")+to_string(layer_idx)+string("_kernel_size_row")] >> kernel_size_row;
    fsSettings[string("layers_")+to_string(layer_idx)+string("_kernel_size_col")] >> kernel_size_col;
    fsSettings[string("layers_")+to_string(layer_idx)+string("_stride_row")] >> stride_row;
    fsSettings[string("layers_")+to_string(layer_idx)+string("_stride_col")] >> stride_col;
    fsSettings[string("layers_")+to_string(layer_idx)+string("_padding")] >> padding;
    
    Eigen::Vector2i kernel_size(kernel_size_row, kernel_size_col);
    Eigen::Vector2i stride(stride_row, stride_col);
    
    return new maxpool2d_layer(kernel_size, stride, padding > 0);
}

nn readParametersCNN(std::string config_file)
{
    vector<shared_ptr<net_layer>> layer_ptrs;
    FILE *fh = fopen(config_file.c_str(),"r");
    if(fh == NULL){
        printf("config_file dosen't exist; wrong config_file path");
        return nn();
    }
    fclose(fh);
    
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }
    
    int num_layers = fsSettings["num_layers"];

    int input_rows = fsSettings["input_rows"];
    int input_cols = fsSettings["input_cols"];
    int input_channels = fsSettings["input_channels"];
    
    layer_ptrs.clear();
    
    string layer_type_string;
    for(int layer_idx = 0; layer_idx < num_layers; layer_idx++){
        fsSettings[string("layers_")+to_string(layer_idx)+string("_type")] >> layer_type_string;
        
        int layer_type = retrieve_layer_type(layer_type_string);
        
        switch(layer_type){
            case RELU:
                layer_ptrs.emplace_back(new relu_layer());
                break;
                
            case SIGMOID:
                layer_ptrs.emplace_back(new sigmoid_layer());
                break;
                
            case SOFTMAX:
                layer_ptrs.emplace_back(new softmax_layer());
                break;
                
            case FLATTEN:
                layer_ptrs.emplace_back(new flatten_layer());
                break;
                
            case LINEAR:
                layer_ptrs.emplace_back(read_linear_layer_parameter(fsSettings, layer_idx));
                break;
            case CONV2D:
                layer_ptrs.emplace_back(read_conv2d_layer_parameter(fsSettings, layer_idx));
                break;
            case MAXPOOL2D:
                layer_ptrs.emplace_back(read_maxpool2d_layer_parameter(fsSettings, layer_idx));
                break;
                
            default:
                break;
        }
        
    }
    
    return nn(layer_ptrs, input_rows, input_cols, input_channels);
    
}

void readParameters(std::string config_file)
{
    FILE *fh = fopen(config_file.c_str(),"r");
    if(fh == NULL){
        ROS_WARN("config_file dosen't exist; wrong config_file path");
        ROS_BREAK();
        return;          
    }
    fclose(fh);

    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    int pn = config_file.find_last_of('/');
    std::string configPath = config_file.substr(0, pn);

    fsSettings["map_res"] >> MAP_RES;
    
    fsSettings["xmin"] >> xmin;
    fsSettings["xmax"] >> xmax;
    fsSettings["ymin"] >> ymin;
    fsSettings["ymax"] >> ymax;
    fsSettings["zmin"] >> zmin;
    fsSettings["zmax"] >> zmax;

    MAP_LIM << xmin, xmax, ymin, ymax, zmin, zmax;

    fsSettings["local_edf_xmin"] >> local_edf_xmin;
    fsSettings["local_edf_xmax"] >> local_edf_xmax;
    fsSettings["local_edf_ymin"] >> local_edf_ymin;
    fsSettings["local_edf_ymax"] >> local_edf_ymax;
    fsSettings["local_edf_zmin"] >> local_edf_zmin;
    fsSettings["local_edf_zmax"] >> local_edf_zmax;

    LOCAL_EDF_LIM << local_edf_xmin, local_edf_xmax, local_edf_ymin, local_edf_ymax, local_edf_zmin, local_edf_zmax;


    cout<<"MAP_RES: "<<MAP_RES<<endl;
    cout<<"MAP_LIM: \n"<<MAP_LIM<<endl;
    cout<<"LOCAL_EDF_LIM: \n"<<LOCAL_EDF_LIM<<endl;

    fsSettings["drone_dimension"] >> DRONE_DIM;
    cout<<"DRONE_DIM: "<<DRONE_DIM<<endl;

    fsSettings["tunnel_entrance_x"] >> tunnel_entrance_x;
    fsSettings["tunnel_entrance_y"] >> tunnel_entrance_y;
    fsSettings["tunnel_entrance_z"] >> tunnel_entrance_z;
    fsSettings["tunnel_entrance_dir_x"] >> tunnel_entrance_dir_x;
    fsSettings["tunnel_entrance_dir_y"] >> tunnel_entrance_dir_y;
    fsSettings["tunnel_entrance_dir_z"] >> tunnel_entrance_dir_z;

    TUNNEL_ENTRANCE_POS = Vector3d(tunnel_entrance_x, tunnel_entrance_y, tunnel_entrance_z);
    TUNNEL_ENTRANCE_DIR = Vector3d(tunnel_entrance_dir_x, tunnel_entrance_dir_y, tunnel_entrance_dir_z).normalized();

    fsSettings["adaptive_speed"] >> ADAPTIVE_SPEED;

    fsSettings["max_tunnel_dimension"] >> TUNNEL_DIM;
    fsSettings["tunnel_step_res"] >> TUNNEL_STEP_RES;
    fsSettings["cross_section_step_res"] >> CROSS_SECTION_STEP_RES;
    fsSettings["tunnel_way_pt_min_interval"] >> TUNNEL_WAY_PT_MIN_INTERVAL;
    fsSettings["grad_max_res"] >> GRAD_MAX_RES;
    fsSettings["plan_range"] >> PLAN_RANGE;

    if(!ADAPTIVE_SPEED){
        fsSettings["flight_speed"] >> FLIGHT_SPEED;
    }

    fsSettings["virtual_flight_progress_speed"] >> VIRTUAL_FLIGHT_PROGRESS_SPEED;
    fsSettings["max_speed"] >> MAX_SPEED;
    fsSettings["max_acc"] >> MAX_ACC;
    fsSettings["max_anguar_speed"] >> MAX_ANGULAR_SPEED;
    fsSettings["max_yaw_dir_curvature_ratio"] >> MAX_YAW_DIR_CURVATURE_RATIO;
    fsSettings["yaw_ahead_length"] >> YAW_AHEAD_LENGTH;

    fsSettings["max_yaw_change_over_distance"] >> MAX_YAW_CHANGE_OVER_DISTANCE;
    fsSettings["max_yaw_center_line_dir_diff"] >> MAX_YAW_CENTER_LINE_DIR_DIFF;

    MAX_YAW_CHANGE_OVER_DISTANCE *= M_PI / 180.0;
    MAX_YAW_CENTER_LINE_DIR_DIFF *= M_PI / 180.0;
    

    fsSettings["hough_circle_threshold"] >> HOUGH_CIRCLE_THRESHOLD;
    fsSettings["hough_rectangle_threshold"] >> HOUGH_RECTANGLE_THRESHOLD;
    fsSettings["vert_section_cos_threshold"] >> VERT_SECTION_COS_THRESHOLD;

    cout << "TUNNEL_ENTRANCE_POS: " << TUNNEL_ENTRANCE_POS.transpose() << endl;
    cout << "TUNNEL_ENTRANCE_DIR: " << TUNNEL_ENTRANCE_DIR.transpose() << endl;

    cout << "TUNNEL_DIM: " << TUNNEL_DIM << endl;
    cout << "TUNNEL_STEP_RES: " << TUNNEL_STEP_RES << endl;
    cout << "CROSS_SECTION_STEP_RES: " << CROSS_SECTION_STEP_RES << endl;
    cout << "TUNNEL_WAY_PT_MIN_INTERVAL: " << TUNNEL_WAY_PT_MIN_INTERVAL << endl;
    cout << "GRAD_MAX_RES: " << GRAD_MAX_RES << endl;
    cout << "PLAN_RANGE: " << PLAN_RANGE << endl;
    cout << "FLIGHT_SPEED: " << FLIGHT_SPEED << endl;
    cout << "MAX_SPEED: " << MAX_SPEED << endl;
    cout << "MAX_ACC: " << MAX_ACC << endl;

    fsSettings["quad_algorithm_id"] >> QUAD_ALGORITHM_ID;
    fsSettings["non_quad_algorithm_id"] >> NON_QUAD_ALGORITHM_ID;
    fsSettings["bspline_degree"] >> BSPLINE_DEGREE;
    fsSettings["w_smooth_1d_jerk"] >> W_SMOOTH_1d_JERK;
    fsSettings["w_smooth_1d_acc"] >> W_SMOOTH_1d_ACC;
    fsSettings["w_smooth_yaw"] >> W_SMOOTH_YAW;
    fsSettings["w_smooth_3d"] >> W_SMOOTH_3d;
    fsSettings["w_interval"] >> W_INTERVAL;
    fsSettings["w_dist"] >> W_DIST;
    fsSettings["w_feasi"] >> W_FEASI;
    fsSettings["w_start"] >> W_START;
    fsSettings["w_end"] >> W_END;
    fsSettings["w_end_hard"] >> W_END_HARD;
    fsSettings["w_guide"] >> W_GUIDE;
    fsSettings["w_waypt"] >> W_WAYPT;
    fsSettings["w_time"] >> W_TIME;
    fsSettings["w_disturbance"] >> W_DISTURBANCE;
    fsSettings["w_vision"] >> W_VISION;
    fsSettings["w_yaw_waypt"] >> W_YAW_WAYPT;
    fsSettings["w_heuristic"] >> W_HEURISTIC;

    fsSettings["distance_cost_origin"] >> DISTANCE_COST_ORIGIN;

    fsSettings["max_iteration_num1"] >> MAX_ITERATION_NUM1;
    fsSettings["max_iteration_num2"] >> MAX_ITERATION_NUM2;
    fsSettings["max_iteration_num3"] >> MAX_ITERATION_NUM3;
    fsSettings["max_iteration_num4"] >> MAX_ITERATION_NUM4;
    fsSettings["max_iteration_time1"] >> MAX_ITERATION_TIME1;
    fsSettings["max_iteration_time2"] >> MAX_ITERATION_TIME2;
    fsSettings["max_iteration_time3"] >> MAX_ITERATION_TIME3;
    fsSettings["max_iteration_time4"] >> MAX_ITERATION_TIME4;
    fsSettings["replan_freq"] >> REPLAN_FREQ;
    fsSettings["use_exact_time_sync"] >> USE_EXACT_TIME_SYNC;
    fsSettings["time_commit"] >> TIME_COMMIT;

    std::string net_file_config; 
    fsSettings["circle_net_file"] >> net_file_config;
    if(!net_file_config.empty()){
        ROS_INFO_STREAM("read circle net: "+configPath+"/"+net_file_config);
        readParametersNet(configPath+"/"+net_file_config, CIRCLE_LINEAR_LAYERS);
    }
    fsSettings["rect_net_file"] >> net_file_config;
    if(!net_file_config.empty()){
        ROS_INFO_STREAM("read rect net: "+configPath+"/"+net_file_config);
        readParametersNet(configPath+"/"+net_file_config, RECT_LINEAR_LAYERS);
    }

    std::string classifier_file_config; 
    fsSettings["classifier_file"] >> classifier_file_config;
    if(!net_file_config.empty()){
        ROS_INFO_STREAM("read classifier net: "+configPath+"/"+classifier_file_config);
        SHAPE_CLASSIFIER_NET = readParametersCNN(configPath+"/"+classifier_file_config);
    }

    cv::FileNode cam_module_node = fsSettings["cam_module"];
    int num_cam_module = cam_module_node["num"];
    printf("camera module number %d\n", num_cam_module);
    CAM_INFO_VEC.resize(num_cam_module);

    cv::FileNode cam_modules_node = cam_module_node["modules"];
    int cur_cam_module = 0;
    for(cv::FileNodeIterator it = cam_modules_node.begin(); it != cam_modules_node.end(); it++, cur_cam_module++){
        
        if(cur_cam_module >= num_cam_module) break;

        CAM_INFO_VEC[cur_cam_module].img_cols_ = (*it)["image_width"];
        CAM_INFO_VEC[cur_cam_module].img_rows_ = (*it)["image_height"];
        CAM_INFO_VEC[cur_cam_module].fx_ = (*it)["cam_fx"];
        CAM_INFO_VEC[cur_cam_module].fy_ = (*it)["cam_fy"];
        CAM_INFO_VEC[cur_cam_module].cx_ = (*it)["cam_cx"];
        CAM_INFO_VEC[cur_cam_module].cy_ = (*it)["cam_cy"];

        CAM_INFO_VEC[cur_cam_module].depth_fx_ = (*it)["depth_fx"];
        CAM_INFO_VEC[cur_cam_module].depth_fy_ = (*it)["depth_fy"];
        CAM_INFO_VEC[cur_cam_module].depth_cx_ = (*it)["depth_cx"];
        CAM_INFO_VEC[cur_cam_module].depth_cy_ = (*it)["depth_cy"];

        (*it)["depth_topic"] >> CAM_INFO_VEC[cur_cam_module].depth_topic_;
        (*it)["cam_pose_topic"] >> CAM_INFO_VEC[cur_cam_module].cam_pose_topic_;
        
        cv::Mat cv_T;
        (*it)["imu_T_cam"] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        CAM_INFO_VEC[cur_cam_module].Tic_ = T;

        cv::Mat cv_R, cv_t;
        (*it)["cam_R_depth"] >> cv_R;
        (*it)["cam_t_depth"] >> cv_t;

        cv::cv2eigen(cv_R, CAM_INFO_VEC[cur_cam_module].cam_R_depth_);
        Quaterniond cam_Q_depth(CAM_INFO_VEC[cur_cam_module].cam_R_depth_);
        cam_Q_depth.normalize();
        CAM_INFO_VEC[cur_cam_module].cam_R_depth_ = cam_Q_depth.toRotationMatrix();
        cv::cv2eigen(cv_t, CAM_INFO_VEC[cur_cam_module].cam_t_depth_);
        
        
    }
    fsSettings["optical_flow_cal_res"] >> OPTICAL_FLOW_CAL_RES;
    fsSettings["max_ray_length"] >> MAX_RAYCAST_LENGTH;
    MAX_RAYCAST_LENGTH += MAP_RES;

    PROB_HIT = fsSettings["prob_hit"];
    PROB_MISS = fsSettings["prob_miss"];
    CLAMP_MIN = fsSettings["clamp_min"];
    CLAMP_MAX = fsSettings["clamp_max"];
    MIN_OCCUPANCY = fsSettings["min_occupancy"];

    UPDATE_FREQ = fsSettings["update_freq"];

    DEPTH_MARGIN = fsSettings["depth_margin"];
    NUM_PIXEL_SKIP = fsSettings["num_pixel_skip"];
    DEPTH_SCALE = fsSettings["depth_scale"];
    MIN_DEPTH = fsSettings["min_depth"];
    MAX_DEPTH = fsSettings["max_depth"];
    MAX_RAY_LENGTH = fsSettings["max_ray_length"];


    cout << "QUAD_ALGORITHM_ID: " << QUAD_ALGORITHM_ID << endl;
    cout << "NON_QUAD_ALGORITHM_ID: " << NON_QUAD_ALGORITHM_ID << endl;
    cout << "BSPLINE_DEGREE: " << BSPLINE_DEGREE << endl;
    cout << "W_SMOOTH_1d_JERK: " << W_SMOOTH_1d_JERK << endl;
    cout << "W_SMOOTH_1d_ACC: " << W_SMOOTH_1d_ACC << endl;
    cout << "W_SMOOTH_YAW: " << W_SMOOTH_YAW << endl;
    cout << "W_SMOOTH_3d: " << W_SMOOTH_3d << endl;
    cout << "W_FEASI: " << W_FEASI << endl;
    cout << "W_START: " << W_START << endl;
    cout << "W_END: " << W_END << endl;
    cout << "W_GUIDE: " << W_GUIDE << endl;
    cout << "W_WAYPT: " << W_WAYPT << endl;
    cout << "W_TIME: " << W_TIME << endl;
    cout << "MAX_ITERATION_NUM1: " << MAX_ITERATION_NUM1 << endl;
    cout << "MAX_ITERATION_NUM2: " << MAX_ITERATION_NUM2 << endl;
    cout << "MAX_ITERATION_NUM3: " << MAX_ITERATION_NUM3 << endl;
    cout << "MAX_ITERATION_NUM4: " << MAX_ITERATION_NUM4 << endl;
    cout << "MAX_ITERATION_TIME1: " << MAX_ITERATION_TIME1 << endl;
    cout << "MAX_ITERATION_TIME2: " << MAX_ITERATION_TIME2 << endl;
    cout << "MAX_ITERATION_TIME3: " << MAX_ITERATION_TIME3 << endl;
    cout << "MAX_ITERATION_TIME4: " << MAX_ITERATION_TIME4 << endl;
    cout << "REPLAN_FREQ: " << REPLAN_FREQ << endl;
    cout<<"USE_EXACT_TIME_SYNC: "<<USE_EXACT_TIME_SYNC<<endl;

    fsSettings.release();
}
