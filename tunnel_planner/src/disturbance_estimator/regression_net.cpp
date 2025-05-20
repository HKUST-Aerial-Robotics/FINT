/*******************************************************
Copyright (C) 2025, Aerial Robotics Group, Hong Kong University of Science and Technology
This file is part of FINT.
Licensed under the GNU General Public License v3.0;
you may not use this file except in compliance with the License.
*******************************************************/
//
//  regression_net.cpp
//  regression_net
//
//  Created by Luqi on 27/1/2023.
//

#include "disturbance_estimator/regression_net.hpp"

Eigen::VectorXf regression_net::relu(Eigen::VectorXf& input){
    Eigen::VectorXf output = input;

    for(int i = 0; i < output.size(); i++){
        float& c  =  output(i);
        c = c > 0.0 ? c : 0.0;
    }
    // for(float& c : output){
    //    *((int*)(&c)) &= ((*((int*)(&c)) + int(0x80000000))>>31);
    // }
    return output;
}

Eigen::VectorXf regression_net::linear(Eigen::VectorXf& input, Eigen::MatrixXf& A, Eigen::VectorXf& b){
    return A * input + b;
}

double regression_net::sigmoid(float input){
    return 1.0 / (1.0 + exp(-input));
}


regression_net::regression_net(vector<pair<Eigen::MatrixXf, Eigen::VectorXf>>& linear_layers): linear_layers_(linear_layers){
}

double regression_net::inference(const Eigen::VectorXf& input){
    
    Eigen::VectorXf output = input;
    for(auto linear_layer_it  = linear_layers_.begin(); linear_layer_it != linear_layers_.end(); ){
        
        output = linear(output, linear_layer_it->first, linear_layer_it->second);
        
        if((++linear_layer_it) == linear_layers_.end()){
            return sigmoid(output(0));
        }
        else{
            output = relu(output);
        }
    }
    
    return -1.0;
}
