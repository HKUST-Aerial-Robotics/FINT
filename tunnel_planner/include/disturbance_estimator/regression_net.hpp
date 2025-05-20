/*******************************************************
Copyright (C) 2025, Aerial Robotics Group, Hong Kong University of Science and Technology
This file is part of FINT.
Licensed under the GNU General Public License v3.0;
you may not use this file except in compliance with the License.
*******************************************************/
//
//  regression_net.hpp
//  regression_net
//
//  Created by Luqi on 27/1/2023.
//

#ifndef regression_net_hpp
#define regression_net_hpp

//67.68
#include <iostream>
#include <Eigen/Eigen>
#include <vector>

using namespace std;

class regression_net{
public:
    static Eigen::VectorXf relu(Eigen::VectorXf& input);
    static Eigen::VectorXf linear(Eigen::VectorXf& input, Eigen::MatrixXf& A, Eigen::VectorXf& b);
    static double sigmoid(float input);
    
    regression_net(vector<pair<Eigen::MatrixXf, Eigen::VectorXf>>& linear_layers);
    
    // input: dimensions (h,w for rectangle, r for circle), pitch, speed
    double inference(const Eigen::VectorXf& input);
    
protected:
    vector<pair<Eigen::MatrixXf, Eigen::VectorXf>> linear_layers_;
};

class circle_regression_net : public regression_net{
public:
    
protected:
    
};

class rect_regression_net : public regression_net{
public:
    
protected:
    
};

#endif /* regression_net_hpp */
