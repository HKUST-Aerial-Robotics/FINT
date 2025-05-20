/*******************************************************
Copyright (C) 2025, Aerial Robotics Group, Hong Kong University of Science and Technology
This file is part of FINT.
Licensed under the GNU General Public License v3.0;
you may not use this file except in compliance with the License.
*******************************************************/
//
//  cnn.hpp
//  cnn
//
//  Created by Luqi on 27/1/2023.
//

#ifndef cnn_hpp
#define cnn_hpp

#include <iostream>
#include <Eigen/Eigen>
#include <vector>
#include <memory>

using namespace std;

enum layer_type{
    UNKNOWN = -1,
    RELU = 0,
    SIGMOID,
    SOFTMAX,
    FLATTEN,
    LINEAR,
    CONV2D,
    MAXPOOL2D,
    DROPOUT
};


class net_layer{
public:
    net_layer(int type) : type_(type){}
    
    virtual void inference(std::vector<Eigen::MatrixXf>& input, std::vector<Eigen::MatrixXf>& output)=0;
protected:
    int type_;
    
};

class relu_layer: public net_layer{
public:
    relu_layer():net_layer(RELU){}
    void inference(std::vector<Eigen::MatrixXf>& input, std::vector<Eigen::MatrixXf>& output);
protected:
    
};

class sigmoid_layer: public net_layer{
public:
    sigmoid_layer():net_layer(SIGMOID){}
    void inference(std::vector<Eigen::MatrixXf>& input, std::vector<Eigen::MatrixXf>& output);
protected:
    
};

class softmax_layer: public net_layer{
public:
    softmax_layer():net_layer(SOFTMAX){}
    void inference(std::vector<Eigen::MatrixXf>& input, std::vector<Eigen::MatrixXf>& output);
protected:
    
};

class flatten_layer: public net_layer{
public:
    flatten_layer():net_layer(FLATTEN){}
    void inference(std::vector<Eigen::MatrixXf>& input, std::vector<Eigen::MatrixXf>& output);
protected:
    
};

class linear_layer: public net_layer{
public:
    linear_layer(Eigen::MatrixXf& A, Eigen::VectorXf& b): net_layer(LINEAR), A_t_(A.transpose()), b_(b){}
    void inference(std::vector<Eigen::MatrixXf>& input, std::vector<Eigen::MatrixXf>& output);
protected:
    Eigen::MatrixXf A_t_;
    Eigen::RowVectorXf b_;
};

class conv2d_layer: public net_layer{
public:
    conv2d_layer(Eigen::Matrix<Eigen::MatrixXf, Eigen::Dynamic, Eigen::Dynamic>& kernels, Eigen::VectorXf& b, Eigen::Vector2i& strides): net_layer(CONV2D), kernels_(kernels), b_(b), strides_(strides){}
    void inference(std::vector<Eigen::MatrixXf>& input, std::vector<Eigen::MatrixXf>& output);
protected:
    Eigen::Matrix<Eigen::MatrixXf, Eigen::Dynamic, Eigen::Dynamic> kernels_;
    Eigen::VectorXf b_;
    Eigen::Vector2i strides_;
};

class maxpool2d_layer: public net_layer{
public:
    maxpool2d_layer(Eigen::Vector2i& kernel_size, Eigen::Vector2i& strides, bool padding = false):net_layer(MAXPOOL2D), kernel_size_(kernel_size), strides_(strides), padding_(padding){}
    void inference(std::vector<Eigen::MatrixXf>& input, std::vector<Eigen::MatrixXf>& output);
protected:
    void inference(Eigen::MatrixXf& input, Eigen::MatrixXf& output);
    
    Eigen::Vector2i kernel_size_;
    Eigen::Vector2i strides_;
    bool padding_;
};

class nn{
public:
    nn(): input_rows_(0), input_cols_(0), input_channels_(1){
    }

    nn(vector<shared_ptr<net_layer>> layer_ptrs, int input_rows, int input_cols, int input_channels = 1): 
        layer_ptrs_(layer_ptrs), input_rows_(input_rows), input_cols_(input_cols), input_channels_(input_channels){
    }

    Eigen::Vector3i get_input_dim(){
        return Eigen::Vector3i(input_rows_, input_cols_, input_channels_);
    }
    
    void inference_net(std::vector<Eigen::MatrixXf>& input, std::vector<Eigen::MatrixXf>& output);
private:
    vector<shared_ptr<net_layer>> layer_ptrs_;

    int input_rows_;
    int input_cols_;
    int input_channels_;
};

#endif /* regression_net_hpp */
