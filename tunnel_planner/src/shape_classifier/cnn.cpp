/*******************************************************
Copyright (C) 2025, Aerial Robotics Group, Hong Kong University of Science and Technology
This file is part of FINT.
Licensed under the GNU General Public License v3.0;
you may not use this file except in compliance with the License.
*******************************************************/
//
//  cnn.cpp
//  cnn
//
//  Created by Luqi on 27/1/2023.
//

#include "shape_classifier/cnn.hpp"


void relu_layer::inference(std::vector<Eigen::MatrixXf>& input, std::vector<Eigen::MatrixXf>& output){
    
    output.resize(input.size());
    for(int channel = 0; channel < input.size(); channel++){
//        Eigen::MatrixXf& input_mat = input[channel];
//        Eigen::MatrixXf& output_mat = output[channel];
        
        output[channel] = input[channel].cwiseMax(0.0f);
        
//        output_mat.resize(input_mat.rows(), input_mat.cols());
//        for(int i = 0; i < output_mat.rows(); i++){
//            for(int j = 0; j < output_mat.cols(); j++){
//                output_mat(i,j) = input_mat(i,j) > 0.0 ? input_mat(i,j) : 0.0;
//            }
//        }
    }
    
}


void sigmoid_layer::inference(std::vector<Eigen::MatrixXf>& input, std::vector<Eigen::MatrixXf>& output){
    output.resize(input.size());
    for(int channel = 0; channel < input.size(); channel++){
        Eigen::MatrixXf& input_mat = input[channel];
        Eigen::MatrixXf& output_mat = output[channel];
        output_mat.resize(input_mat.rows(), input_mat.cols());
        for(int i = 0; i < output_mat.rows(); i++){
            for(int j = 0; j < output_mat.cols(); j++){
                output_mat(i,j) = 1.0f / (1.0f + exp(-input_mat(i,j)));
            }
        }
    }
}


void softmax_layer::inference(std::vector<Eigen::MatrixXf>& input, std::vector<Eigen::MatrixXf>& output){
    output.resize(input.size());

    for(int channel = 0; channel < input.size(); channel++){
        Eigen::MatrixXf& input_mat = input[channel];
        Eigen::MatrixXf& output_mat = output[channel];
        output_mat.resize(input_mat.rows(), input_mat.cols());
        
        for(int i = 0; i < output_mat.rows(); i++){
            float sum_exp = 0;
            for(int j = 0; j < output_mat.cols(); j++){
                float tmp_exp = exp(input_mat(i,j));
                output_mat(i,j) = tmp_exp;
                sum_exp += tmp_exp;
            }
            output_mat.row(i) /= sum_exp;
        }
    }


}


void flatten_layer::inference(std::vector<Eigen::MatrixXf>& input, std::vector<Eigen::MatrixXf>& output){
    int total_size = 0;
    for(auto& input_mat: input){
        total_size += input_mat.size();
    }

    output.assign(1, Eigen::MatrixXf(1, total_size));

    int output_idx = 0;
    for(int channel = 0; channel < input.size(); channel++){
        Eigen::MatrixXf& input_mat = input[channel];
        for(int i = 0; i < input_mat.rows(); i++){
            for(int j = 0; j < input_mat.cols(); j++){
                output[0](0, output_idx++) = input_mat(i,j);
            }
        }
    }
    
//    std::cout<<"after flatten:\n"<<output[0]<<std::endl;

}

void linear_layer::inference(std::vector<Eigen::MatrixXf>& input, std::vector<Eigen::MatrixXf>& output){
    output.resize(input.size());

    for(int channel_idx = 0; channel_idx < input.size(); channel_idx++){
        output[channel_idx].resize(input[channel_idx].rows(), b_.cols());
        for(int row_idx = 0; row_idx < input[channel_idx].rows(); row_idx++){
            output[channel_idx].row(row_idx) = input[channel_idx].row(row_idx) * A_t_ + b_;
        }
    }
    
//    cout<<"after linear"<<endl;
//    for(int i = 0; i < output.size(); i++){
//        cout<<"out channel "<<i<<" size: "<<output[i].rows()<<"\t"<<output[i].cols()<<endl;
//        cout<<output[i]<<endl<<endl;
//    }
//    cout<<"linear finish"<<endl;
}

void conv2d_layer::inference(std::vector<Eigen::MatrixXf>& input, std::vector<Eigen::MatrixXf>& output){
    output.resize(kernels_.rows());

    long output_rows = (input[0].rows() - kernels_(0,0).rows()) / strides_.x() + 1L;
    long output_cols = (input[0].cols() - kernels_(0,0).cols()) / strides_.y() + 1L;

    output.assign(kernels_.rows(), Eigen::MatrixXf::Zero(output_rows, output_cols));

    for(int output_mat_idx = 0; output_mat_idx < kernels_.rows(); output_mat_idx++){
        Eigen::MatrixXf& output_mat = output[output_mat_idx];
        output_mat.setConstant(b_(output_mat_idx));

        for(int input_mat_idx = 0; input_mat_idx < input.size(); input_mat_idx++){

            Eigen::MatrixXf& kernel_mat = kernels_(output_mat_idx, input_mat_idx);
            Eigen::MatrixXf& input_mat = input[input_mat_idx];
            
            for(int row_idx = 0; row_idx < output_rows; row_idx++){
                for(int col_idx = 0; col_idx < output_cols; col_idx++){

                    int input_row_idx = row_idx * strides_.x();
                    int input_col_idx = col_idx * strides_.y();
                    
                    output_mat(row_idx, col_idx) += kernel_mat.cwiseProduct(input_mat.block(input_row_idx, input_col_idx, kernel_mat.rows(), kernel_mat.cols())).sum();

//                    float& output_element = output_mat(row_idx, col_idx);
//                    for(int kernel_row_idx = 0; kernel_row_idx < kernel_mat.rows(); kernel_row_idx++){
//                        for(int kernel_col_idx = 0; kernel_col_idx < kernel_mat.cols(); kernel_col_idx++){
//                            output_element += kernel_mat(kernel_row_idx, kernel_col_idx) * input_mat(input_row_idx + kernel_row_idx, input_col_idx + kernel_col_idx);
//                        }
//                    }

                }
            }
        }
    }
    
//    std::cout<<"after conv2d size:\n"<<output.size()<<std::endl;
//    std::cout<<"after conv2d size:\n"<<output[0].rows()<<"\t"<<output[0].cols()<<std::endl;
    
//    cout<<"out check: "<<output[14](7,14)<<endl;
//    for(int i = 0; i < output.size(); i++){
//        cout<<"out channel "<<i<<endl;
//        cout<<output[i]<<endl<<endl;
//    }
//
//    cout<<"end conv2d"<<endl;
}


void maxpool2d_layer::inference(std::vector<Eigen::MatrixXf>& input, std::vector<Eigen::MatrixXf>& output){
    output.resize(input.size());
    for(size_t idx = 0; idx < input.size(); idx++){
        inference(input[idx], output[idx]);
    }

}

void maxpool2d_layer::inference(Eigen::MatrixXf& input, Eigen::MatrixXf& output){

    int input_rows = int(input.rows());
    int input_cols = int(input.cols());

    int remain_rows = (input_rows - kernel_size_.x()) % strides_.x();
    int remain_cols = (input_cols - kernel_size_.y()) % strides_.y();

    int output_rows_ori = (input_rows - kernel_size_.x()) / strides_.x() + 1;
    int output_cols_ori = (input_cols - kernel_size_.y()) / strides_.y() + 1;

    int output_rows = output_rows_ori;
    int output_cols = output_cols_ori;

    if(padding_){

        if(remain_rows > 0){
            output_rows++;
        }
        if(remain_cols > 0){
            output_cols++;
        }

    }

    output.resize(output_rows, output_cols);

    for(int i = 0; i < output_rows; i++){
        int diff_block_input_row_size = strides_.x() * i + kernel_size_.x() - input_rows;
        int block_row_size = diff_block_input_row_size > 0 ? kernel_size_.x() - diff_block_input_row_size : kernel_size_.x();

        for(int j = 0; j < output_cols; j++){
            int diff_block_input_col_size = strides_.y() * j + kernel_size_.y() - input_cols;
            int block_col_size = diff_block_input_col_size > 0 ? kernel_size_.y() - diff_block_input_col_size : kernel_size_.y();

            output(i,j) = input.block(strides_.x() * i, strides_.y() * j, block_row_size, block_col_size).maxCoeff();

        }

    }


}

void nn::inference_net(std::vector<Eigen::MatrixXf>& input, std::vector<Eigen::MatrixXf>& output){
    
    std::vector<Eigen::MatrixXf> last_output = input;
    for(auto& net_layer_ptr : layer_ptrs_){
        net_layer_ptr->inference(last_output, output);
        if(net_layer_ptr == layer_ptrs_.back()){
            break;
        }
        else{
            last_output = output;
        }
    }
    
}

