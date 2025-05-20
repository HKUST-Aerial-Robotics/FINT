/*******************************************************
Copyright (C) 2025, Aerial Robotics Group, Hong Kong University of Science and Technology
This file is part of FINT.
Licensed under the GNU General Public License v3.0;
you may not use this file except in compliance with the License.
*******************************************************/

#include "hough/hough_rectangle.hpp"
#include <math.h>
#include <stdlib.h>
#include <Eigen/Dense>
#include <array>
#include <iostream>
#include <tuple>
#include "array"

using namespace std;
using namespace Eigen;


HoughRectangle::HoughRectangle() : m_img(), m_thetaBins(), m_thetaMin(), m_thetaMax(), m_rhoBins(), m_theta_vec(), m_r_min(), m_r_max(), m_enhance_edge_length(){};


HoughRectangle::HoughRectangle(int thetaBins, int rhoBins, double thetaMin, double thetaMax, int r_min, int r_max, int enhance_edge_length,  double pixel_res, double max_inlier_dist) {
    m_thetaBins = thetaBins;
    m_thetaMin = thetaMin;
    m_thetaMax = thetaMax;
    m_rhoBins = rhoBins;
    
    m_pixelRes = pixel_res;
    
    m_squared_max_inlier_pixel_dist = pow(max_inlier_dist / pixel_res, 2);
    
    m_r_min = r_min;
    m_r_max = r_max;
    m_enhance_edge_length = enhance_edge_length;
    
    m_theta_vec = VectorXd::LinSpaced(thetaBins, thetaMin, thetaMax - (thetaMax - thetaMin)/thetaBins);
    m_rho_vec = VectorXd::LinSpaced(rhoBins, -r_max, r_max - (2.0 * r_max)/rhoBins);
    
    m_rhoRes = (m_rho_vec(rhoBins-1)-m_rho_vec(0))/(rhoBins-1);
    m_thetaRes = (m_theta_vec(thetaBins-1)-m_theta_vec(0))/(thetaBins-1);
    
    acc.setZero(rhoBins, thetaBins);
    enhanced_acc.setZero(rhoBins, thetaBins);
    
    pts.reserve(100 * 100);
    pts.clear();
    
}


HoughRectangle::HoughRectangle(cv::Mat& img, int thetaBins, int rhoBins, double thetaMin, double thetaMax, int r_min, int r_max, int enhance_edge_length,  double pixel_res, double max_inlier_dist) {
    
    if(img.channels() == 3)
    cvtColor(img, m_img, COLOR_BGR2GRAY);
    else
    m_img = img;
    
    m_img_center << img.rows / 2,  img.cols / 2;
    
    m_thetaBins = thetaBins;
    m_thetaMin = thetaMin;
    m_thetaMax = thetaMax;
    m_rhoBins = rhoBins;
    
    m_pixelRes = pixel_res;
    
    m_squared_max_inlier_pixel_dist = pow(max_inlier_dist / pixel_res, 2);
    
    m_r_min = r_min;
    m_r_max = r_max;
    m_enhance_edge_length = enhance_edge_length;
    
    m_theta_vec = VectorXd::LinSpaced(thetaBins, thetaMin, thetaMax - (thetaMax - thetaMin)/thetaBins);
    
    m_rho_vec = VectorXd::LinSpaced(rhoBins, -r_max, r_max - (2.0 * r_max)/rhoBins);
    m_rhoRes = (m_rho_vec(rhoBins-1)-m_rho_vec(0))/(rhoBins-1);
    m_thetaRes = (m_theta_vec(thetaBins-1)-m_theta_vec(0))/(thetaBins-1);
    
    acc.setZero(rhoBins, thetaBins);
    enhanced_acc.setZero(rhoBins, thetaBins);
    
    pts.reserve(img.rows * img.cols);
    pts.clear();
}

void HoughRectangle::set_image(const cv::Mat& img){
    if(img.channels() == 3)
    cvtColor(img, m_img, COLOR_BGR2GRAY);
    else
    m_img = img;
    
    m_img_center << img.rows / 2,  img.cols / 2;
    
    acc.setZero();
    enhanced_acc.setZero();
    
    pts.clear();
}


void HoughRectangle::ring(const int& r_min, const int& r_max) {
    
    if(m_img.empty()){
        cout<<"image not set!"<<endl;
        return;
    }
    
    double center_x, center_y;
    
    center_x = m_img.rows / 2;
    center_y = m_img.cols / 2;
    
    
    for (int i = 0; i < m_img.rows; ++i) {
        for (int j = 0; j < m_img.cols; ++j) {
            double dist = sqrt(pow(i - center_x, 2) + pow(j - center_y, 2));
            if (dist < r_min or dist > r_max) {
                m_img.at<uchar>(i, j) = 0;
            }
        }
    }
    
}


void HoughRectangle::windowed_hough(const int& r_min,
    const int& r_max) {
        
        if(m_img.empty()){
            cout<<"image not set!"<<endl;
            return;
        }
        
        hough_transform();
        
    }
    
    
    
    void HoughRectangle::hough_transform() {
        
        VectorXi vecX = VectorXi::LinSpaced(m_img.rows, 0, m_img.rows - 1);
        VectorXi vecY = VectorXi::LinSpaced(m_img.cols, 0, m_img.cols - 1);
        int mid_X = m_img.rows / 2;
        int mid_Y = m_img.cols / 2;
        
        vecX = vecX.array() - mid_X;
        vecY = vecY.array() - mid_Y;
        
        VectorXd cosT, sinT;
        cosT.resize(m_thetaBins+1);
        sinT.resize(m_thetaBins+1);
        cosT.topRows(m_thetaBins)= cos(m_theta_vec.array() * M_PI / 180.0);
        cosT(m_thetaBins) = cos(m_thetaMax * M_PI / 180.0);
        sinT.topRows(m_thetaBins) = sin(m_theta_vec.array() * M_PI / 180.0);
        sinT(m_thetaBins) = sin(m_thetaMax * M_PI / 180.0);
        
        
        long idx_rho;
        VectorXd rho_vec_tmp;
        for (int i = 0; i < m_img.rows; ++i) {
            for (int j = 0; j < m_img.cols; ++j) {
                uchar pixel_intensity = m_img.at<uchar>(i, j);
                if ( pixel_intensity > uchar(50)) {
                    
                    pts.emplace_back(i,j);
                    pixel_intensity = uchar(255);
                    
                    // find the local extreme of the sinusoidal curve
                    double local_extreme_theta = atan(static_cast<double>(vecY[j])/static_cast<double>(vecX[i]));
                    double local_extreme_rho = static_cast<double>(vecX[i]) * cos(local_extreme_theta) + static_cast<double>(vecY[j]) * sin(local_extreme_theta);
                    
                    
                    // generate sinusoidal curve
                    rho_vec_tmp = vecX[i] * cosT + vecY[j] * sinT;
                    
                    // Find corresponding position and fill accumulator
                    for (int k = 0; k < m_thetaBins; ++k) {
                        
                        double lower_theta = m_theta_vec(k) * M_PI / 180.0;
                        double lower_rho = rho_vec_tmp(k);
                        
                        double upper_theta = k+1 == m_thetaBins ? m_thetaMax * M_PI / 180.0 : m_theta_vec(k+1) * M_PI / 180.0;
                        double upper_rho = rho_vec_tmp(k+1);
                        
                        
                        double rho_min = min(lower_rho, upper_rho);
                        double rho_max = max(lower_rho, upper_rho);
                        
                        // local extreme theta reached
                        if(lower_theta < local_extreme_theta &&  upper_theta > local_extreme_theta){
                            
                            rho_min = min(local_extreme_rho, rho_min);
                            rho_max = max(local_extreme_rho, rho_max);
                            
                        }
                        
                        
                        int idx_rho_min = max(static_cast<int>(floor((rho_min - m_rho_vec(0))/ m_rhoRes)), 0);
                        
                        int idx_rho_max = min(static_cast<int>(floor((rho_max - m_rho_vec(0))/ m_rhoRes)), m_rhoBins - 1);
                        
                        if(idx_rho_min > idx_rho_max){
                            continue;
                        }
                        
                        
                        // Fill accumulator
                        acc.block(idx_rho_min,k, idx_rho_max - idx_rho_min + 1,1)+=VectorXd::Ones(idx_rho_max - idx_rho_min + 1) * (static_cast<double>(pixel_intensity) * m_pixelRes);
                        
                    }
                }
            }
        }
    }
    
    
    void HoughRectangle::enhance_hough(const int& half_w, const int& half_h) {
        
        for (int i = half_h; i < acc.rows() - half_h; ++i) {
            for (int j = 0; j < acc.cols(); ++j) {
                
                double enhance_sum = 0.0;
                long block_size = 0;
                
                int min_rho = i - half_h;
                int max_rho = i + half_h;
                
                int min_theta = j - half_w;
                int max_theta = j + half_w;
                
                int rho_size = max_rho - min_rho + 1;
                int theta_size = max_theta - min_theta + 1;
                
                
                if(min_theta >= 0L && max_theta < acc.cols()){
                    
                    enhance_sum = acc.block(min_rho, min_theta, rho_size, theta_size).sum();
                    block_size = rho_size * theta_size;
                    
                }else if(min_theta < 0L && max_theta < acc.cols()){
                    
                    //                min_theta overshoot
                    int overshoot_min_theta = min_theta;
                    while(overshoot_min_theta < 0){
                        overshoot_min_theta += acc.cols();
                    }
                    
                    int overshoot_min_rho = static_cast<int>(round((-m_rho_vec(max_rho)-m_rho_vec(0)) / m_rhoRes - 1.0));
                    int overshoot_max_rho = overshoot_min_rho + (max_rho - min_rho);
                    
                    enhance_sum = acc.block(min_rho, 0, rho_size, max_theta+1).sum() + acc.block(overshoot_min_rho, overshoot_min_theta, rho_size, acc.cols() - overshoot_min_theta).sum();
                    
                    block_size = rho_size * (max_theta + acc.cols() - overshoot_min_theta + 1);
                    
                }else if(min_theta >= 0 && max_theta >= acc.cols()){
                    
                    //                max_theta overshoot
                    int overshoot_max_theta = max_theta;
                    while(overshoot_max_theta >= acc.cols()){
                        overshoot_max_theta -= acc.cols();
                    }
                    
                    int overshoot_min_rho = static_cast<int>(round((-m_rho_vec(max_rho)-m_rho_vec(0)) / m_rhoRes - 1.0));
                    int overshoot_max_rho = overshoot_min_rho + (max_rho - min_rho);
                    
                    enhance_sum = acc.block(min_rho, min_theta, rho_size, acc.cols() - min_theta).sum() + acc.block(overshoot_min_rho, 0, rho_size, overshoot_max_theta+1).sum();
                    
                    block_size = rho_size * (acc.cols() - min_theta + overshoot_max_theta + 1);
                }
                
                
                if (enhance_sum == 0) {
                    enhanced_acc(i, j) = 0.0;
                } else {
                    enhanced_acc(i, j) = pow(acc(i, j), 2) * block_size / enhance_sum;
                }
            }
        }
        
    }
    
    
    std::tuple<std::vector<double>, std::vector<double>> HoughRectangle::index_rho_theta(
        const std::vector<std::array<int, 2>>& indexes) {
            
            std::vector<double> rho_max(indexes.size());
            std::vector<double> theta_max(indexes.size());
            
            for (int i = 0; i < indexes.size(); ++i) {
                rho_max[i] = m_rho_vec[indexes[i][0]];
                theta_max[i] = m_theta_vec(indexes[i][1]);
            }
            
            return std::make_tuple(rho_max, theta_max);
        }
        
        std::vector<std::array<int, 2>> HoughRectangle::find_local_maximum(const Eigen::MatrixXd& hough, const double& threshold) {
            std::vector<std::array<int, 2>> idxs;
            
            for (int i = 0; i < hough.rows(); ++i) {
                for (int j = 0; j < hough.cols(); ++j) {
                    if (hough(i, j) > threshold) {
                        std::array<int, 2> x = {i, j};
                        idxs.emplace_back(x);
                    }
                }
            }
            
            std::sort(idxs.begin(), idxs.end(), hough_cmp_gt(hough));
            
            return idxs;
        }
        
        
        bool HoughRectangle::detect_rectangle_from_hough_lines(const std::vector<std::array<int, 2>>& line_indexes, const double &T_rho, const double &T_t, const double &T_L, const double &T_alpha, const double &E_weight, rectangleShape &result){
            // Match peaks into pairs
            std::vector<std::array<double, 6>> pairs;  // 1st: rho, 2nd: theta
            std::array<double, 6> pair;
            for (int i = 0; i < line_indexes.size(); ++i) {
                double rho_i = m_rho_vec(line_indexes[i][0]) + 0.5 * m_rhoRes;
                double theta_i = m_theta_vec(line_indexes[i][1]) + 0.5 * m_thetaRes;
                double acc_i = acc(line_indexes[i][0], line_indexes[i][1]);
                
                for (int j = i+1; j < line_indexes.size(); ++j) {
                    
                    double rho_j = m_rho_vec(line_indexes[j][0]) + 0.5 * m_rhoRes;
                    double theta_j = m_theta_vec(line_indexes[j][1]) + 0.5 * m_thetaRes;
                    
                    double d_theta = abs(theta_i - theta_j);
                    double mean_theta = 0.5 * (theta_i + theta_j);
                    double d_theta_pi = theta_i > theta_j ? abs(theta_i - 180 - theta_j) : abs(theta_j - 180 - theta_i);
                    
                    double acc_j = acc(line_indexes[j][0], line_indexes[j][1]);
                    
                    double sum_rho = rho_i + rho_j;
                    double d_rho = abs(rho_i - rho_j);
                    
                    double sum_acc = acc_i + acc_j;
                    double d_acc = abs(acc_i - acc_j);
                    
                    if(d_theta > d_theta_pi){
                        d_theta = d_theta_pi;
                        mean_theta = 0.5 * (theta_i + theta_j - 180);
                        mean_theta = mean_theta < m_thetaMin ? mean_theta + 180.0 : mean_theta;
                        
                        double sign_i = rho_i < 0.0 ? -1 : 1;
                        double sign_j = rho_j < 0.0 ? -1 : 1;
                        
                        sum_rho = mean_theta - theta_i < mean_theta - theta_j ? sign_i * abs(rho_i - rho_j) : sign_j * abs(rho_i - rho_j);
                        
                        d_rho = abs(rho_i + rho_j);
                        
                    }
                    
                    // Parralelism
                    if (d_theta > T_t) continue;
                    
                    // Approximately same length
                    if(d_acc > 0.5 * T_L * sum_acc) continue;
                    
                    // Construct extended peak
                    pair[0] = 0.5 * d_rho;
                    
                    if(pair[0] < m_r_min || pair[0] > m_r_max) continue;
                    
                    pair[1] = mean_theta;
                    pair[2] = sum_rho;      // error measure on rho
                    pair[3] = d_theta;  // error measure on theta
                    pair[4] = sum_acc;
                    pair[5] = d_acc;
                    
                    pairs.emplace_back(pair);
                }
            }
            
            bool detect_rectangle = false;
            
            double min_squared_criteria = std::numeric_limits<double>::max();
            for (int i = 0; i < pairs.size(); i++) {
                for (int j = i+1; j < pairs.size(); j++) {
                    
                    // Orthogonality
                    double delta_alpha = abs(abs(pairs[i][1] - pairs[j][1]) - 90);
                    if (delta_alpha > T_alpha) continue;
                    
                    
                    
                    double squared_angle_criteria = (pow(pairs[i][3],2) + pow(pairs[j][3],2) + pow(delta_alpha,2));
                    
                    double squared_acc_criteria = (1.0 / pow(pairs[i][4],2) + 1.0 / pow(pairs[j][4],2));
                    
                    double squared_criteria = squared_angle_criteria + E_weight * squared_acc_criteria;
                    
                    if(squared_criteria < min_squared_criteria){
                        min_squared_criteria = squared_criteria;
                        double angle_i = pairs[i][1] * M_PI / 180.0;
                        double angle_j = pairs[j][1] * M_PI / 180.0;
                        
                        double sum_rho_i = pairs[i][2];
                        double sum_rho_j = pairs[j][2];
                        
                        result.set_shape_half(angle_i, pairs[i][0], pairs[j][0]);
                        result.set_center(Vector2d(0.5*(pairs[i][2]*cos(angle_i)+pairs[j][2]*cos(angle_j)) + m_img_center.x(), 0.5*(pairs[i][2]*sin(angle_i)+pairs[j][2]*sin(angle_j))+ m_img_center.y()
                    ) );
                    detect_rectangle = true;
                }
                
            }
        }
        
        return detect_rectangle;
        
    }
    
    
    int HoughRectangle::count_outliers(rectangleShape &result){
        int outlier_cnt = 0;
        
        std::array<Eigen::Vector2d, 4> corners = result.output_corners();
        
        std::array<Eigen::Vector2d, 4> edge_dirs;
        for(int i = 0; i < 4; i++){
            edge_dirs[i] = (corners[(i+1)%4] - corners[i]).normalized();
        }
        
        
        
        for(Vector2d& pt : pts){
            bool is_outlier_pt = true;
            
            for(int i = 0; i < 4; i++){
                if((pt - corners[i]).squaredNorm() <= m_squared_max_inlier_pixel_dist){
                    is_outlier_pt = false;
                    break;
                }
                
                Vector2d pt_corner_1_diff = pt - corners[i];
                Vector2d pt_corner_2_diff = pt - corners[(i+1)%4];
                
                double pt_corner_1_diff_proj_edge = pt_corner_1_diff.dot(edge_dirs[i]);
                double pt_corner_2_diff_proj_edge = pt_corner_2_diff.dot(edge_dirs[i]);
                
                if(pt_corner_1_diff_proj_edge >= 0.0 && pt_corner_2_diff_proj_edge <= 0.0){
                    if((pt_corner_1_diff - pt_corner_1_diff_proj_edge * edge_dirs[i]).squaredNorm() <= m_squared_max_inlier_pixel_dist){
                        is_outlier_pt = false;
                        break;
                    }
                }
                
                
            }
            
            if(is_outlier_pt){
                outlier_cnt++;
            }
            
        }
        
        return outlier_cnt;
    }
    
    
    bool HoughRectangle::detect_rectangle(const double threshold_line, const double T_rho, const double T_t, const double T_L, const double T_alpha, const double E_weight, rectangleShape &result, int& outlier_cnt){
        
        if(m_img.empty()){
            cout<<"image not set!"<<endl;
            return false;
        }
        
        windowed_hough(m_r_min, m_r_max);
        
        enhance_hough(m_enhance_edge_length, m_enhance_edge_length);
        
        
        std::vector<std::array<int, 2>> local_max_idx = find_local_maximum(enhanced_acc, threshold_line);
        if(local_max_idx.size() > 500){
            // printf("Too many lines!!!!!!!!!!!!!!!!!!\n");
            return false;
        }
        
        
        if(!detect_rectangle_from_hough_lines(local_max_idx, T_rho, T_t, T_L, T_alpha, E_weight, result)){
            return false;
        }
        else{
            outlier_cnt = count_outliers(result);
        }
        
        return true;
    }
