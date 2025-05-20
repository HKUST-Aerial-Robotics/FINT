/*******************************************************
Copyright (C) 2025, Aerial Robotics Group, Hong Kong University of Science and Technology
This file is part of FINT.
Licensed under the GNU General Public License v3.0;
you may not use this file except in compliance with the License.
*******************************************************/
//
//  hough_circle.cpp
//  hough_circle
//
//  Created by Luqi on 6/2/2023.
//

#include "hough/hough_circle.hpp"

using namespace std;
using namespace Eigen;

HoughCircle::HoughCircle(){
    
}

HoughCircle::HoughCircle(double max_center_dev, double rMin, double rMax, double pixel_res, double max_inlier_dist){

    m_pixelRes = pixel_res;

    m_max_inlier_pixel_dist = max_inlier_dist / pixel_res;

    m_rMin = static_cast<int>(floor(rMin/pixel_res));
    m_rMax = static_cast<int>(ceil(rMax/pixel_res));

    m_r_vec = VectorXi::LinSpaced(m_rMax - m_rMin + 1, m_rMin, m_rMax);
    acc.resize(m_r_vec.rows());

    m_maxCenterDev = static_cast<int>(ceil(max_center_dev/pixel_res));

    int acc_size = 2 * m_maxCenterDev + 1;

    for(int i = 0; i < acc.size(); i++){
//        acc[i] = MatrixXd::Zero(img.rows, img.cols);
        acc[i] = MatrixXd::Zero(acc_size, acc_size);
    }

    have_acc.assign(m_rMax - m_rMin + 1, false);

    pts.reserve(100 * 100);
}

HoughCircle::HoughCircle(double max_center_dev, int rMin_pixel, int rMax_pixel, double pixel_res, double max_inlier_dist){

    m_pixelRes = pixel_res;

    m_max_inlier_pixel_dist = max_inlier_dist / pixel_res;

    m_rMin = rMin_pixel;
    m_rMax = rMax_pixel;

    m_r_vec = VectorXi::LinSpaced(m_rMax - m_rMin + 1, m_rMin, m_rMax);
    acc.resize(m_r_vec.rows());

    m_maxCenterDev = static_cast<int>(ceil(max_center_dev/pixel_res));

    int acc_size = 2 * m_maxCenterDev + 1;

    for(int i = 0; i < acc.size(); i++){
//        acc[i] = MatrixXd::Zero(img.rows, img.cols);
        acc[i] = MatrixXd::Zero(acc_size, acc_size);
    }

    have_acc.assign(m_rMax - m_rMin + 1, false);

    pts.reserve(100 * 100);
}


void HoughCircle::set_image(const cv::Mat &img){
    m_img = img;
    int max_center_dev = min(min((img.rows) / 2, (img.cols)/2), m_maxCenterDev);
    m_centerMin << img.rows / 2 - max_center_dev, img.cols/2 - max_center_dev;
    m_centerMax << img.rows / 2 + max_center_dev, img.cols/2 + max_center_dev;

    if(max_center_dev == m_maxCenterDev){
        for(auto& acc_mat : acc){
            acc_mat.setZero();
        }
    }
    else{

        m_maxCenterDev = max_center_dev;
        int acc_size = 2 * m_maxCenterDev + 1;
        for(auto& acc_mat : acc){
            acc_mat = MatrixXd::Zero(acc_size, acc_size);
        }
    }

    pts.clear();
}

void HoughCircle::accum_circle(Eigen::MatrixXd& acc_r, const Eigen::Vector2i &position, unsigned int radius){

    int f = 1 - radius;
    int ddF_x = 1;
    int ddF_y = -2 * radius;
    int x = 0;
    int y = radius;

    accum_pixel(acc_r, Vector2i(position.x(), position.y() + radius));
    accum_pixel(acc_r, Vector2i(position.x(), position.y() - radius));
    accum_pixel(acc_r, Vector2i(position.x() + radius, position.y()));
    accum_pixel(acc_r, Vector2i(position.x() - radius, position.y()));

    while(x < y)
    {
    if(f >= 0)
    {
      y--;
      ddF_y += 2;
      f += ddF_y;
    }

    x++;
    ddF_x += 2;
    f += ddF_x;

    accum_pixel(acc_r, Vector2i(position.x() + x, position.y() + y));
    accum_pixel(acc_r, Vector2i(position.x() - x, position.y() + y));
    accum_pixel(acc_r, Vector2i(position.x() + x, position.y() - y));
    accum_pixel(acc_r, Vector2i(position.x() - x, position.y() - y));
    accum_pixel(acc_r, Vector2i(position.x() + y, position.y() + x));
    accum_pixel(acc_r, Vector2i(position.x() - y, position.y() + x));
    accum_pixel(acc_r, Vector2i(position.x() + y, position.y() - x));
    accum_pixel(acc_r, Vector2i(position.x() - y, position.y() - x));
    }
}

void HoughCircle::accum_pixel(Eigen::MatrixXd& acc_r, const Vector2i& pos_idx)
{
  /* bounds checking */
  if(pos_idx.x() < 0 || pos_idx.x() >= acc_r.rows() ||
     pos_idx.y() < 0 || pos_idx.y() >= acc_r.cols())
  {
    return;
  }
  
  acc_r(pos_idx.x(), pos_idx.y()) += 1.0;
}

void HoughCircle::hough_transform(){
    
    for (int i = 0; i < m_img.rows; ++i) {
        for (int j = 0; j < m_img.cols; ++j) {
            uchar pixel_intensity = m_img.at<uchar>(i, j);
            if ( pixel_intensity > uchar(50)) {
                pts.emplace_back(Vector3i(i,j,pixel_intensity));
                
                for(int x = m_centerMin.x(); x <= m_centerMax.x(); ++x ){
                    for(int y = m_centerMin.y(); y <= m_centerMax.y(); ++y){
                        double r = sqrt(pow(x-i,2)+pow(y-j,2));
                        int r_idx = static_cast<int>(round(r - m_rMin));
                        if(r_idx >= 0 && r_idx < m_r_vec.size()){
                            acc[r_idx](x-m_centerMin.x(), y-m_centerMin.y()) += (pixel_intensity * m_pixelRes / r);
                            have_acc[r_idx] = true;
                        }
                        
                    }
                }
                
                
                
            }
        }
        
    }
    
}

bool HoughCircle::detect_circle(circleShape &result, double threshold, int& outlier_cnt){
    
    hough_transform();

    Vector3i max_acc_idx(0,0,0);
    double max_acc = 0.0, search_acc = 0.0;
    
    for(int acc_idx  = 0; acc_idx < acc.size(); acc_idx++){
        if(!have_acc[acc_idx])
            continue;
        
        for(int row_idx = 0; row_idx < acc[acc_idx].rows(); row_idx++){
            for(int col_idx = 0; col_idx < acc[acc_idx].cols(); col_idx++){
                search_acc = acc[acc_idx](row_idx,col_idx);
                if(search_acc > max_acc){
                    max_acc = search_acc;
                    max_acc_idx << acc_idx, row_idx, col_idx;
                }

            }
        }
        
    }
    
    Vector2d center(max_acc_idx(1) + m_centerMin.x(), max_acc_idx(2) + m_centerMin.y());
    double radius = m_r_vec(max_acc_idx(0));
    result.set_center(center);
    result.set_r(radius);

    outlier_cnt = 0;
    for(auto& img_pt_intensity : pts){
        double dist = (img_pt_intensity.head(2).cast<double>() - center).norm();
        if(abs(dist - radius) > m_max_inlier_pixel_dist){
            outlier_cnt++;
        }
    }

    return (max_acc > threshold);
}
