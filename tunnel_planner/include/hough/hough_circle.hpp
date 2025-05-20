/*******************************************************
Copyright (C) 2025, Aerial Robotics Group, Hong Kong University of Science and Technology
This file is part of FINT.
Licensed under the GNU General Public License v3.0;
you may not use this file except in compliance with the License.
*******************************************************/
//
//  hough_circle.hpp
//  hough_Circle
//
//  Created by Luqi on 6/2/2023.
//

#ifndef hough_circle_hpp
#define hough_circle_hpp

#include <stdio.h>
#include <Eigen/Dense>
#include <array>
#include <tuple>
#include "opencv2/opencv.hpp"

using namespace cv;

class circleShape{
public:
    
    
    inline void set_r(const double &_r){
        r = _r;
    }
    
    inline void set_center(const Eigen::Vector2d &_center){
        center = _center;
    }
    
    
    inline Eigen::Vector2d get_center(){
        return center;
    }
    
    inline double get_r(){
        return r;
    }
    
private:
    
    Eigen::Vector2d center;
    double r;
    
};

class HoughCircle {
    
   protected:
    int m_rMin;
    int m_rMax;
    int m_maxCenterDev;
    
    Eigen::Vector2i m_centerMin;
    Eigen::Vector2i m_centerMax;

    double m_pixelRes;
    
    double m_max_inlier_pixel_dist;
    
    std::vector<Eigen::MatrixXd> acc;
    std::vector<bool> have_acc;
    std::vector<Eigen::Vector3i> pts;
    
    Eigen::VectorXi m_r_vec;
    
    Mat m_img;


   public:

    HoughCircle();
    HoughCircle(double max_center_dev = 0.15, double rMin = 0.0, double rMax = 1.0, double pixel_res = 0.04, double max_inlier_dist = 0.05);
    HoughCircle(double max_center_dev = 0.15, int rMin_pixel = 0, int rMax_pixel = 10, double pixel_res = 0.04, double max_inlier_dist = 0.05);

    void set_image(const cv::Mat &img);

    void hough_transform();
    
    void accum_circle(Eigen::MatrixXd& acc_r, const Eigen::Vector2i &position, unsigned int radius);
    
    void accum_pixel(Eigen::MatrixXd& acc_r, const Eigen::Vector2i& pos_idx);

    
    bool detect_circle(circleShape &result, double threshold, int& outlier_cnt);

private:
    
};
#endif
