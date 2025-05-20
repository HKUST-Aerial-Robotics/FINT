/*******************************************************
Copyright (C) 2025, Aerial Robotics Group, Hong Kong University of Science and Technology
This file is part of FINT.
Licensed under the GNU General Public License v3.0;
you may not use this file except in compliance with the License.
*******************************************************/
#pragma once
#include <Eigen/Dense>
#include <array>
#include <tuple>
#include "opencv2/opencv.hpp"

using namespace cv;

class rectangleShape{
    public:
    
    inline void set_angle(const double &_angle){
        angle = _angle;
    }
    
    inline void set_edge_half(const double &_half_la, const double &_half_lb){
        set_edge(2.0 * _half_la, 2.0 * _half_lb);
    }
    
    inline void set_edge(const double &_la, const double &_lb){
        la = _la;
        lb = _lb;
    }
    
    inline void set_center(const Eigen::Vector2d &_center){
        center = _center;
    }
    
    inline void set_shape_half(const double &_angle, const double &_half_la, const double &_half_lb){
        set_angle(_angle);
        set_edge_half(_half_la, _half_lb);
    }
    
    std::array<Eigen::Vector2d, 4> output_corners(){
        std::array<Eigen::Vector2d, 4> corners;
        
        Eigen::Matrix2d rot_m;
        rot_m << cos(angle), -sin(angle), sin(angle), cos(angle);
        
        corners[0] = center + rot_m * Eigen::Vector2d(0.5 * la, 0.5 * lb);
        corners[1] = center + rot_m * Eigen::Vector2d(0.5 * la, -0.5 * lb);
        corners[2] = center + rot_m * Eigen::Vector2d(-0.5 * la, -0.5 * lb);
        corners[3] = center + rot_m * Eigen::Vector2d(-0.5 * la, 0.5 * lb);
        
        return corners;
    }
    
    inline Eigen::Vector2d get_center(){
        return center;
    }
    
    inline Eigen::Vector2d get_edge(){
        return Eigen::Vector2d(la,lb);
    }
    
    inline double get_angle(){
        return angle;
    }
    
    private:
    
    Eigen::Vector2d center;
    // angle between la and x-axis
    double angle;
    double la;
    double lb;
    
};

class HoughRectangle {
    
    protected:
    int m_thetaBins;
    int m_thetaMin;
    int m_thetaMax;
    int m_rhoBins;
    double m_rhoRes;
    double m_thetaRes;
    
    double m_pixelRes;
    
    double m_squared_max_inlier_pixel_dist;
    
    int m_r_min;
    int m_r_max;
    int m_enhance_edge_length;
    
    Eigen::Vector2i m_img_center;
    
    Eigen::MatrixXd acc;
    Eigen::MatrixXd enhanced_acc;
    
    std::vector<Eigen::Vector2d> pts;
    
    struct hough_cmp_gt
    {
        hough_cmp_gt(const Eigen::MatrixXd& _hough) : hough(_hough) {}
        inline bool operator()(std::array<int, 2> l1, std::array<int, 2> l2) const
        {
            return hough(l1[0], l1[1]) > hough(l2[0], l2[1]) || (hough(l1[0], l1[1]) == hough(l2[0], l2[1]) && l1[0] * hough.cols()  + l1[1] < l2[0] * hough.cols()  + l2[1]);
        }
        const Eigen::MatrixXd& hough;
    };
    
    
    Eigen::VectorXd m_theta_vec;
    Eigen::VectorXd m_rho_vec;
    
    Mat m_img;
    
    public:
    
    HoughRectangle();
    HoughRectangle(int thetaBins = 256, int rhoBins = 256, double thetaMin = -90,
        double thetaMax = 90, int r_min = 0, int r_max = std::numeric_limits<int>::max(), int enhance_edge_length = 1, double pixel_res = 0.04, double max_inlier_dist = 0.05);
        HoughRectangle(cv::Mat &img, int thetaBins = 256, int rhoBins = 256, double thetaMin = -90,
            double thetaMax = 90, int r_min = 0, int r_max = std::numeric_limits<int>::max(), int enhance_edge_length = 1,  double pixel_res = 0.04, double max_inlier_dist = 0.05);  // declaration
            
            public:
            
            void set_image(const cv::Mat &img);
            
            void hough_transform();
            
            void windowed_hough(const int &r_min, const int &r_max);
            
            void enhance_hough(const int& half_w, const int& half_h);
            
            void ring(const int &r_min, const int &r_max);
            
            std::tuple<std::vector<double>, std::vector<double>> index_rho_theta(const std::vector<std::array<int, 2>> &indexes);
            
            std::vector<std::array<int, 2>> find_local_maximum(const Eigen::MatrixXd& hough, const double& threshold);
            
            bool detect_rectangle_from_hough_lines(const std::vector<std::array<int, 2>>& line_indexes, const double &T_rho, const double &T_t, const double &T_L, const double &T_alpha, const double &E_weight, rectangleShape &result);
            
            bool detect_rectangle(const double threshold_line, const double T_rho, const double T_t, const double T_L, const double T_alpha, const double E_weight, rectangleShape &result, int& outlier_cnt);
            
            private:
            
            int count_outliers(rectangleShape &result);
            
};