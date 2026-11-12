#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct AlgoParams {
    // 恢复你之前调好的完美参数
    cv::Scalar blue_low = cv::Scalar(80, 60, 70); 
    cv::Scalar blue_high = cv::Scalar(95, 255, 255);
    
    // 红色
    cv::Scalar red_low1 = cv::Scalar(0, 50, 50);
    cv::Scalar red_high1 = cv::Scalar(15, 255, 255);
    cv::Scalar red_low2 = cv::Scalar(160, 50, 50);
    cv::Scalar red_high2 = cv::Scalar(179, 255, 255);

    int min_area = 100;
    int max_area = 50000;
    float circularity_thresh = 0.1; 
};

class CircleRecognition {
public:
    CircleRecognition();
    ~CircleRecognition() = default;

    void loadParams(const std::string& filename);
    void saveParams(const std::string& filename);

    void switchAimColor();
    bool isAimingRed() const { return aim_red_; }
    cv::Scalar getStatusTextColor() const;
    std::string getStatusText() const;

    bool detect(cv::Mat& frame, cv::Point2f& out_center, float& out_radius);
    void drawTarget(cv::Mat& frame, const cv::Point2f& center, float radius);

    cv::Mat getLastMask() const { return last_mask_; }

private:
    AlgoParams params_;
    bool aim_red_ = false;
    cv::Mat kernel_;

    bool has_target_last_frame_ = false;
    cv::Point2f last_target_center_;
    float last_target_radius_ = 0.0f; 
    const float scale_ = 0.5f;
    cv::Mat last_mask_;
};