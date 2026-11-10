#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// 算法参数结构体
struct AlgoParams {
    cv::Scalar blue_low = cv::Scalar(100, 70, 50);
    cv::Scalar blue_high = cv::Scalar(140, 255, 255);
    cv::Scalar red_low1 = cv::Scalar(0, 70, 50);
    cv::Scalar red_high1 = cv::Scalar(10, 255, 255);
    cv::Scalar red_low2 = cv::Scalar(160, 70, 50);
    cv::Scalar red_high2 = cv::Scalar(180, 255, 255);
    int min_area = 500;
    int max_area = 50000;
    float circularity_thresh = 0.6;
};

class CircleRecognition {
public:
    CircleRecognition();
    ~CircleRecognition() = default;

    // 加载/保存参数
    void loadParams(const std::string& filename);
    void saveParams(const std::string& filename);

    // 切换瞄准颜色
    void switchAimColor();
    bool isAimingRed() const { return aim_red_; }
    cv::Scalar getStatusTextColor() const;
    std::string getStatusText() const;

    // 核心检测函数：输入原始图像，输出是否找到目标，以及目标在原图的坐标和半径
    bool detect(cv::Mat& frame, cv::Point2f& out_center, float& out_radius);

    // 可视化绘制函数
    void drawTarget(cv::Mat& frame, const cv::Point2f& center, float radius);

private:
    AlgoParams params_;
    bool aim_red_ = false;
    cv::Mat kernel_;

    // 动态 ROI 状态记录
    bool has_target_last_frame_ = false;
    cv::Point2f last_target_center_;

    // 速度优化：降采样比例
    const float scale_ = 0.5f;
};