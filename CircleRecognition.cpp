#include "CircleRecognition.h"
#include <iostream>

using namespace cv;
using namespace std;

CircleRecognition::CircleRecognition() {
    // 初始化形态学核，避免在每帧循环中重复分配内存
    kernel_ = getStructuringElement(MORPH_RECT, Size(5, 5));
}

void CircleRecognition::loadParams(const string& filename) {
    FileStorage fs(filename, FileStorage::READ);
    if (fs.isOpened()) {
        fs["blue_low"] >> params_.blue_low;
        fs["blue_high"] >> params_.blue_high;
        fs["red_low1"] >> params_.red_low1;
        fs["red_high1"] >> params_.red_high1;
        fs["red_low2"] >> params_.red_low2;
        fs["red_high2"] >> params_.red_high2;
        fs["min_area"] >> params_.min_area;
        fs["max_area"] >> params_.max_area;
        fs["circularity_thresh"] >> params_.circularity_thresh;
        fs.release();
        cout << "参数加载成功: " << filename << endl;
    } else {
        cout << "未找到参数文件，使用默认参数: " << filename << endl;
    }
}

void CircleRecognition::saveParams(const string& filename) {
    FileStorage fs(filename, FileStorage::WRITE);
    if (fs.isOpened()) {
        fs << "blue_low" << params_.blue_low;
        fs << "blue_high" << params_.blue_high;
        fs << "red_low1" << params_.red_low1;
        fs << "red_high1" << params_.red_high1;
        fs << "red_low2" << params_.red_low2;
        fs << "red_high2" << params_.red_high2;
        fs << "min_area" << params_.min_area;
        fs << "max_area" << params_.max_area;
        fs << "circularity_thresh" << params_.circularity_thresh;
        fs.release();
        cout << "参数已保存至: " << filename << endl;
    }
}

void CircleRecognition::switchAimColor() {
    aim_red_ = !aim_red_;
}

Scalar CircleRecognition::getStatusTextColor() const {
    return aim_red_ ? Scalar(0, 0, 255) : Scalar(255, 0, 0); // BGR 红色或蓝色
}

string CircleRecognition::getStatusText() const {
    return aim_red_ ? "Aiming: RED" : "Aiming: BLUE";
}

bool CircleRecognition::detect(Mat& frame, Point2f& out_center, float& out_radius) {
    Mat hsv, mask, mask1, mask2;
    Mat process_frame;

    // 【1. 动态 ROI】判断是否在上一帧目标周围裁剪
    Rect roi_rect;
    if (has_target_last_frame_) {
        int rx = max(0, (int)last_target_center_.x - 200);
        int ry = max(0, (int)last_target_center_.y - 200);
        int rw = min(400, frame.cols - rx);
        int rh = min(400, frame.rows - ry);
        roi_rect = Rect(rx, ry, rw, rh);
    } else {
        roi_rect = Rect(0, 0, frame.cols, frame.rows); // 全图搜索
    }
    
    Mat roi_img = frame(roi_rect);

    // 【2. Scale Down 降采样】缩小图像，提升处理速度
    Mat small_img;
    resize(roi_img, small_img, Size(), scale_, scale_);
    cvtColor(small_img, hsv, COLOR_BGR2HSV);

    // 【3. 提取颜色掩膜】
    if (aim_red_) {
        inRange(hsv, params_.red_low1, params_.red_high1, mask1);
        inRange(hsv, params_.red_low2, params_.red_high2, mask2);
        bitwise_or(mask1, mask2, mask);
    } else {
        inRange(hsv, params_.blue_low, params_.blue_high, mask);
    }

    // 【4. 形态学去噪】
    morphologyEx(mask, mask, MORPH_OPEN, kernel_);
    morphologyEx(mask, mask, MORPH_CLOSE, kernel_);

    // 【5. 轮廓提取与过滤】
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Point2f best_center(-1, -1);
    float best_radius = 0;
    double min_dist = 9999999.0;
    Point2f search_center = has_target_last_frame_ ? last_target_center_ : Point2f(frame.cols/2.0, frame.rows/2.0);

    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        
        // 注意：面积是在降采样图上算的，阈值也需要按比例缩放 (scale^2 = 0.25)
        if (area > params_.min_area * scale_ * scale_ && area < params_.max_area * scale_ * scale_) {
            
            // 包围盒长宽比初筛（提速，干掉长条）
            Rect boundRect = boundingRect(contours[i]);
            float aspect_ratio = (float)boundRect.width / boundRect.height;
            if (aspect_ratio < 0.6 || aspect_ratio > 1.6) continue;

            // 圆度过滤
            double perimeter = arcLength(contours[i], true);
            if (perimeter == 0) continue;
            double circularity = 4 * CV_PI * area / (perimeter * perimeter);
            
            if (circularity > params_.circularity_thresh) {
                // 【6. 质心法抗残缺】用图像矩求质心，替代最小包围圆
                Moments m = moments(contours[i]);
                if (m.m00 != 0) {
                    // 降采样图上的局部质心
                    Point2f local_center(m.m10 / m.m00, m.m01 / m.m00);
                    
                    // 坐标映射回原图: (小图坐标 / scale) + ROI偏移
                    Point2f global_center(local_center.x / scale_ + roi_rect.x, local_center.y / scale_ + roi_rect.y);
                    
                    // 半径映射 (为了画图好看，还是用最小包围圆算半径)
                    Point2f tmp_c; float tmp_r;
                    minEnclosingCircle(contours[i], tmp_c, tmp_r);
                    float global_radius = tmp_r / scale_;

                    // 选出距离搜索中心最近的一个目标
                    double dist = norm(global_center - search_center);
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_center = global_center;
                        best_radius = global_radius;
                    }
                }
            }
        }
    }

    // 更新状态并输出
    if (best_center.x != -1) {
        has_target_last_frame_ = true;
        last_target_center_ = best_center;
        out_center = best_center;
        out_radius = best_radius;
        return true;
    } else {
        has_target_last_frame_ = false; // 丢失目标，下一帧全图搜索
        return false;
    }
}

void CircleRecognition::drawTarget(Mat& frame, const Point2f& center, float radius) {
    // 绿色实心圆 (圆心)
    circle(frame, center, 5, Scalar(0, 255, 0), FILLED);
    // 青色轮廓 (拟合圆)
    circle(frame, center, (int)radius, Scalar(255, 255, 0), 2);

    // 在画面上打印坐标
    string posText = "U=" + to_string((int)center.x) + ", V=" + to_string((int)center.y);
    putText(frame, posText, Point(20, 50), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
}