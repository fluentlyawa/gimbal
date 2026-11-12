#include "CircleRecognition.h"
#include <iostream>

using namespace cv;
using namespace std;

CircleRecognition::CircleRecognition() {
    kernel_ = getStructuringElement(MORPH_RECT, Size(5, 5));
}

void CircleRecognition::loadParams(const string& filename) {
    FileStorage fs(filename, FileStorage::READ);
    if (fs.isOpened()) {
        fs["blue_low"] >> params_.blue_low;
        fs["blue_high"] >> params_.blue_high;
        fs.release();
    }
}

void CircleRecognition::saveParams(const string& filename) {
    FileStorage fs(filename, FileStorage::WRITE);
    if (fs.isOpened()) {
        fs << "blue_low" << params_.blue_low;
        fs << "blue_high" << params_.blue_high;
        fs.release();
    }
}

void CircleRecognition::switchAimColor() {
    aim_red_ = !aim_red_;
}

Scalar CircleRecognition::getStatusTextColor() const {
    return aim_red_ ? Scalar(0, 0, 255) : Scalar(255, 0, 0);
}

string CircleRecognition::getStatusText() const {
    return aim_red_ ? "Aiming: RED" : "Aiming: BLUE";
}

bool CircleRecognition::detect(Mat& frame, Point2f& out_center, float& out_radius) {
    Mat hsv, mask, mask1, mask2;
    
    Rect roi_rect;
    if (has_target_last_frame_) {
        int rx = max(0, (int)last_target_center_.x - 200);
        int ry = max(0, (int)last_target_center_.y - 200);
        int rw = min(400, frame.cols - rx);
        int rh = min(400, frame.rows - ry);
        roi_rect = Rect(rx, ry, rw, rh);
    } else {
        roi_rect = Rect(0, 0, frame.cols, frame.rows);
    }
    
    Mat roi_img = frame(roi_rect);
    Mat small_img;
    resize(roi_img, small_img, Size(), scale_, scale_);
    cvtColor(small_img, hsv, COLOR_BGR2HSV);

    if (aim_red_) {
        inRange(hsv, params_.red_low1, params_.red_high1, mask1);
        inRange(hsv, params_.red_low2, params_.red_high2, mask2);
        bitwise_or(mask1, mask2, mask);
    } else {
        inRange(hsv, params_.blue_low, params_.blue_high, mask);
    }

    Mat kernel_small = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(mask, mask, MORPH_OPEN, kernel_small);
    
    Mat kernel_close = getStructuringElement(MORPH_ELLIPSE, Size(15, 15));
    morphologyEx(mask, mask, MORPH_CLOSE, kernel_close);

    last_mask_ = mask; 

    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Point2f best_center(-1, -1);
    float best_radius = 0;
    double min_dist = 9999999.0;
    Point2f search_center = has_target_last_frame_ ? last_target_center_ : Point2f(frame.cols/2.0, frame.rows/2.0);

    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area < 100 * scale_ * scale_) continue; 

        Point2f tmp_c; float tmp_r;
        minEnclosingCircle(contours[i], tmp_c, tmp_r);
        float global_radius = tmp_r / scale_;
        
        // 尺寸拦截：防小杂点，防大墙壁
        if (global_radius < 20 || global_radius > 150) continue; 

        Rect boundRect = boundingRect(contours[i]);
        float aspect_ratio = (float)boundRect.width / boundRect.height;
        
        // 【唯一核心防御】必须是正方形包围盒，直接秒杀左边长条反光！
        if (aspect_ratio < 0.7 || aspect_ratio > 1.4) continue;

        Point2f global_center(tmp_c.x / scale_ + roi_rect.x, tmp_c.y / scale_ + roi_rect.y);
        double dist = norm(global_center - search_center);
        
        if (dist < min_dist) {
            min_dist = dist;
            best_center = global_center;
            best_radius = global_radius;
        }
    }

    if (best_center.x != -1) {
        if (!has_target_last_frame_) {
            last_target_center_ = best_center;
            last_target_radius_ = best_radius;
        } else {
            // EMA 平滑
            last_target_center_.x = last_target_center_.x * 0.8f + best_center.x * 0.2f;
            last_target_center_.y = last_target_center_.y * 0.8f + best_center.y * 0.2f;
            last_target_radius_ = last_target_radius_ * 0.8f + best_radius * 0.2f;
        }

        has_target_last_frame_ = true;
        out_center = last_target_center_;
        out_radius = last_target_radius_;
        return true;
    } else {
        has_target_last_frame_ = false;
        return false;
    }
}

void CircleRecognition::drawTarget(Mat& frame, const Point2f& center, float radius) {
    int half_side = (int)radius;
    int left = center.x - half_side;
    int top = center.y - half_side;
    int right = center.x + half_side;
    int bottom = center.y + half_side;

    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0), 2);
    line(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0), 1);
    line(frame, Point(right, top), Point(left, bottom), Scalar(0, 255, 0), 1);
    circle(frame, center, 3, Scalar(0, 0, 255), FILLED);

    string posText = "U=" + to_string((int)center.x) + ", V=" + to_string((int)center.y);
    putText(frame, posText, Point(20, 50), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
}