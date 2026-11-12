#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include "CircleRecognition.h"

using namespace std;

int cameraIndex = 0;
int width = 1280;
int height = 720;
int fps = 60;

// 【修改这里】改回手动曝光，并降低亮度，消除墙壁反光
int cameraAutoExposure = 1; // 1 表示手动
int cameraExposure = 150;   // 150 左右（如果画面还是太亮，就改到 100）

int main() {
    cv::VideoCapture camera(cameraIndex, cv::CAP_V4L2);
    if (!camera.isOpened()) {
        std::cerr << "相机打开失败！\n";
        return 1;
    }
    camera.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    camera.set(cv::CAP_PROP_FRAME_WIDTH, width);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    camera.set(cv::CAP_PROP_FPS, fps);
    
    camera.set(cv::CAP_PROP_AUTO_EXPOSURE, cameraAutoExposure);
    camera.set(cv::CAP_PROP_EXPOSURE, cameraExposure); // 设定手动曝光值
    camera.set(cv::CAP_PROP_AUTO_WB, 1); 

    CircleRecognition circleRec;
    // 强制不读 yml，用头文件参数
    // circleRec.loadParams("circle_params.yml");

    cv::Mat image;
    while (true) {
        camera >> image;
        if (image.empty()) break;

        cv::Point2f center;
        float radius = 0.0f;
        bool found = circleRec.detect(image, center, radius);

        cv::Mat mask = circleRec.getLastMask();
        if (!mask.empty()) {
            cv::imshow("Mask", mask);
        }

        if (found) {
            circleRec.drawTarget(image, center, radius);
        } else {
            cv::putText(image, "NO TARGET", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 2);
        }

        cv::putText(image, circleRec.getStatusText(), cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, circleRec.getStatusTextColor(), 2);

        cv::imshow("Camera", image);
        int key = cv::waitKey(3);
        if (key == 'q' || key == 27) break;
        if (key == 'c' || key == 'C') circleRec.switchAimColor(); 
        if (key == 's' || key == 'S') circleRec.saveParams("circle_params.yml"); 
    }

    camera.release();
    cv::destroyAllWindows();
    return 0;
}