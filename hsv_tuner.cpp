#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// 全局变量，用于存储滑动条的值
int hmin = 0, smin = 50, vmin = 50;
int hmax = 15, smax = 255, vmax = 255;
Mat img, hsv, mask;

// 鼠标回调函数：获取鼠标悬停处的 HSV 值
void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (event == EVENT_MOUSEMOVE) {
        if (x >= 0 && x < hsv.cols && y >= 0 && y < hsv.rows) {
            Vec3b pixel = hsv.at<Vec3b>(y, x);
            cout << "鼠标位置 (" << x << ", " << y << ") -> "
                 << "H: " << (int)pixel[0] << " "
                 << "S: " << (int)pixel[1] << " "
                 << "V: " << (int)pixel[2] << endl;
        }
    }
}

int main() {
    VideoCapture cap(0, CAP_V4L2);
    if (!cap.isOpened()) {
        cerr << "相机打不开！" << endl;
        return -1;
    }
    cap.set(CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(CAP_PROP_FRAME_HEIGHT, 720);

    namedWindow("Camera", WINDOW_AUTOSIZE);
    namedWindow("Mask", WINDOW_AUTOSIZE);

    createTrackbar("H Min", "Mask", &hmin, 179);
    createTrackbar("H Max", "Mask", &hmax, 179);
    createTrackbar("S Min", "Mask", &smin, 255);
    createTrackbar("S Max", "Mask", &smax, 255);
    createTrackbar("V Min", "Mask", &vmin, 255);
    createTrackbar("V Max", "Mask", &vmax, 255);

    setMouseCallback("Camera", onMouse);

    while (true) {
        cap >> img;
        if (img.empty()) break;

        cvtColor(img, hsv, COLOR_BGR2HSV);

        Scalar lower(hmin, smin, vmin);
        Scalar upper(hmax, smax, vmax);
        inRange(hsv, lower, upper, mask);

        Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
        morphologyEx(mask, mask, MORPH_OPEN, kernel);
        morphologyEx(mask, mask, MORPH_CLOSE, kernel);

        imshow("Camera", img);
        imshow("Mask", mask);

        int key = waitKey(10);
        if (key == 27) break;
    }

    return 0;
}