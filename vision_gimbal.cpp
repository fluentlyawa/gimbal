#include "Serial.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/calib3d.hpp>
#include <chrono>
#include <cmath>
#include <iostream>
#include <mutex>
#include <thread>
#include <atomic>
#include <string>
#include <vector>

#include "CircleRecognition.h"
#include "PixelToWorld.h"
#include "DelayAndFire.h"
#include "FittingCircle.h"

using namespace std;

int cameraIndex = 0;
int width = 1280;
int height = 720;
int fps = 60;

// 【关键修改】降低曝光时间，减少拖影，压暗背景
int cameraAutoExposure = 1; 
int cameraExposure = 50;   

bool useSerial = true; 
int serialPort = 1;
int serialBaudRate = 230400;

static std::chrono::steady_clock::time_point g_programStart;
static CircleRecognition circleRec;
cv::Mat map1, map2;

static PixelToWorldConfig makePixelConfig() {
    PixelToWorldConfig cfg;
    cfg.fx = 1374.222; cfg.fy = 1374.027;
    cfg.cx = 685.776; cfg.cy = 361.667;
    cfg.T_camera_to_gimbal = PixelToWorld::makeRigidTransform(
        cv::Matx33d(1,0,0, 0,-1,0, 0,0,1), cv::Vec3d(0, 0.1102, 0.00934));
    cfg.gimbal_origin_in_world = cv::Vec3d(0, 0, 0);
    cfg.target_plane_normal_world = cv::Vec3d(0, 0, 1); 
    cfg.target_plane_offset = 2.0; // 记得改你的真实距离
    return cfg;
}
static PixelToWorld pixelToWorld(makePixelConfig());

static DelayAndFireConfig makeDelayConfig() {
    DelayAndFireConfig cfg;
    cfg.tCommAndMotorDelay = 0.010;
    cfg.tSystemDelay = 0.005;
    return cfg;
}
static DelayAndFire delayAndFire(makeDelayConfig());

static FittingCircleConfig makeFitConfig() {
    FittingCircleConfig cfg;
    cfg.predictAheadSeconds = 0.15; 
    cfg.useFixedPlaneNormal = true;
    cfg.planeNormal = cv::Vec3d(0, 0, 1);
    cfg.minRadius = 0.01; 
    cfg.maxRadius = 2.0; 
    cfg.minPointsToFit = 5; 
    cfg.maxRmsError = 0.5; 
    return cfg;
}
static FittingCircle fittingCircle(makeFitConfig());

TRoMaC::Uart uart;
TRoMaC::GimbalFeedback receiveData{};
std::mutex receiveMutex;
bool hasFeedback = false;
std::chrono::steady_clock::time_point receivedAt;
std::atomic<bool> running(true);

void receiveLoop() {
    while (running.load()) {
        if (!uart.ReadData()) continue;
        TRoMaC::GimbalFeedback data{};
        if (uart.tryGetGimbalFeedback(data) && std::isfinite(data.yaw_angle) && std::isfinite(data.pitch_angle)) {
            std::lock_guard<std::mutex> lock(receiveMutex);
            receiveData = data;
            receivedAt = std::chrono::steady_clock::now();
            hasFeedback = true;
        }
    }
}

void motorLoggingLoop() {
    using namespace std::chrono;
    auto nextTime = steady_clock::now();
    while (running.load()) {
        TRoMaC::GimbalFeedback localFeedback;
        bool valid = false;
        {
            std::lock_guard<std::mutex> lock(receiveMutex);
            localFeedback = receiveData;
            valid = hasFeedback;
        }
        if (valid) {
            const double tNow = duration<double>(steady_clock::now() - g_programStart).count();
            delayAndFire.pushGimbalSample(tNow, localFeedback.yaw_angle, localFeedback.pitch_angle);
        }
        nextTime += milliseconds(1);
        std::this_thread::sleep_until(nextTime);
    }
}

int main() {
    cv::VideoCapture camera(cameraIndex, cv::CAP_V4L2);
    if (!camera.isOpened()) return 1;
    camera.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    camera.set(cv::CAP_PROP_FRAME_WIDTH, width);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    camera.set(cv::CAP_PROP_FPS, fps);
    camera.set(cv::CAP_PROP_AUTO_EXPOSURE, cameraAutoExposure);
    camera.set(cv::CAP_PROP_EXPOSURE, cameraExposure); 
    camera.set(cv::CAP_PROP_AUTO_WB, 1); 

    cv::Mat cameraMatrix = (cv::Mat_<double>(3,3) << 
        1374.22213315259, 0, 685.7755805760301,
        0, 1374.027198266754, 361.6667378499474,
        0, 0, 1);
    cv::Mat distCoeffs = (cv::Mat_<double>(1,5) << 
        0.0198620468, 0.0105557462, -0.0002627896, 0.0000933322, -0.0762669442);
    cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(),
                                cameraMatrix, cv::Size(width, height), 
                                CV_16SC2, map1, map2);

    if (useSerial && !uart.Open(serialPort, serialBaudRate)) {
        std::cerr << "【警告】串口打开失败！程序使用 0 度角计算坐标。\n";
        useSerial = false;
    }

    g_programStart = std::chrono::steady_clock::now();
    std::thread receiver, motorLogger;
    if (useSerial) {
        receiver = std::thread(receiveLoop);
        motorLogger = std::thread(motorLoggingLoop);
    }

    cv::Mat image, undistortedImage;
    while (running.load()) {
        camera >> image;
        if (image.empty()) break;

        cv::remap(image, undistortedImage, map1, map2, cv::INTER_LINEAR);
        const double tCapture = std::chrono::duration<double>(std::chrono::steady_clock::now() - g_programStart).count();

        cv::Point2f targetCenter;
        float targetRadius = 0.0f;
        bool found = circleRec.detect(undistortedImage, targetCenter, targetRadius);

        // 【恢复 Mask 窗口】
        cv::Mat mask = circleRec.getLastMask();
        if (!mask.empty()) {
            cv::imshow("Mask", mask);
        }

        if (found) {
            circleRec.drawTarget(undistortedImage, targetCenter, targetRadius);

            float histYaw = 0.0f, histPitch = 0.0f;
            bool hasHistory = delayAndFire.getAnglesAt(tCapture, histYaw, histPitch);
            
            if (!hasHistory) {
                histYaw = 0.0f; 
                histPitch = 0.0f;
            }

            BackProjectionResult bp = pixelToWorld.BackProjection(targetCenter.x, targetCenter.y, histPitch, histYaw);
            
            cv::putText(undistortedImage, "=== FITTING & PREDICTING ===", cv::Point(20, 150), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 255), 2);

            if (bp.valid) {
                fittingCircle.addObservation(bp.world_point, tCapture);
                
                std::string pointCountStr = "History Points: " + std::to_string(fittingCircle.history().size());
                cv::putText(undistortedImage, pointCountStr, cv::Point(20, 180), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

                const double tNow = std::chrono::duration<double>(std::chrono::steady_clock::now() - g_programStart).count();
                PredictResult pred = fittingCircle.predict(tNow);

                if (pred.valid) {
                    cv::putText(undistortedImage, 
                        cv::format("Fit: R=%.2f, Omega=%.2f rad/s", 
                        fittingCircle.currentFit().radius, pred.angularVelocityRad),
                        cv::Point(20, 210), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);

                    cv::putText(undistortedImage, 
                        cv::format("Pred Pos: (%.2f, %.2f)", pred.predictedPoint.x, pred.predictedPoint.y),
                        cv::Point(20, 240), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);
                } else {
                    cv::putText(undistortedImage, "Collecting...", cv::Point(20, 210), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 165, 255), 2);
                }
            }
        } else {
            cv::putText(undistortedImage, "NO TARGET", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 2);
        }

        cv::imshow("Camera", undistortedImage);

        int key = cv::waitKey(3);
        if (key == 'q' || key == 27) break;
        if (key == 'c' || key == 'C') circleRec.switchAimColor();
    }

    running.store(false);
    if (receiver.joinable()) receiver.join();
    if (motorLogger.joinable()) motorLogger.join();
    uart.Close();
    camera.release();
    cv::destroyAllWindows();
    return 0;
}