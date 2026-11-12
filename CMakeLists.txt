cmake_minimum_required(VERSION 3.10)
project(VisionGimbal LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio)
find_package(Threads REQUIRED)

# 包含串口子目录
add_subdirectory(Serial)

# 主程序
add_executable(vision_gimbal
    vision_gimbal.cpp
    CircleRecognition.cpp
)

# 调参工具
add_executable(hsv_tuner hsv_tuner.cpp)
target_include_directories(hsv_tuner PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(hsv_tuner PRIVATE ${OpenCV_LIBS})

# 主程序头文件目录
target_include_directories(vision_gimbal PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/Serial
    ${OpenCV_INCLUDE_DIRS}
)

# 【已修正】把大写 Serial 改成小写 serial
target_link_libraries(vision_gimbal PRIVATE
    ${OpenCV_LIBS} 
    TRoMaC::serial 
    Threads::Threads
)