#ifndef DELAY_AND_FIRE_H_
#define DELAY_AND_FIRE_H_

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <opencv2/core.hpp>
#include <vector>

// ============================================================================
// GimbalSample：一个电机角度采样点。
// 后台线程（1000Hz）会不断往队列里塞这个结构体。
// ============================================================================
struct GimbalSample
{
    double timeSeconds; // 采样发生的绝对时间（秒）
    float yawDeg;       // 此时刻的 Yaw 角（度）
    float pitchDeg;     // 此时刻的 Pitch 角（度）
};

// ============================================================================
// LockFreeRingBuffer：无锁环形队列
// 为什么需要无锁？后台线程 1000Hz 写入，主线程 60Hz 读取，如果用普通的
// std::mutex，频繁抢锁会导致电机采集线程卡顿，从而丢失重要的历史角度。
// 这里利用 std::atomic 的“写头读尾”机制，实现极速并发。
// ============================================================================
class LockFreeRingBuffer
{
public:
    // 构造函数：size 必须是 2 的幂（如 1024, 2048），
    // 这样可以用位运算 (size - 1) 代替取模运算，速度极快。
    explicit LockFreeRingBuffer(std::size_t size);
    ~LockFreeRingBuffer();

    // 后台线程调用：写入一个采样点。
    void push(const GimbalSample& sample);

    // 主线程调用：查询指定时间戳的电机角度。
    // 因为 1000Hz 采样很难刚好命中你要的时间戳，这里使用“线性插值”计算。
    bool getAnglesAt(double queryTime, float& outYaw, float& outPitch) const;

private:
    std::size_t mask_;                 // 取模掩码（size - 1）
    std::vector<GimbalSample> buffer_; // 存储数据的底层数组
    
    // 原子变量：无锁并发的核心
    std::atomic<std::size_t> head_;    // 写入位置（后台线程只改它）
    std::atomic<std::size_t> tail_;    // 读取位置（主线程只改它）
};

// ============================================================================
// DelayAndFireConfig：延迟与开火配置。
// ============================================================================
struct DelayAndFireConfig
{
    // 1. 延时参数 (单位：秒)
    double tCommAndMotorDelay; // t3: 串口通信+电机响应延迟（如 0.01s）
    double tSystemDelay;       // t5: 图像处理、系统调度等延迟（如 0.005s）
    
    // 2. 弹道参数
    double projectileSpeed;      // 弹丸初速度（如 10.0 m/s）
    bool useConstantFlightTime;  // 是否使用常数飞行时间（近距离可直接填常数）
    double constantFlightTime;   // 若使用常数，填这个值（如 0.15s）
    
    // 3. 开火参数
    double fireAngleToleranceDeg; // 角度误差小于此值认为对准（如 1.0 度）
    double fireStableTimeMs;      // 必须连续对准多少毫秒才开火（如 100ms，防抖）
    double fireCooldownMs;        // 两次开火最小间隔（如 600ms，防连发）

    // 4. 物理常量
    double gravity;            // 重力加速度 (9.8 m/s^2)

    DelayAndFireConfig();
};

// ============================================================================
// DelayAndFireOutput：单帧输出结果。
// ============================================================================
struct DelayAndFireOutput
{
    bool valid;                  // 本帧逻辑是否有效
    bool fire;                   // 是否触发开火
    double tCapture;             // 图像曝光时间戳
    float histYawDeg;            // 历史队列插值得到的 yaw（用于调试）
    float histPitchDeg;          // 历史队列插值得到的 pitch（用于调试）
    double tLookahead;           // 总提前量时间（秒）
    double ballisticPitchCompDeg;// 抛物线补偿角度（度）
};

// ============================================================================
// DelayAndFire：时间戳对齐、动态预测、弹道补偿与开火逻辑。
// ============================================================================
class DelayAndFire
{
public:
    explicit DelayAndFire(const DelayAndFireConfig& config = DelayAndFireConfig());

    void setConfig(const DelayAndFireConfig& config);
    const DelayAndFireConfig& config() const;

    // 重置开火状态
    void reset();

    // 给后台线程调用：压入历史角度
    void pushGimbalSample(double timeSeconds, float yawDeg, float pitchDeg);

    // 给主线程调用：查询历史角度（供 PixelToWorld 使用）
    bool getAnglesAt(double queryTime, float& outYaw, float& outPitch) const;

    // 主线程调用：计算预测时间、弹道补偿、判断开火
    // 输入：时间、距离、目标绝对角度、当前真实角度
    // 输出：开火指令、补偿角度等
    DelayAndFireOutput update(
        double tCapture,         // 图像捕获时间
        double tNow,             // 当前系统时间
        double distanceMeters,   // 云台到目标的直线距离
        double targetYawDeg,     // CPControl算出的目标 Yaw
        double targetPitchDeg,   // CPControl算出的目标 Pitch
        double currentYawDeg,    // 电机当前反馈的 Yaw
        double currentPitchDeg); // 电机当前反馈的 Pitch

    // 计算飞行时间 t4
    double computeFlightTime(double distanceMeters) const;
    
    // 计算抛物线补偿角
    double computeBallisticCompensation(double distanceMeters, double flightTime) const;

private:
    DelayAndFireConfig config_;
    LockFreeRingBuffer history_;

    // 开火状态机
    bool isAimed_;
    double aimedStartTime_; // 连续对准的起始时间
    double lastFireTime_;   // 上次开火时间
};

#endif // DELAY_AND_FIRE_H_