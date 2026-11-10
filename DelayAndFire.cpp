#include "DelayAndFire.h"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace
{
const double kRadToDeg = 180.0 / 3.14159265358979323846;

// 工具：求最短角度差（处理 ±180 度的翻转）
// 例如 170 和 -170，直接相减是 340，实际上最短距离是 -20（跨越边界）
double shortestAngleDiffDeg(double target, double current)
{
    double diff = target - current;
    diff = diff - 360.0 * std::floor(diff / 360.0 + 0.5);
    return diff;
}
} // namespace

// ============================================================================
// LockFreeRingBuffer 实现
// ============================================================================
LockFreeRingBuffer::LockFreeRingBuffer(std::size_t size)
    : mask_(size - 1), buffer_(size), head_(0), tail_(0)
{
    if ((size & (size - 1)) != 0)
    {
        std::cerr << "[RingBuffer] 警告: 大小必须是 2 的幂（如 1024, 2048）\n";
    }
}

LockFreeRingBuffer::~LockFreeRingBuffer() = default;

void LockFreeRingBuffer::push(const GimbalSample& sample)
{
    // memory_order_relaxed：只保证原子性，不保证顺序，因为是写线程独享 head
    const std::size_t currentHead = head_.load(std::memory_order_relaxed);
    const std::size_t nextHead = (currentHead + 1) & mask_;
    
    // 如果队列满了，覆盖最旧的数据（移动 tail）
    if (nextHead == tail_.load(std::memory_order_acquire))
    {
        tail_.store((nextHead + 1) & mask_, std::memory_order_release);
    }
    
    // 写入数据
    buffer_[currentHead] = sample;
    // memory_order_release：确保 buffer_ 的写入对读线程可见后，再更新 head
    head_.store(nextHead, std::memory_order_release);
}

bool LockFreeRingBuffer::getAnglesAt(double queryTime, float& outYaw, float& outPitch) const
{
    const std::size_t currentTail = tail_.load(std::memory_order_acquire);
    const std::size_t currentHead = head_.load(std::memory_order_acquire);
    
    if (currentTail == currentHead) return false; // 队列为空

    // 为了二分查找和插值，将环形队列里的数据拷贝到一个临时 vector 中。
    // 每次只拷贝当前存在的样本（通常几十到几百个），开销极小。
    std::vector<GimbalSample> temp;
    temp.reserve(128); 
    
    std::size_t idx = currentTail;
    while (idx != currentHead)
    {
        temp.push_back(buffer_[idx]);
        idx = (idx + 1) & mask_;
    }

    // 如果查询时间比最旧的还旧，返回最旧的
    if (queryTime <= temp.front().timeSeconds)
    {
        outYaw = temp.front().yawDeg;
        outPitch = temp.front().pitchDeg;
        return true;
    }
    // 如果查询时间比最新的还新，返回最新的
    if (queryTime >= temp.back().timeSeconds)
    {
        outYaw = temp.back().yawDeg;
        outPitch = temp.back().pitchDeg;
        return true;
    }

    // 二分查找：找到第一个 >= queryTime 的位置
    auto it = std::lower_bound(temp.begin(), temp.end(), queryTime,
        [](const GimbalSample& s, double t) { return s.timeSeconds < t; });

    if (it == temp.begin()) return false;
    
    auto prev = it - 1;
    const double t1 = prev->timeSeconds;
    const double t2 = it->timeSeconds;
    
    // 线性插值：(y - y1) / (t - t1) = (y2 - y1) / (t2 - t1)
    if (std::fabs(t2 - t1) < 1e-6)
    {
        outYaw = it->yawDeg;
        outPitch = it->pitchDeg;
    }
    else
    {
        const double ratio = (queryTime - t1) / (t2 - t1);
        outYaw = prev->yawDeg + ratio * (it->yawDeg - prev->yawDeg);
        outPitch = prev->pitchDeg + ratio * (it->pitchDeg - prev->pitchDeg);
    }
    
    return true;
}

// ============================================================================
// DelayAndFire 实现
// ============================================================================
DelayAndFireConfig::DelayAndFireConfig()
    : tCommAndMotorDelay(0.01), // 默认 10ms
      tSystemDelay(0.005),      // 默认 5ms
      projectileSpeed(10.0),    // 10 m/s
      useConstantFlightTime(true),
      constantFlightTime(0.15), // 默认 0.15s
      fireAngleToleranceDeg(1.0),
      fireStableTimeMs(100.0),  // 100ms 稳定期（约 6 帧）
      fireCooldownMs(600.0),    // 600ms 间隔
      gravity(9.8)
{
}

DelayAndFire::DelayAndFire(const DelayAndFireConfig& config)
    : config_(config),
      history_(2048), // 1000Hz 下存 2 秒历史数据
      isAimed_(false),
      aimedStartTime_(0.0),
      lastFireTime_(0.0)
{
}

void DelayAndFire::setConfig(const DelayAndFireConfig& config) { config_ = config; }
const DelayAndFireConfig& DelayAndFire::config() const { return config_; }

void DelayAndFire::reset()
{
    isAimed_ = false;
    aimedStartTime_ = 0.0;
    lastFireTime_ = 0.0;
}

void DelayAndFire::pushGimbalSample(double timeSeconds, float yawDeg, float pitchDeg)
{
    GimbalSample sample;
    sample.timeSeconds = timeSeconds;
    sample.yawDeg = yawDeg;
    sample.pitchDeg = pitchDeg;
    history_.push(sample);
}

bool DelayAndFire::getAnglesAt(double queryTime, float& outYaw, float& outPitch) const
{
    return history_.getAnglesAt(queryTime, outYaw, outPitch);
}

// 计算 t4：弹丸飞行时间
double DelayAndFire::computeFlightTime(double distanceMeters) const
{
    if (config_.useConstantFlightTime) return config_.constantFlightTime;
    if (config_.projectileSpeed > 1e-6) return distanceMeters / config_.projectileSpeed;
    return 0.0;
}

// 计算抛物线补偿角
// 原理：弹丸受重力下坠 h = 0.5 * g * t^2
// 补偿角 = atan(下坠高度 / 距离)
double DelayAndFire::computeBallisticCompensation(double distanceMeters, double flightTime) const
{
    if (distanceMeters < 1e-6) return 0.0;
    const double drop = 0.5 * config_.gravity * flightTime * flightTime;
    const double compensationRad = std::atan2(drop, distanceMeters);
    return compensationRad * kRadToDeg;
}

DelayAndFireOutput DelayAndFire::update(
    double tCapture,
    double tNow,
    double distanceMeters,
    double targetYawDeg,
    double targetPitchDeg,
    double currentYawDeg,
    double currentPitchDeg)
{
    DelayAndFireOutput out;
    out.valid = false;
    out.fire = false;
    out.tCapture = tCapture;
    
    // 1. 调试用：查一下历史角度（主程序其实已经查过了，这里为了输出日志）
    history_.getAnglesAt(tCapture, out.histYawDeg, out.histPitchDeg);

    // 2. 计算总提前量时间 tLookahead
    const double tFlight = computeFlightTime(distanceMeters);
    // 过去的延迟（图像处理延迟） + 未来的延迟（通信+电机） + 飞行时间
    const double tPast = tNow - tCapture;
    const double tLookahead = tPast + config_.tSystemDelay + config_.tCommAndMotorDelay + tFlight;
    out.tLookahead = tLookahead;
    
    // 3. 计算抛物线补偿
    const double pitchComp = computeBallisticCompensation(distanceMeters, tFlight);
    out.ballisticPitchCompDeg = pitchComp;

    // 4. 判断是否在瞄准范围
    const double yawErr = std::fabs(shortestAngleDiffDeg(targetYawDeg, currentYawDeg));
    const double pitchErr = std::fabs(targetPitchDeg - currentPitchDeg);

    const bool isAligned = (yawErr < config_.fireAngleToleranceDeg) && 
                           (pitchErr < config_.fireAngleToleranceDeg);

    // 5. 状态机：基于时间的防抖开火
    bool shouldFire = false;
    if (isAligned)
    {
        // 记录刚开始对准的时间
        if (!isAimed_)
        {
            isAimed_ = true;
            aimedStartTime_ = tNow;
        }
        
        const double aimedDurationMs = (tNow - aimedStartTime_) * 1000.0;
        const double sinceLastFireMs = (tNow - lastFireTime_) * 1000.0;

        // 必须连续对准超过 fireStableTimeMs，且距离上次开火超过冷却时间
        if (aimedDurationMs >= config_.fireStableTimeMs && 
            sinceLastFireMs >= config_.fireCooldownMs)
        {
            shouldFire = true;
            lastFireTime_ = tNow;
            isAimed_ = false; // 开火后强制打断连续状态，防止连射
        }
    }
    else
    {
        // 一旦偏离，立刻打断连续对准计时
        isAimed_ = false;
        aimedStartTime_ = 0.0;
    }

    out.valid = true;
    out.fire = shouldFire;
    return out;
}