#include "CPControl.h"

#include <cmath>

namespace
{
const double kPi = 3.14159265358979323846;
const double kDegToRad = kPi / 180.0;
const double kRadToDeg = 180.0 / kPi;

double dot3(const cv::Vec3d& a, const cv::Vec3d& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

double norm3(const cv::Vec3d& v)
{
    return std::sqrt(dot3(v, v));
}

cv::Vec3d toVec3d(const cv::Point3d& p)
{
    return cv::Vec3d(p.x, p.y, p.z);
}
} // namespace

// ============================================================================
// CPControlConfig 默认值
// ============================================================================
CPControlConfig::CPControlConfig()
    : gimbalOriginInWorld(0.0, 0.0, 0.0),  // G 原点默认在世界原点
      yawSign(1.0),                        // yaw 默认不翻符号
      pitchSign(1.0),                      // pitch 默认不翻符号
      yawOffsetDeg(0.0),                   // yaw 零偏 0
      pitchOffsetDeg(0.0),                 // pitch 零偏 0
      emaAlphaYaw(0.3),                    // yaw EMA 系数 0.3
      emaAlphaPitch(0.3),                  // pitch EMA 系数 0.3
      maxYawRateDegPerSec(180.0),          // yaw 最大 180 度/秒
      maxPitchRateDegPerSec(120.0),        // pitch 最大 120 度/秒
      targetJumpDistanceMeters(0.5),       // 跳变阈值 0.5 m
      ballisticPitchOffsetDeg(0.0),        // 弹道固定补偿 0
      ballisticPitchPerMeterDeg(0.0),      // 弹道距离补偿 0
      useBallisticDistance(false)          // 默认不用距离补偿
{
}

// ============================================================================
// CPControlInput / CPControlOutput 默认构造
// ============================================================================
CPControlInput::CPControlInput()
    : targetWorld(0.0, 0.0, 0.0),
      currentYawDeg(0.0),
      currentPitchDeg(0.0),
      dtSeconds(0.0),
      targetValid(false)
{
}

CPControlOutput::CPControlOutput()
    : valid(false),
      targetYawDeg(0.0),
      targetPitchDeg(0.0),
      rawYawDeg(0.0),
      rawPitchDeg(0.0),
      distanceMeters(0.0),
      targetJumped(false)
{
}

// ============================================================================
// CPControl 构造 / 配置
// ============================================================================
CPControl::CPControl(const CPControlConfig& config)
    : config_(config),
      smoothYawDeg_(0.0),
      smoothPitchDeg_(0.0),
      hasSmooth_(false),
      lastTargetWorld_(0.0, 0.0, 0.0),
      hasLastTarget_(false)
{
}

void CPControl::setConfig(const CPControlConfig& config)
{
    config_ = config;
}

const CPControlConfig& CPControl::config() const
{
    return config_;
}

CPControlConfig& CPControl::config()
{
    return config_;
}

void CPControl::reset()
{
    smoothYawDeg_ = 0.0;
    smoothPitchDeg_ = 0.0;
    hasSmooth_ = false;
    lastTargetWorld_ = cv::Point3d(0.0, 0.0, 0.0);
    hasLastTarget_ = false;
}

// ============================================================================
// 最短角差
// ============================================================================
// 返回 target - current，规约到 [-180, 180]。
// 例如 target=170, current=-170，返回 +20，而不是 +340。
double CPControl::shortestAngleDiffDeg(double target, double current)
{
    double diff = target - current;
    // 把 diff 规约到 [-180, 180]。
    diff = diff - 360.0 * std::floor(diff / 360.0 + 0.5);
    return diff;
}

// ============================================================================
// 主接口：World -> 绝对角度
// ============================================================================
CPControlOutput CPControl::update(const CPControlInput& in)
{
    CPControlOutput out;

    // 1. 检查输入。
    if (!in.targetValid)
    {
        // 目标无效，本帧不更新平滑器，直接返回无效。
        return out;
    }

    if (!std::isfinite(in.targetWorld.x) ||
        !std::isfinite(in.targetWorld.y) ||
        !std::isfinite(in.targetWorld.z) ||
        !std::isfinite(in.currentYawDeg) ||
        !std::isfinite(in.currentPitchDeg) ||
        !std::isfinite(in.dtSeconds) ||
        in.dtSeconds <= 0.0)
    {
        return out;
    }

    // 2. 计算目标相对 G 原点的世界方向。
    const cv::Vec3d origin = config_.gimbalOriginInWorld;
    const cv::Vec3d target = toVec3d(in.targetWorld);
    const cv::Vec3d dWorld = target - origin;

    const double distance = norm3(dWorld);
    out.distanceMeters = distance;

    if (distance < 1e-6)
    {
        // 目标就在云台原点，无法算方向。
        return out;
    }

    // 3. 方向分解求原始 yaw/pitch。
    // 按 PixelToWorld 的旋转约定：
    //   D_W.x =  sin(yaw) * cos(pitch)
    //   D_W.y = -sin(pitch)
    //   D_W.z =  cos(yaw) * cos(pitch)
    //
    // 反解：
    //   yaw   = atan2(D_x, D_z)
    //   pitch = atan2(-D_y, sqrt(D_x^2 + D_z^2))
    const double horizontal = std::sqrt(dWorld[0] * dWorld[0] +
                                        dWorld[2] * dWorld[2]);

    double rawYawRad = std::atan2(dWorld[0], dWorld[2]);
    double rawPitchRad = std::atan2(-dWorld[1], horizontal);

    // 转成度。
    double rawYawDeg = rawYawRad * kRadToDeg;
    double rawPitchDeg = rawPitchRad * kRadToDeg;

    // 4. 应用符号修正和零偏。
    // 注意顺序：先乘符号，再加零偏。
    rawYawDeg = config_.yawSign * rawYawDeg + config_.yawOffsetDeg;
    rawPitchDeg = config_.pitchSign * rawPitchDeg + config_.pitchOffsetDeg;

    out.rawYawDeg = rawYawDeg;
    out.rawPitchDeg = rawPitchDeg;

    // 5. 目标跳变检测。
    // 如果本帧目标世界坐标与上一帧距离超过阈值，认为换了目标，重置平滑器。
    bool targetJumped = false;
    if (hasLastTarget_)
    {
        const cv::Vec3d last = toVec3d(lastTargetWorld_);
        const double jumpDist = norm3(target - last);
        if (jumpDist > config_.targetJumpDistanceMeters)
        {
            targetJumped = true;
            hasSmooth_ = false; // 重置 EMA 状态
        }
    }

    out.targetJumped = targetJumped;

    // 6. EMA 平滑。
    // 第一帧没有历史，直接使用原始值。
    if (!hasSmooth_)
    {
        smoothYawDeg_ = rawYawDeg;
        smoothPitchDeg_ = rawPitchDeg;
        hasSmooth_ = true;
    }
    else
    {
        // yaw 用最短角差平滑，避免 ±180 跳变。
        const double yawDiff = shortestAngleDiffDeg(rawYawDeg, smoothYawDeg_);
        const double alphaYaw = std::max(0.0, std::min(1.0, config_.emaAlphaYaw));
        smoothYawDeg_ = smoothYawDeg_ + alphaYaw * yawDiff;

        // pitch 没有环绕问题，直接平滑。
        const double alphaPitch = std::max(0.0, std::min(1.0, config_.emaAlphaPitch));
        smoothPitchDeg_ = (1.0 - alphaPitch) * smoothPitchDeg_ +
                          alphaPitch * rawPitchDeg;
    }

    // 7. 限速。
    // 计算从当前反馈到平滑目标的角差，单帧最大变化量 = rate * dt。
    double deltaYaw = shortestAngleDiffDeg(smoothYawDeg_, in.currentYawDeg);
    double deltaPitch = smoothPitchDeg_ - in.currentPitchDeg;

    const double maxDeltaYaw = config_.maxYawRateDegPerSec * in.dtSeconds;
    const double maxDeltaPitch = config_.maxPitchRateDegPerSec * in.dtSeconds;

    if (std::fabs(deltaYaw) > maxDeltaYaw)
    {
        deltaYaw = (deltaYaw > 0.0) ? maxDeltaYaw : -maxDeltaYaw;
    }
    if (std::fabs(deltaPitch) > maxDeltaPitch)
    {
        deltaPitch = (deltaPitch > 0.0) ? maxDeltaPitch : -maxDeltaPitch;
    }

    // 8. 输出绝对目标角度 = 当前反馈 + 限速后的增量。
    double targetYawDeg = in.currentYawDeg + deltaYaw;
    double targetPitchDeg = in.currentPitchDeg + deltaPitch;

    // 9. 弹道补偿。
    // 注意：PixelToWorld 中 pitch 正方向是“向下”，
    // 所以“抬高炮口”意味着 pitch 减小。
    // 用户填的 ballisticPitchOffsetDeg 是“抬高多少度”，
    // 这里内部取负号，加到 pitch 上。
    double ballisticComp = config_.ballisticPitchOffsetDeg;
    if (config_.useBallisticDistance)
    {
        ballisticComp += config_.ballisticPitchPerMeterDeg * distance;
    }
    targetPitchDeg -= ballisticComp;

    // 10. 填输出。
    out.valid = true;
    out.targetYawDeg = targetYawDeg;
    out.targetPitchDeg = targetPitchDeg;

    // 11. 更新历史。
    lastTargetWorld_ = in.targetWorld;
    hasLastTarget_ = true;

    return out;
}