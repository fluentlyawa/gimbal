#ifndef CP_CONTROL_H_
#define CP_CONTROL_H_

#include <opencv2/core.hpp>

// ============================================================================
// CPControlConfig：所有可调参数。
// 调参只改这个结构体，不用动算法代码。
// ============================================================================
struct CPControlConfig
{
    // ------------------------------------------------------------------------
    // 1. 坐标系参数
    // ------------------------------------------------------------------------
    // G 原点（云台 pitch/yaw 轴交点）在世界坐标系中的位置，单位：米。
    // 必须和 PixelToWorldConfig.gimbal_origin_in_world 一致。
    cv::Vec3d gimbalOriginInWorld;

    // 角度符号修正：+1.0 或 -1.0。
    // 如果下位机反馈的 yaw 正方向与 PixelToWorld 的 rotationY 一致，填 +1.0。
    // 不一致就填 -1.0。
    double yawSign;

    // pitch 符号修正。
    // PixelToWorld 中 rotationX 正方向是“向下”，
    // 如果下位机 pitch 正方向是“向上”，填 -1.0。
    double pitchSign;

    // 角度零偏，单位：度。
    // 云台机械零位不在几何零位时，用来补偿。
    double yawOffsetDeg;
    double pitchOffsetDeg;

    // ------------------------------------------------------------------------
    // 2. EMA 平滑参数
    // ------------------------------------------------------------------------
    // yaw 的 EMA 平滑系数，范围 0~1。
    // 越大越信新测量，越小越平滑。
    // 推荐 0.2~0.5。
    double emaAlphaYaw;

    // pitch 的 EMA 平滑系数。
    double emaAlphaPitch;

    // ------------------------------------------------------------------------
    // 3. 限速参数
    // ------------------------------------------------------------------------
    // 最大 yaw 角速度，单位：度/秒。
    // 目标跳变时，单帧最大变化量 = maxYawRateDegPerSec * dt。
    double maxYawRateDegPerSec;

    // 最大 pitch 角速度，单位：度/秒。
    double maxPitchRateDegPerSec;

    // ------------------------------------------------------------------------
    // 4. 目标跳变检测
    // ------------------------------------------------------------------------
    // 世界坐标跳变超过这个距离（米），认为换了目标，重置平滑器。
    double targetJumpDistanceMeters;

    // ------------------------------------------------------------------------
    // 5. 弹道补偿
    // ------------------------------------------------------------------------
    // 固定 pitch 角度偏移，单位：度。
    // 正数表示抬高炮口（因为 PixelToWorld 中 pitch 向下为正，
    // 所以这里内部会做符号处理，你填“抬高多少度”即可）。
    double ballisticPitchOffsetDeg;

    // 距离每远 1 m，额外抬高多少度。
    // 例如 0.1 表示每远 1 m 抬高 0.1 度。
    double ballisticPitchPerMeterDeg;

    // 是否启用距离相关弹道补偿。
    bool useBallisticDistance;

    // 构造函数：给所有参数默认值。
    CPControlConfig();
};

// ============================================================================
// CPControlInput：一次控制的输入。
// ============================================================================
struct CPControlInput
{
    cv::Point3d targetWorld;   // 世界目标点，单位：米
    double currentYawDeg;      // 当前云台 yaw 反馈，单位：度
    double currentPitchDeg;    // 当前云台 pitch 反馈，单位：度
    double dtSeconds;          // 距上一帧时间，单位：秒
    bool targetValid;          // 本帧目标是否有效

    CPControlInput();
};

// ============================================================================
// CPControlOutput：一次控制的输出。
// ============================================================================
struct CPControlOutput
{
    bool valid;                // 本帧输出是否有效
    double targetYawDeg;       // 期望绝对 yaw，单位：度
    double targetPitchDeg;     // 期望绝对 pitch，单位：度
    double rawYawDeg;          // 未平滑、未限速、未补偿的原始 yaw
    double rawPitchDeg;        // 未平滑、未限速、未补偿的原始 pitch
    double distanceMeters;     // 云台到目标的世界距离
    bool targetJumped;         // 本帧是否检测到目标跳变

    CPControlOutput();
};

// ============================================================================
// CPControl：世界坐标 -> 绝对角度 + EMA 平滑 + 限速 + 弹道补偿。
// ============================================================================
class CPControl
{
public:
    // 构造：可传配置，不传就用默认。
    explicit CPControl(const CPControlConfig& config = CPControlConfig());

    // 修改配置。
    void setConfig(const CPControlConfig& config);

    // 读取配置。
    const CPControlConfig& config() const;
    CPControlConfig& config();

    // 重置平滑器状态。
    void reset();

    // 主接口：World -> 绝对角度。
    CPControlOutput update(const CPControlInput& in);

    // 工具：最短角差，返回 [-180, 180] 度。
    static double shortestAngleDiffDeg(double target, double current);

private:
    CPControlConfig config_;

    // EMA 平滑状态。
    double smoothYawDeg_;
    double smoothPitchDeg_;
    bool hasSmooth_;

    // 上一帧目标世界坐标，用于跳变检测。
    cv::Point3d lastTargetWorld_;
    bool hasLastTarget_;
};

#endif // CP_CONTROL_H_