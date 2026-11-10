#ifndef FITTING_CIRCLE_H_
#define FITTING_CIRCLE_H_

// OpenCV 核心头文件：cv::Point3d、cv::Vec3d、cv::Matx33d 等。
#include <opencv2/core.hpp>

// C++ 标准库：size_t、deque、vector。
#include <cstddef>
#include <deque>
#include <vector>

// ============================================================================
// FittingCircleConfig：所有可调参数。
// 调参只改这里，不用动 .cpp。
// ============================================================================
struct FittingCircleConfig
{
    // ------------------------------------------------------------------------
    // 1. 预测参数
    // ------------------------------------------------------------------------
    // 预测多少秒后的位置，补偿曝光 + 传输 + 处理延迟。
    // 0.15 表示预测 150 ms 后。
    double predictAheadSeconds;

    // ------------------------------------------------------------------------
    // 2. 角速度融合
    // ------------------------------------------------------------------------
    // 你手动设置的能量机关转速，单位 rad/s。
    // 0 表示“没有先验，完全信观测”。
    double givenAngularVelocityRad;

    // 给定速度的权重，0~1。
    // 0 完全信观测，1 完全信给定。仅 useKalman=false 时生效。
    double givenVelocityWeight;

    // 是否用一维卡尔曼融合观测和给定速度。
    // true 时忽略 givenVelocityWeight，用 Q/R 自动融合。
    bool useKalman;

    // 卡尔曼过程噪声：越大越信观测。
    double kalmanProcessNoise;

    // 卡尔曼观测噪声：越大越信给定。
    double kalmanMeasureNoise;

    // 加权线性回归时间常数，单位秒。
    // 越小越只信最近的点，越大越平滑。
    double velocityWeightTau;

    // 角速度限幅，单位 rad/s。
    // 估计值超出范围会被夹住，防止飞掉。
    double minAngularVelocityRad;
    double maxAngularVelocityRad;

    // ------------------------------------------------------------------------
    // 3. 历史点管理
    // ------------------------------------------------------------------------
    // 最多保留多少个观测点。
    std::size_t maxHistoryPoints;

    // 超过这个年龄（秒）的点直接丢弃。
    double maxObservationAgeSeconds;

    // 最少多少个点才允许拟合。
    std::size_t minPointsToFit;

    // ------------------------------------------------------------------------
    // 4. 平面约束
    // ------------------------------------------------------------------------
    // 是否使用固定平面法向。
    // 能量机关竖直面，建议 true。
    bool useFixedPlaneNormal;

    // 固定法向，世界系。
    // (0,0,1) 表示靶子绕世界 Z 轴转。
    cv::Vec3d planeNormal;

    // ------------------------------------------------------------------------
    // 5. 质量检测
    // ------------------------------------------------------------------------
    // 拟合残差上限（米），超过则本次无效。
    double maxRmsError;

    // 半径下限（米）。
    double minRadius;

    // 半径上限（米）。
    double maxRadius;

    // 最小角度覆盖（弧度）。
    // 覆盖太小圆心和半径病态，不该输出。
    double minCoverageRad;

    // ------------------------------------------------------------------------
    // 6. IRLS 鲁棒加权
    // ------------------------------------------------------------------------
    // IRLS 迭代次数。0 表示不做。
    int irlsIterations;

    // IRLS delta 系数：delta = 残差中位数 × 这个值。
    double irlsDeltaScale;

    // ------------------------------------------------------------------------
    // 7. Gauss-Newton 几何精修
    // ------------------------------------------------------------------------
    // GN 迭代次数。0 表示不做。
    int gnIterations;

    // GN 步长收敛阈值。
    double gnStepTolerance;

    // ------------------------------------------------------------------------
    // 8. RANSAC 初始拟合
    // ------------------------------------------------------------------------
    // 是否用 RANSAC 做初始拟合。
    bool useRansac;

    // RANSAC 最大迭代次数。
    int ransacIterations;

    // RANSAC 阈值下限（米）。
    // 实际阈值 = max(3σ, 这个值)。
    double ransacThreshold;

    // ------------------------------------------------------------------------
    // 9. unwrap 方式
    // ------------------------------------------------------------------------
    // true：预测式 unwrap，能处理相邻点转角 > π。
    bool usePredictedUnwrap;

    // 构造函数：给所有参数默认值。
    FittingCircleConfig();
};

// ============================================================================
// Observation：一次观测。
// ============================================================================
struct Observation
{
    cv::Point3d point;    // 世界坐标
    double timeSeconds;   // 时间戳
    bool valid;           // 是否有效

    Observation();
};

// ============================================================================
// CircleFitResult：圆拟合结果。
// ============================================================================
struct CircleFitResult
{
    bool valid;              // 是否成功
    cv::Point3d center;      // 圆心
    double radius;           // 半径
    cv::Vec3d planeNormal;   // 平面法向
    cv::Vec3d planeU;        // 平面 u 轴
    cv::Vec3d planeV;        // 平面 v 轴
    std::size_t numPoints;   // 总点数
    std::size_t numInliers;  // 内点数
    double inlierRatio;      // 内点比例
    double fitRmsError;      // 残差 RMS
    double coverageRad;      // 角度覆盖范围
    double confidence;       // 综合置信度 0~1

    CircleFitResult();
};

// ============================================================================
// PredictResult：预测结果。
// ============================================================================
struct PredictResult
{
    bool valid;                         // 是否成功
    cv::Point3d currentPoint;           // 当前拟合位置
    cv::Point3d predictedPoint;         // 预测位置
    double currentAngleRad;             // 当前角度
    double predictedAngleRad;           // 预测角度
    double angularVelocityRad;          // 融合角速度
    double observedAngularVelocityRad;  // 纯观测角速度
    double predictAheadSeconds;         // 本次预测时长
    double currentTimeSeconds;          // 本次预测时间

    PredictResult();
};

// ============================================================================
// FittingCircle：圆轨迹拟合与预测。
// ============================================================================
class FittingCircle
{
public:
    // 构造。
    explicit FittingCircle(const FittingCircleConfig& config = FittingCircleConfig());

    // 设置配置。
    void setConfig(const FittingCircleConfig& config);

    // 读配置。
    const FittingCircleConfig& config() const;
    FittingCircleConfig& config();

    // 添加一次观测。
    void addObservation(const cv::Point3d& worldPoint, double timeSeconds);

    // 清空历史。
    void reset();

    // 当前是否有有效圆。
    bool hasValidCircle() const;

    // 当前拟合结果。
    CircleFitResult currentFit() const;

    // 用配置的 predictAheadSeconds 预测。
    PredictResult predict(double currentTimeSeconds) const;

    // 用指定时长预测。
    PredictResult predict(double currentTimeSeconds, double predictAheadSeconds) const;

    // 查看历史。
    const std::deque<Observation>& history() const;

private:
    // 内部拟合结构。
    struct FitInternal
    {
        CircleFitResult fit;              // 拟合结果
        std::vector<double> angles;       // unwrap 后角度
        std::vector<double> times;        // 时间戳
        std::vector<cv::Point3d> points;  // 世界坐标
        bool valid;                       // 是否有效

        FitInternal();
    };

    // 从历史拟合圆。
    FitInternal fitFromHistory() const;

    // 丢弃过期点。
    void pruneHistory();

    // 计算点相对圆心的 wrapped 角度。
    double computeAngle(const cv::Point3d& p, const CircleFitResult& fit) const;

    // 把 wrapped 展开到最接近 ref 的角度。
    static double unwrapNear(double wrapped, double ref);

    // 求中位数（会修改输入）。
    static double computeMedian(std::vector<double>& v);

    // 普通 Kasa 拟合。
    bool fitCircle2D(const std::vector<double>& xs,
                     const std::vector<double>& ys,
                     double& cx, double& cy, double& r) const;

    // RANSAC 初始拟合。
    bool ransacCircle2D(const std::vector<double>& xs,
                        const std::vector<double>& ys,
                        double& cx, double& cy, double& r,
                        std::vector<int>& inliers) const;

    // IRLS 精修。
    void irlsCircle2D(const std::vector<double>& xs,
                      const std::vector<double>& ys,
                      double& cx, double& cy, double& r) const;

    // GN 精修。
    void gnCircle2D(const std::vector<double>& xs,
                    const std::vector<double>& ys,
                    double& cx, double& cy, double& r) const;

    // PCA 估法向。
    cv::Vec3d pcaNormal(const cv::Vec3d& centroid) const;

    // 加权回归估角速度。
    double estimateObservedAngularVelocity(const FitInternal& fit) const;

    // 加权融合。
    double fuseByWeight(double observed) const;

    // 卡尔曼融合。
    double kalmanAngularVelocity(const FitInternal& fit) const;

    FittingCircleConfig config_;        // 配置
    std::deque<Observation> history_;   // 历史点
};

#endif // FITTING_CIRCLE_H_