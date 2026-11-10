#include "FittingCircle.h"

#include <algorithm>
#include <cmath>
#include <random>

// ============================================================================
// 匿名命名空间：只在本文件可见的工具函数。
// ============================================================================
namespace
{
// 圆周率。
const double kPi = 3.14159265358979323846;
// 2π，unwrap 用。
const double kTwoPi = 2.0 * kPi;

// 三维点积。
double dot3(const cv::Vec3d& a, const cv::Vec3d& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// 三维长度。
double norm3(const cv::Vec3d& v)
{
    return std::sqrt(dot3(v, v));
}

// Point3d -> Vec3d。
cv::Vec3d toVec3d(const cv::Point3d& p)
{
    return cv::Vec3d(p.x, p.y, p.z);
}

// Vec3d -> Point3d。
cv::Point3d toPoint3d(const cv::Vec3d& v)
{
    return cv::Point3d(v[0], v[1], v[2]);
}

// 给定法向 n，构造平面正交基 u、v。
// u、v 都在平面上，n 是法向。
void buildPlaneBasis(const cv::Vec3d& n, cv::Vec3d& u, cv::Vec3d& v)
{
    // 归一化 n。
    cv::Vec3d nn = n;
    const double len = norm3(nn);
    if (len < 1e-12)
    {
        nn = cv::Vec3d(0.0, 0.0, 1.0);
    }
    else
    {
        nn /= len;
    }

    // 选一个和 nn 最不平行的坐标轴。
    cv::Vec3d axis(1.0, 0.0, 0.0);
    if (std::fabs(nn[0]) > 0.9)
    {
        axis = cv::Vec3d(0.0, 1.0, 0.0);
    }

    // u = n × axis，归一化。
    u = nn.cross(axis);
    if (norm3(u) < 1e-12)
    {
        axis = cv::Vec3d(0.0, 0.0, 1.0);
        u = nn.cross(axis);
    }
    u /= norm3(u);

    // v = n × u，归一化。
    v = nn.cross(u);
    v /= norm3(v);
}

// 固定种子随机数，方便调试复现。
std::mt19937& getRng()
{
    static std::mt19937 rng(12345);
    return rng;
}

// 三点定圆。
// 返回 false 表示三点共线或计算无效。
bool circleFrom3Points(double x1, double y1,
                       double x2, double y2,
                       double x3, double y3,
                       double& cx, double& cy, double& r)
{
    // 行列式，正比于三角形有向面积。
    // d = 0 表示三点共线，无外接圆。
    const double d = 2.0 * (x1 * (y2 - y3) +
                            x2 * (y3 - y1) +
                            x3 * (y1 - y2));
    if (std::fabs(d) < 1e-12)
    {
        return false;
    }

    // 三个点的模平方。
    const double x1sq = x1 * x1 + y1 * y1;
    const double x2sq = x2 * x2 + y2 * y2;
    const double x3sq = x3 * x3 + y3 * y3;

    // 克拉默法则解圆心。
    cx = (x1sq * (y2 - y3) +
          x2sq * (y3 - y1) +
          x3sq * (y1 - y2)) / d;

    cy = (x1sq * (x3 - x2) +
          x2sq * (x1 - x3) +
          x3sq * (x2 - x1)) / d;

    // 半径 = 圆心到第一点的距离。
    r = std::hypot(x1 - cx, y1 - cy);

    // 检查数值有效性。
    return std::isfinite(cx) && std::isfinite(cy) && std::isfinite(r);
}
} // namespace

// ============================================================================
// 配置默认值
// ============================================================================
FittingCircleConfig::FittingCircleConfig()
    : predictAheadSeconds(0.15),          // 预测 150 ms
      givenAngularVelocityRad(0.0),       // 无先验速度
      givenVelocityWeight(0.3),           // 偏信观测
      useKalman(false),                   // 先用加权融合
      kalmanProcessNoise(0.01),           // 过程噪声
      kalmanMeasureNoise(0.01),           // 观测噪声
      velocityWeightTau(0.4),             // 时间常数 0.4 s
      minAngularVelocityRad(-6.0),        // 角速度下限
      maxAngularVelocityRad(6.0),         // 角速度上限
      maxHistoryPoints(30),               // 最多 30 点
      maxObservationAgeSeconds(3.0),      // 点最大 3 s
      minPointsToFit(6),                  // 至少 6 点
      useFixedPlaneNormal(true),          // 固定法向
      planeNormal(0.0, 0.0, 1.0),         // 默认 Z 轴
      maxRmsError(0.05),                  // 残差上限 5 cm
      minRadius(0.05),                    // 半径下限 5 cm
      maxRadius(0.5),                     // 半径上限 50 cm
      minCoverageRad(1.0),                // 覆盖至少 57°
      irlsIterations(2),                  // IRLS 2 轮
      irlsDeltaScale(1.5),                // IRLS delta 系数
      gnIterations(5),                    // GN 5 轮
      gnStepTolerance(1e-6),              // GN 收敛阈值
      useRansac(true),                    // 开 RANSAC
      ransacIterations(100),              // RANSAC 100 次
      ransacThreshold(0.01),              // RANSAC 阈值下限 1 cm
      usePredictedUnwrap(true)            // 预测式 unwrap
{
}

// ============================================================================
// 各种结果结构体默认构造
// ============================================================================
Observation::Observation()
    : point(0.0, 0.0, 0.0), timeSeconds(0.0), valid(false)
{
}

CircleFitResult::CircleFitResult()
    : valid(false),
      center(0.0, 0.0, 0.0),
      radius(0.0),
      planeNormal(0.0, 0.0, 1.0),
      planeU(1.0, 0.0, 0.0),
      planeV(0.0, 1.0, 0.0),
      numPoints(0),
      numInliers(0),
      inlierRatio(0.0),
      fitRmsError(0.0),
      coverageRad(0.0),
      confidence(0.0)
{
}

PredictResult::PredictResult()
    : valid(false),
      currentPoint(0.0, 0.0, 0.0),
      predictedPoint(0.0, 0.0, 0.0),
      currentAngleRad(0.0),
      predictedAngleRad(0.0),
      angularVelocityRad(0.0),
      observedAngularVelocityRad(0.0),
      predictAheadSeconds(0.0),
      currentTimeSeconds(0.0)
{
}

// ============================================================================
// FitInternal 默认构造
// ============================================================================
FittingCircle::FitInternal::FitInternal()
    : valid(false)
{
}

// ============================================================================
// 构造 / 配置 / 历史管理
// ============================================================================
FittingCircle::FittingCircle(const FittingCircleConfig& config)
    : config_(config)
{
}

void FittingCircle::setConfig(const FittingCircleConfig& config)
{
    config_ = config;
}

const FittingCircleConfig& FittingCircle::config() const
{
    return config_;
}

FittingCircleConfig& FittingCircle::config()
{
    return config_;
}

void FittingCircle::reset()
{
    history_.clear();
}

const std::deque<Observation>& FittingCircle::history() const
{
    return history_;
}

// 添加一次观测。
void FittingCircle::addObservation(const cv::Point3d& worldPoint, double timeSeconds)
{
    // 检查输入是否有限。
    if (!std::isfinite(worldPoint.x) ||
        !std::isfinite(worldPoint.y) ||
        !std::isfinite(worldPoint.z) ||
        !std::isfinite(timeSeconds))
    {
        return;
    }

    // 构造并加入。
    Observation obs;
    obs.point = worldPoint;
    obs.timeSeconds = timeSeconds;
    obs.valid = true;
    history_.push_back(obs);

    // 清理过期点。
    pruneHistory();
}

// 丢弃过期点。
void FittingCircle::pruneHistory()
{
    // 数量上限。
    while (history_.size() > config_.maxHistoryPoints)
    {
        history_.pop_front();
    }

    // 年龄上限：以最后一个点的时间为参考。
    if (history_.empty())
    {
        return;
    }

    const double latestTime = history_.back().timeSeconds;
    while (!history_.empty() &&
           latestTime - history_.front().timeSeconds > config_.maxObservationAgeSeconds)
    {
        history_.pop_front();
    }
}

// ============================================================================
// 工具函数
// ============================================================================

// 把 wrapped 展开到最接近 ref 的角度。
// wrapped + n * 2π，n = round((ref - wrapped) / 2π)。
double FittingCircle::unwrapNear(double wrapped, double ref)
{
    const double n = std::round((ref - wrapped) / kTwoPi);
    return wrapped + n * kTwoPi;
}

// 求中位数（会修改输入 vector）。
double FittingCircle::computeMedian(std::vector<double>& v)
{
    if (v.empty())
    {
        return 0.0;
    }
    const std::size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    return v[mid];
}

// 计算点相对圆心的 wrapped 角度。
double FittingCircle::computeAngle(const cv::Point3d& p, const CircleFitResult& fit) const
{
    // 点到圆心的向量。
    const cv::Vec3d d = toVec3d(p) - toVec3d(fit.center);
    // 投影到平面基。
    const double x = dot3(d, fit.planeU);
    const double y = dot3(d, fit.planeV);
    // atan2 返回 [-π, π]。
    return std::atan2(y, x);
}

// ============================================================================
// 2D 圆拟合子函数
// ============================================================================

// 普通 Kasa 最小二乘拟合。
// 方程：x² + y² + Dx + Ey + F = 0。
// 对每点：Dxi + Eyi + F = -(xi² + yi²)。
bool FittingCircle::fitCircle2D(const std::vector<double>& xs,
                                const std::vector<double>& ys,
                                double& cx, double& cy, double& r) const
{
    const std::size_t n = xs.size();
    if (n < 3 || ys.size() != n)
    {
        return false;
    }

    // 构造 A、b。
    cv::Mat A(static_cast<int>(n), 3, CV_64F);
    cv::Mat b(static_cast<int>(n), 1, CV_64F);

    for (std::size_t i = 0; i < n; ++i)
    {
        A.at<double>(static_cast<int>(i), 0) = xs[i];
        A.at<double>(static_cast<int>(i), 1) = ys[i];
        A.at<double>(static_cast<int>(i), 2) = 1.0;
        b.at<double>(static_cast<int>(i), 0) = -(xs[i] * xs[i] + ys[i] * ys[i]);
    }

    // 最小二乘解 [D, E, F]。
    cv::Mat sol;
    if (!cv::solve(A, b, sol, cv::DECOMP_SVD))
    {
        return false;
    }

    const double D = sol.at<double>(0, 0);
    const double E = sol.at<double>(1, 0);
    const double F = sol.at<double>(2, 0);

    // 反解圆心和半径。
    cx = -0.5 * D;
    cy = -0.5 * E;
    const double rSq = cx * cx + cy * cy - F;

    if (rSq <= 0.0)
    {
        return false;
    }

    r = std::sqrt(rSq);
    return true;
}

// RANSAC 初始拟合。
// 1. 先普通拟合，用残差估计噪声，确定自适应阈值。
// 2. 随机抽 3 点定圆，统计内点。
// 3. 取内点最多的模型。
bool FittingCircle::ransacCircle2D(const std::vector<double>& xs,
                                   const std::vector<double>& ys,
                                   double& cx, double& cy, double& r,
                                   std::vector<int>& inliers) const
{
    const std::size_t n = xs.size();
    if (n < 5)
    {
        return false;
    }

    // 第一步：普通拟合，估计噪声。
    double cx0 = 0.0, cy0 = 0.0, r0 = 0.0;
    if (!fitCircle2D(xs, ys, cx0, cy0, r0))
    {
        return false;
    }

    // 残差 = |点到圆心的距离 - 半径|。
    std::vector<double> residuals(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        residuals[i] = std::fabs(std::hypot(xs[i] - cx0, ys[i] - cy0) - r0);
    }

    // 自适应阈值：max(3σ_mad, 固定下限)。
    // σ_mad = 1.4826 × MAD。
    std::vector<double> tmp = residuals;
    const double med = computeMedian(tmp);
    tmp = residuals;
    for (std::size_t i = 0; i < n; ++i)
    {
        tmp[i] = std::fabs(tmp[i] - med);
    }
    const double mad = computeMedian(tmp);
    const double sigma = 1.4826 * mad;
    double threshold = std::max(3.0 * sigma, config_.ransacThreshold);
    if (threshold < 1e-6)
    {
        threshold = std::max(1e-6, config_.ransacThreshold);
    }

    // 随机数。
    std::mt19937& rng = getRng();
    std::uniform_int_distribution<std::size_t> dist(0, n - 1);

    // 最优模型。
    int bestCount = -1;
    double bestCx = cx0;
    double bestCy = cy0;
    double bestR = r0;
    std::vector<int> bestInliers;

    // RANSAC 迭代。
    for (int iter = 0; iter < config_.ransacIterations; ++iter)
    {
        // 抽 3 个不同点。
        const std::size_t i1 = dist(rng);
        const std::size_t i2 = dist(rng);
        const std::size_t i3 = dist(rng);
        if (i1 == i2 || i2 == i3 || i1 == i3)
        {
            continue;
        }

        // 三点定圆。
        double tcx = 0.0, tcy = 0.0, tr = 0.0;
        if (!circleFrom3Points(xs[i1], ys[i1],
                               xs[i2], ys[i2],
                               xs[i3], ys[i3],
                               tcx, tcy, tr))
        {
            continue;
        }

        // 统计内点。
        std::vector<int> inl;
        inl.reserve(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            const double d = std::fabs(std::hypot(xs[i] - tcx, ys[i] - tcy) - tr);
            if (d < threshold)
            {
                inl.push_back(static_cast<int>(i));
            }
        }

        // 更新最优。
        if (static_cast<int>(inl.size()) > bestCount)
        {
            bestCount = static_cast<int>(inl.size());
            bestCx = tcx;
            bestCy = tcy;
            bestR = tr;
            bestInliers = inl;
        }
    }

    // RANSAC 失败就回退到普通拟合。
    if (bestCount < 3)
    {
        cx = cx0;
        cy = cy0;
        r = r0;
        inliers.clear();
        return true;
    }

    cx = bestCx;
    cy = bestCy;
    r = bestR;
    inliers = bestInliers;
    return true;
}

// IRLS 鲁棒加权精修。
// 残差大的点权重小，反复迭代。
void FittingCircle::irlsCircle2D(const std::vector<double>& xs,
                                 const std::vector<double>& ys,
                                 double& cx, double& cy, double& r) const
{
    const std::size_t n = xs.size();
    if (n < 3)
    {
        return;
    }

    const double scale = std::max(1.0, config_.irlsDeltaScale);

    for (int iter = 0; iter < config_.irlsIterations; ++iter)
    {
        // 1. 算当前残差。
        std::vector<double> residuals(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            residuals[i] = std::fabs(std::hypot(xs[i] - cx, ys[i] - cy) - r);
        }

        // 2. delta = 残差中位数 × scale。
        std::vector<double> tmp = residuals;
        const double med = computeMedian(tmp);
        const double delta = std::max(med * scale, 1e-6);

        // 3. 构造加权最小二乘：A 和 b 每行乘 sqrt(w_i)。
        cv::Mat A(static_cast<int>(n), 3, CV_64F);
        cv::Mat b(static_cast<int>(n), 1, CV_64F);

        for (std::size_t i = 0; i < n; ++i)
        {
            const double ratio = residuals[i] / delta;
            const double w = 1.0 / (1.0 + ratio * ratio);
            const double sw = std::sqrt(w);

            A.at<double>(static_cast<int>(i), 0) = xs[i] * sw;
            A.at<double>(static_cast<int>(i), 1) = ys[i] * sw;
            A.at<double>(static_cast<int>(i), 2) = sw;
            b.at<double>(static_cast<int>(i), 0) = -(xs[i] * xs[i] + ys[i] * ys[i]) * sw;
        }

        // 4. 解加权最小二乘。
        cv::Mat sol;
        if (!cv::solve(A, b, sol, cv::DECOMP_SVD))
        {
            break;
        }

        const double D = sol.at<double>(0, 0);
        const double E = sol.at<double>(1, 0);
        const double F = sol.at<double>(2, 0);

        const double newCx = -0.5 * D;
        const double newCy = -0.5 * E;
        const double newRsq = newCx * newCx + newCy * newCy - F;

        if (newRsq <= 0.0)
        {
            break;
        }

        const double newR = std::sqrt(newRsq);

        // 5. 检查变化量，太小就停。
        const double change =
            std::fabs(newCx - cx) + std::fabs(newCy - cy) + std::fabs(newR - r);

        cx = newCx;
        cy = newCy;
        r = newR;

        if (change < 1e-6)
        {
            break;
        }
    }
}

// GN 几何精修。
// 最小化 e_i = |P_i - C| - r。
// 雅可比 J_i = [-(xi-cx)/d_i, -(yi-cy)/d_i, -1]。
// 解 (J^T J) Δ = -J^T e。
void FittingCircle::gnCircle2D(const std::vector<double>& xs,
                               const std::vector<double>& ys,
                               double& cx, double& cy, double& r) const
{
    const std::size_t n = xs.size();
    if (n < 3)
    {
        return;
    }

    for (int iter = 0; iter < config_.gnIterations; ++iter)
    {
        // 累加 J^T J 和 J^T e。
        cv::Matx33d JtJ = cv::Matx33d::zeros();
        cv::Vec3d Jte(0.0, 0.0, 0.0);

        for (std::size_t i = 0; i < n; ++i)
        {
            const double dx = xs[i] - cx;
            const double dy = ys[i] - cy;
            const double d = std::hypot(dx, dy);

            if (d < 1e-9)
            {
                continue;
            }

            const double e = d - r;
            const cv::Vec3d J(-dx / d, -dy / d, -1.0);

            // JtJ += J * J^T。
            for (int a = 0; a < 3; ++a)
            {
                for (int b = 0; b < 3; ++b)
                {
                    JtJ(a, b) += J[a] * J[b];
                }
                Jte[a] += J[a] * e;
            }
        }

        // 检查奇异。
        const double det = cv::determinant(cv::Mat(JtJ));
        if (std::fabs(det) < 1e-12)
        {
            break;
        }

        // 解 Δ。
        cv::Vec3d delta = JtJ.inv() * cv::Vec3d(-Jte[0], -Jte[1], -Jte[2]);

        // 步长限幅，防止发散。
        const double stepLen = std::sqrt(delta[0] * delta[0] +
                                         delta[1] * delta[1] +
                                         delta[2] * delta[2]);
        if (stepLen > 0.5)
        {
            delta *= 0.5 / stepLen;
        }

        // 更新。
        cx += delta[0];
        cy += delta[1];
        r += delta[2];

        // 步长小就停。
        const double newStep = std::sqrt(delta[0] * delta[0] +
                                         delta[1] * delta[1] +
                                         delta[2] * delta[2]);
        if (newStep < config_.gnStepTolerance)
        {
            break;
        }
    }
}

// PCA 估平面法向。
cv::Vec3d FittingCircle::pcaNormal(const cv::Vec3d& centroid) const
{
    // 累加协方差。
    cv::Matx33d cov = cv::Matx33d::zeros();

    for (std::size_t i = 0; i < history_.size(); ++i)
    {
        const cv::Vec3d d = toVec3d(history_[i].point) - centroid;
        for (int r = 0; r < 3; ++r)
        {
            for (int c = 0; c < 3; ++c)
            {
                cov(r, c) += d[r] * d[c];
            }
        }
    }

    // SVD，最小奇异值方向是法向。
    cv::Mat w, u, vt;
    cv::SVD::compute(cv::Mat(cov), w, u, vt);

    return cv::Vec3d(vt.at<double>(2, 0),
                     vt.at<double>(2, 1),
                     vt.at<double>(2, 2));
}

// ============================================================================
// 核心：从历史拟合圆
// ============================================================================
FittingCircle::FitInternal FittingCircle::fitFromHistory() const
{
    FitInternal out;

    // 1. 点太少，失败。
    if (history_.size() < std::max<std::size_t>(3, config_.minPointsToFit))
    {
        return out;
    }

    const std::size_t n = history_.size();

    // 2. 求质心。
    cv::Vec3d centroid(0.0, 0.0, 0.0);
    for (std::size_t i = 0; i < n; ++i)
    {
        centroid += toVec3d(history_[i].point);
    }
    centroid /= static_cast<double>(n);

    // 3. 确定法向。
    cv::Vec3d normal = config_.planeNormal;
    if (!config_.useFixedPlaneNormal)
    {
        normal = pcaNormal(centroid);
    }

    const double nLen = norm3(normal);
    if (nLen < 1e-12)
    {
        return out;
    }
    normal /= nLen;

    // 4. 构造平面基。
    cv::Vec3d u, v;
    buildPlaneBasis(normal, u, v);

    // 5. 投影到 2D。
    std::vector<double> xs(n), ys(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        const cv::Vec3d d = toVec3d(history_[i].point) - centroid;
        xs[i] = dot3(d, u);
        ys[i] = dot3(d, v);
    }

    // 6. 坐标归一化：平移到质心，缩放到平均距离 1。
    //    解决大圆、大坐标时的数值病态。
    double mx = 0.0, my = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        mx += xs[i];
        my += ys[i];
    }
    mx /= static_cast<double>(n);
    my /= static_cast<double>(n);

    double meanR = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        meanR += std::hypot(xs[i] - mx, ys[i] - my);
    }
    meanR /= static_cast<double>(n);

    if (meanR < 1e-9)
    {
        return out;
    }

    std::vector<double> xsN(n), ysN(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        xsN[i] = (xs[i] - mx) / meanR;
        ysN[i] = (ys[i] - my) / meanR;
    }

    // 7. 初始拟合。
    double cxN = 0.0, cyN = 0.0, rN = 0.0;
    bool fitOk = false;

    if (config_.useRansac && n >= 5)
    {
        std::vector<int> inliers;
        fitOk = ransacCircle2D(xsN, ysN, cxN, cyN, rN, inliers);
    }

    if (!fitOk)
    {
        fitOk = fitCircle2D(xsN, ysN, cxN, cyN, rN);
    }

    if (!fitOk)
    {
        return out;
    }

    // 8. IRLS 精修。
    if (config_.irlsIterations > 0)
    {
        irlsCircle2D(xsN, ysN, cxN, cyN, rN);
    }

    // 9. GN 精修。
    if (config_.gnIterations > 0)
    {
        gnCircle2D(xsN, ysN, cxN, cyN, rN);
    }

    // 10. 反归一化。
    const double cx2 = mx + cxN * meanR;
    const double cy2 = my + cyN * meanR;
    const double r2 = rN * meanR;

    // 11. 反投影回 3D。
    const cv::Vec3d center3d = centroid + cx2 * u + cy2 * v;

    // 12. 算残差。
    std::vector<double> residuals(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        residuals[i] = std::hypot(xs[i] - cx2, ys[i] - cy2) - r2;
    }

    double rms = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        rms += residuals[i] * residuals[i];
    }
    rms = std::sqrt(rms / static_cast<double>(n));

    // 13. 算覆盖率：角度排序，找最大间隔。
    std::vector<double> anglesWrapped(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        anglesWrapped[i] = std::atan2(ys[i] - cy2, xs[i] - cx2);
    }

    std::vector<double> sorted = anglesWrapped;
    std::sort(sorted.begin(), sorted.end());

    double maxGap = 0.0;
    for (std::size_t i = 1; i < n; ++i)
    {
        maxGap = std::max(maxGap, sorted[i] - sorted[i - 1]);
    }
    maxGap = std::max(maxGap, sorted[0] + kTwoPi - sorted.back());

    const double coverage = kTwoPi - maxGap;

    // 14. 算内点率。
    const double inlierThreshold = std::max(3.0 * rms, config_.maxRmsError * 0.5);
    std::size_t inlierCount = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
        if (std::fabs(residuals[i]) < inlierThreshold)
        {
            ++inlierCount;
        }
    }
    const double inlierRatio = static_cast<double>(inlierCount) / static_cast<double>(n);

    // 15. 算置信度：覆盖率 × 残差得分 × 内点率。
    const double coverageScore = std::min(1.0, coverage / kPi);
    const double rmsScore = std::exp(-rms / (0.5 * config_.maxRmsError + 1e-9));
    const double confidence = coverageScore * rmsScore * inlierRatio;

    // 16. 质量检查。
    const bool valid =
        (rms < config_.maxRmsError) &&
        (r2 >= config_.minRadius && r2 <= config_.maxRadius) &&
        (coverage >= config_.minCoverageRad);

    if (!valid)
    {
        return out;
    }

    // 17. 填结果。
    CircleFitResult fit;
    fit.valid = true;
    fit.center = toPoint3d(center3d);
    fit.radius = r2;
    fit.planeNormal = normal;
    fit.planeU = u;
    fit.planeV = v;
    fit.numPoints = n;
    fit.numInliers = inlierCount;
    fit.inlierRatio = inlierRatio;
    fit.fitRmsError = rms;
    fit.coverageRad = coverage;
    fit.confidence = confidence;

    out.fit = fit;
    out.valid = true;

    // 18. 角度 unwrap。
    out.times.resize(n);
    out.points.resize(n);
    out.angles.resize(n);

    double prevUnwrapped = anglesWrapped[0];
    double omegaGuess = config_.givenAngularVelocityRad;

    out.times[0] = history_[0].timeSeconds;
    out.points[0] = history_[0].point;
    out.angles[0] = prevUnwrapped;

    for (std::size_t i = 1; i < n; ++i)
    {
        const double dt = history_[i].timeSeconds - history_[i - 1].timeSeconds;
        const double wrapped = anglesWrapped[i];

        double unwrapped;
        if (config_.usePredictedUnwrap)
        {
            // 预测式：ref = prevUnwrapped + omegaGuess * dt。
            const double expected = prevUnwrapped + omegaGuess * dt;
            unwrapped = unwrapNear(wrapped, expected);
        }
        else
        {
            // 普通相邻差 unwrap。
            unwrapped = wrapped;
            if (unwrapped < prevUnwrapped - kPi)
            {
                unwrapped += kTwoPi;
            }
            else if (unwrapped > prevUnwrapped + kPi)
            {
                unwrapped -= kTwoPi;
            }
        }

        out.times[i] = history_[i].timeSeconds;
        out.points[i] = history_[i].point;
        out.angles[i] = unwrapped;

        // 用当前差分更新 omegaGuess，指数平滑。
        if (dt > 1e-6)
        {
            double instantOmega = (unwrapped - prevUnwrapped) / dt;

            if (instantOmega > config_.maxAngularVelocityRad)
            {
                instantOmega = config_.maxAngularVelocityRad;
            }
            if (instantOmega < config_.minAngularVelocityRad)
            {
                instantOmega = config_.minAngularVelocityRad;
            }

            omegaGuess = 0.7 * omegaGuess + 0.3 * instantOmega;
        }

        prevUnwrapped = unwrapped;
    }

    return out;
}

bool FittingCircle::hasValidCircle() const
{
    return fitFromHistory().valid;
}

CircleFitResult FittingCircle::currentFit() const
{
    return fitFromHistory().fit;
}

// ============================================================================
// 角速度估计
// ============================================================================

// 加权线性回归估计观测角速度。
// 权重 w_i = exp(-(t_last - t_i) / tau)，越近权重越大。
double FittingCircle::estimateObservedAngularVelocity(const FitInternal& fit) const
{
    const std::size_t n = fit.times.size();
    if (n < 2)
    {
        return 0.0;
    }

    const double tau = std::max(0.01, config_.velocityWeightTau);
    const double tLast = fit.times.back();

    // 1. 算权重。
    std::vector<double> weights(n);
    double wSum = 0.0;
    double tMean = 0.0;
    double aMean = 0.0;

    for (std::size_t i = 0; i < n; ++i)
    {
        weights[i] = std::exp(-(tLast - fit.times[i]) / tau);
        wSum += weights[i];
        tMean += weights[i] * fit.times[i];
        aMean += weights[i] * fit.angles[i];
    }

    if (wSum < 1e-12)
    {
        return 0.0;
    }

    tMean /= wSum;
    aMean /= wSum;

    // 2. 加权线性回归：ω = Σw(t-t̄)(θ-θ̄) / Σw(t-t̄)²。
    double num = 0.0;
    double den = 0.0;

    for (std::size_t i = 0; i < n; ++i)
    {
        const double dt = fit.times[i] - tMean;
        const double da = fit.angles[i] - aMean;
        num += weights[i] * dt * da;
        den += weights[i] * dt * dt;
    }

    if (den < 1e-12)
    {
        return 0.0;
    }

    double omega = num / den;

    // 3. 鲁棒化：剔除离群点，重算一次。
    if (n >= 4)
    {
        // 算残差。
        std::vector<double> residuals(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            residuals[i] = fit.angles[i] - (aMean + omega * (fit.times[i] - tMean));
        }

        // MAD 阈值。
        std::vector<double> tmp = residuals;
        const double med = computeMedian(tmp);
        tmp = residuals;
        for (std::size_t i = 0; i < n; ++i)
        {
            tmp[i] = std::fabs(tmp[i] - med);
        }
        const double mad = computeMedian(tmp);
        const double threshold = std::max(3.0 * 1.4826 * mad, 1e-3);

        // 离群点权重清零。
        double wSum2 = 0.0;
        double tMean2 = 0.0;
        double aMean2 = 0.0;

        for (std::size_t i = 0; i < n; ++i)
        {
            if (std::fabs(residuals[i] - med) > threshold)
            {
                weights[i] = 0.0;
            }
            wSum2 += weights[i];
            tMean2 += weights[i] * fit.times[i];
            aMean2 += weights[i] * fit.angles[i];
        }

        // 重算。
        if (wSum2 > 1e-9)
        {
            tMean2 /= wSum2;
            aMean2 /= wSum2;

            double num2 = 0.0;
            double den2 = 0.0;

            for (std::size_t i = 0; i < n; ++i)
            {
                const double dt = fit.times[i] - tMean2;
                const double da = fit.angles[i] - aMean2;
                num2 += weights[i] * dt * da;
                den2 += weights[i] * dt * dt;
            }

            if (den2 > 1e-12)
            {
                omega = num2 / den2;
            }
        }
    }

    return omega;
}

// 加权融合。
// 没给先验速度时，完全信观测。
double FittingCircle::fuseByWeight(double observed) const
{
    if (std::fabs(config_.givenAngularVelocityRad) < 1e-9)
    {
        return observed;
    }

    const double w = std::max(0.0, std::min(1.0, config_.givenVelocityWeight));
    return w * config_.givenAngularVelocityRad + (1.0 - w) * observed;
}

// 一维卡尔曼：状态 [θ, ω]。
// 过程：θ' = θ + ω*dt。
// 观测：θ。
double FittingCircle::kalmanAngularVelocity(const FitInternal& fit) const
{
    const std::size_t n = fit.times.size();
    if (n < 2)
    {
        return config_.givenAngularVelocityRad;
    }

    // 初始状态。
    double theta = fit.angles[0];
    double omega = config_.givenAngularVelocityRad;

    // 初始协方差。
    double P00 = kPi * kPi;
    double P01 = 0.0;
    double P10 = 0.0;
    double P11 = 100.0;

    const double q = config_.kalmanProcessNoise;
    const double r = config_.kalmanMeasureNoise;

    // 逐点更新。
    for (std::size_t i = 1; i < n; ++i)
    {
        const double dt = fit.times[i] - fit.times[i - 1];
        if (dt <= 0.0)
        {
            continue;
        }

        // 预测。
        theta += omega * dt;

        // P = F P F^T + Q。
        const double p00New = P00 + dt * (P10 + P01) + dt * dt * P11 + q;
        const double p01New = P01 + dt * P11;
        const double p10New = P10 + dt * P11;
        const double p11New = P11 + q;

        P00 = p00New;
        P01 = p01New;
        P10 = p10New;
        P11 = p11New;

        // 更新。
        const double y = fit.angles[i] - theta;
        const double S = P00 + r;
        if (S < 1e-12)
        {
            continue;
        }

        const double K0 = P00 / S;
        const double K1 = P10 / S;

        theta += K0 * y;
        omega += K1 * y;

        // P = (I - K H) P。
        const double p00u = (1.0 - K0) * P00;
        const double p01u = (1.0 - K0) * P01;
        const double p10u = P10 - K1 * P00;
        const double p11u = P11 - K1 * P01;

        P00 = p00u;
        P01 = p01u;
        P10 = p10u;
        P11 = p11u;
    }

    return omega;
}

// ============================================================================
// 预测
// ============================================================================
PredictResult FittingCircle::predict(double currentTimeSeconds) const
{
    return predict(currentTimeSeconds, config_.predictAheadSeconds);
}

PredictResult FittingCircle::predict(double currentTimeSeconds,
                                     double predictAheadSeconds) const
{
    PredictResult out;
    out.currentTimeSeconds = currentTimeSeconds;
    out.predictAheadSeconds = predictAheadSeconds;

    // 1. 拟合。
    const FitInternal fit = fitFromHistory();
    if (!fit.valid || fit.angles.empty())
    {
        return out;
    }

    // 2. 观测角速度。
    const double observed = estimateObservedAngularVelocity(fit);
    out.observedAngularVelocityRad = observed;

    // 3. 融合。
    double omega = config_.useKalman
        ? kalmanAngularVelocity(fit)
        : fuseByWeight(observed);

    // 4. 限幅。
    if (config_.maxAngularVelocityRad > config_.minAngularVelocityRad)
    {
        omega = std::max(config_.minAngularVelocityRad,
                         std::min(config_.maxAngularVelocityRad, omega));
    }

    out.angularVelocityRad = omega;

    // 5. 外推到当前时间。
    const double lastTime = fit.times.back();
    const double lastAngle = fit.angles.back();
    double dtNow = currentTimeSeconds - lastTime;

    // 太久没更新就不外推。
    if (dtNow > config_.maxObservationAgeSeconds)
    {
        dtNow = 0.0;
    }

    const double angleAtNow = lastAngle + omega * dtNow;

    out.currentAngleRad = angleAtNow;
    out.predictedAngleRad = angleAtNow + omega * predictAheadSeconds;

    // 6. 由角度反算 3D 位置。
    const cv::Vec3d center = toVec3d(fit.fit.center);
    const cv::Vec3d u = fit.fit.planeU;
    const cv::Vec3d v = fit.fit.planeV;
    const double radius = fit.fit.radius;

    const cv::Vec3d cur = center + radius * (std::cos(out.currentAngleRad) * u +
                                             std::sin(out.currentAngleRad) * v);
    const cv::Vec3d pred = center + radius * (std::cos(out.predictedAngleRad) * u +
                                              std::sin(out.predictedAngleRad) * v);

    out.currentPoint = toPoint3d(cur);
    out.predictedPoint = toPoint3d(pred);
    out.valid = true;

    return out;
}