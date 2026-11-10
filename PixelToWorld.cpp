#include "PixelToWorld.h"

#include <cmath>

// 匿名命名空间：这里的函数只在当前 .cpp 内部使用，不暴露给外部。
namespace
{
// 三维向量点积：a · b = ax*bx + ay*by + az*bz。
double dot3(const cv::Vec3d& a, const cv::Vec3d& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// 三维向量长度：||v|| = sqrt(v · v)。
double norm3(const cv::Vec3d& v)
{
    return std::sqrt(dot3(v, v));
}

// cv::Vec3d 转 cv::Point3d，方便对外返回。
cv::Point3d toPoint3d(const cv::Vec3d& v)
{
    return cv::Point3d(v[0], v[1], v[2]);
}
} // namespace

// ============================================================================
// PixelToWorldConfig 默认构造：给一组能直接跑的默认值。
// ============================================================================
PixelToWorldConfig::PixelToWorldConfig()
    : fx(1300.0),           // 默认焦距 x，先按你程序里的 kFocalPx
      fy(1300.0),           // 默认焦距 y
      cx(640.0),            // 默认光心 x，1280 宽度的一半
      cy(360.0),            // 默认光心 y，720 高度的一半
      // 默认 T_cg：R = diag(1,-1,1)，t = 0。
      // 每一行是 [R 的一行, t 的一个分量]。
      T_camera_to_gimbal(
          1.0, 0.0, 0.0, 0.0,
          0.0, -1.0, 0.0, 0.0,
          0.0, 0.0, 1.0, 0.0,
          0.0, 0.0, 0.0, 1.0),
      gimbal_origin_in_world(0.0, 0.0, 0.0),        // G 原点暂时放在世界原点
      target_plane_normal_world(0.0, 0.0, 1.0),     // 目标平面法向量 n = (0,0,1)
      target_plane_offset(2.0),                     // 平面 Z = 2 m
      use_plane_point(false),                       // 默认用 n·X=d，不用平面点
      target_plane_point_world(0.0, 0.0, 2.0),      // 备用平面点
      yaw_offset_deg(0.0),                          // yaw 零偏
      pitch_offset_deg(0.0),                        // pitch 零偏
      yaw_sign(1.0),                                // yaw 正方向
      pitch_sign(1.0),                              // pitch 正方向
      angles_in_degrees(true),                      // 输入角度默认是度
      require_intersection_in_front(true)           // 只接受相机前方的交点
{
}

// ============================================================================
// BackProjectionResult 默认构造：先标记为无效。
// ============================================================================
BackProjectionResult::BackProjectionResult()
    : valid(false),
      world_point(0.0, 0.0, 0.0),
      camera_origin_world(0.0, 0.0, 0.0),
      ray_direction_world(0.0, 0.0, 0.0),
      plane_parameter(-1.0)
{
}

// ============================================================================
// PixelToWorld 构造 / 配置
// ============================================================================
PixelToWorld::PixelToWorld(const PixelToWorldConfig& config)
    : config_(config)
{
}

void PixelToWorld::setConfig(const PixelToWorldConfig& config)
{
    config_ = config;
}

const PixelToWorldConfig& PixelToWorld::config() const
{
    return config_;
}

PixelToWorldConfig& PixelToWorld::config()
{
    return config_;
}

// 角度 -> 弧度。如果 angles_in_degrees 为 true，就乘 pi/180。
double PixelToWorld::angleToRad(double angle) const
{
    const double kPi = 3.14159265358979323846;
    return config_.angles_in_degrees ? angle * kPi / 180.0 : angle;
}

// ============================================================================
// 基本旋转矩阵
// ============================================================================
// 绕 X 轴旋转：
//   Rx = [1   0    0  ]
//        [0  cos  -sin]
//        [0  sin   cos]
cv::Matx33d PixelToWorld::rotationX(double rad)
{
    const double c = std::cos(rad);
    const double s = std::sin(rad);
    return cv::Matx33d(1.0, 0.0, 0.0,
                       0.0,   c,  -s,
                       0.0,   s,   c);
}

// 绕 Y 轴旋转：
//   Ry = [ cos  0  sin]
//        [  0   1   0 ]
//        [-sin  0  cos]
cv::Matx33d PixelToWorld::rotationY(double rad)
{
    const double c = std::cos(rad);
    const double s = std::sin(rad);
    return cv::Matx33d( c, 0.0,   s,
                        0.0, 1.0, 0.0,
                       -s, 0.0,   c);
}

// 绕 Z 轴旋转：
//   Rz = [cos  -sin  0]
//        [sin   cos  0]
//        [ 0     0   1]
cv::Matx33d PixelToWorld::rotationZ(double rad)
{
    const double c = std::cos(rad);
    const double s = std::sin(rad);
    return cv::Matx33d( c,  -s, 0.0,
                        s,   c, 0.0,
                        0.0, 0.0, 1.0);
}

// ============================================================================
// 齐次变换工具
// ============================================================================

// 组装 T = [R t; 0 1]。
// 注意：最后一行固定是 0 0 0 1。
cv::Matx44d PixelToWorld::makeRigidTransform(const cv::Matx33d& R, const cv::Vec3d& t)
{
    return cv::Matx44d(
        R(0,0), R(0,1), R(0,2), t[0],
        R(1,0), R(1,1), R(1,2), t[1],
        R(2,0), R(2,1), R(2,2), t[2],
        0.0,    0.0,    0.0,    1.0);
}

// 刚体变换求逆：
//   T = [R t; 0 1]
//   T^{-1} = [R^T  -R^T t; 0 1]
// 这里利用 R 是正交矩阵：R^{-1} = R^T。
cv::Matx44d PixelToWorld::invertRigidTransform(const cv::Matx44d& T)
{
    // 从 4x4 中取出旋转部分 R。
    const cv::Matx33d R(
        T(0,0), T(0,1), T(0,2),
        T(1,0), T(1,1), T(1,2),
        T(2,0), T(2,1), T(2,2));

    // 从 4x4 中取出平移部分 t。
    const cv::Vec3d t(T(0,3), T(1,3), T(2,3));

    // R 的转置就是 R 的逆。
    const cv::Matx33d Rt = R.t();

    // 逆变换的平移：t_inv = -R^T * t。
    const cv::Vec3d tInv = -Rt * t;

    // 重新组装成 4x4。
    return makeRigidTransform(Rt, tInv);
}

// 点变换：P' = T * [P; 1]。
// 为什么最后一维是 1？因为平移 t 要作用在点上。
cv::Vec3d PixelToWorld::transformPoint(const cv::Matx44d& T, const cv::Vec3d& p)
{
    // 升到齐次坐标 [Px, Py, Pz, 1]。
    const cv::Vec4d ph(p[0], p[1], p[2], 1.0);

    // 4x4 * 4x1 = 4x1。
    const cv::Vec4d r = T * ph;

    // 取前三维就是变换后的点。
    return cv::Vec3d(r[0], r[1], r[2]);
}

// 方向变换：D' = T * [D; 0]。
// 为什么最后一维是 0？因为方向不应该被平移影响。
cv::Vec3d PixelToWorld::transformDirection(const cv::Matx44d& T, const cv::Vec3d& d)
{
    // 升到齐次方向 [Dx, Dy, Dz, 0]。
    const cv::Vec4d dh(d[0], d[1], d[2], 0.0);

    const cv::Vec4d r = T * dh;

    return cv::Vec3d(r[0], r[1], r[2]);
}

// ============================================================================
// 云台 G -> 世界 W 的旋转
// ============================================================================
cv::Matx33d PixelToWorld::gimbalToWorldRotation(double pitch, double yaw) const
{
    const double kPi = 3.14159265358979323846;

    // 加零偏、乘符号修正，再转弧度。
    const double yawRad =
        config_.yaw_sign *
        (angleToRad(yaw) + config_.yaw_offset_deg * kPi / 180.0);

    const double pitchRad =
        config_.pitch_sign *
        (angleToRad(pitch) + config_.pitch_offset_deg * kPi / 180.0);

    // 先绕 X 轴 pitch，再绕 Y 轴 yaw。
    // 矩阵乘法从右往左作用：
    //   R_gw * P_G = Ry(yaw) * (Rx(pitch) * P_G)
    // 即先对点做 pitch，再做 yaw。
    return rotationY(yawRad) * rotationX(pitchRad);
}

// G -> W 的完整齐次变换：T_gw = [R_gw  O_gw; 0 1]。
cv::Matx44d PixelToWorld::gimbalToWorldTransform(double pitch, double yaw) const
{
    const cv::Matx33d R_gw = gimbalToWorldRotation(pitch, yaw);
    return makeRigidTransform(R_gw, config_.gimbal_origin_in_world);
}

// C -> W：先 C -> G，再 G -> W。
// 矩阵乘法顺序：
//   P_W = T_gw * P_G = T_gw * (T_cg * P_C) = (T_gw * T_cg) * P_C
// 所以 T_cw = T_gw * T_cg。
cv::Matx44d PixelToWorld::totalTransform(double pitch, double yaw) const
{
    return gimbalToWorldTransform(pitch, yaw) * config_.T_camera_to_gimbal;
}

// ============================================================================
// 像素 -> 相机射线
// ============================================================================
// 针孔模型反推：
//   Xc/Zc = (Px - cx) / fx
//   Yc/Zc = (Py - cy) / fy
// 令 Zc = 1，得到射线方向 D_c = ((Px-cx)/fx, (Py-cy)/fy, 1)。
cv::Vec3d PixelToWorld::pixelRayCamera(double Px, double Py) const
{
    // 防止 fx/fy 为 0 导致除零。
    if (std::fabs(config_.fx) < 1e-12 || std::fabs(config_.fy) < 1e-12)
    {
        return cv::Vec3d(0.0, 0.0, 1.0);
    }

    return cv::Vec3d(
        (Px - config_.cx) / config_.fx,
        (Py - config_.cy) / config_.fy,
        1.0);
}

// ============================================================================
// 核心：反投影
// ============================================================================
BackProjectionResult PixelToWorld::BackProjection(double Px,
                                                  double Py,
                                                  double pitch,
                                                  double yaw) const
{
    BackProjectionResult result;

    // 输入必须是有限数，否则直接失败。
    if (!std::isfinite(Px) || !std::isfinite(Py) ||
        !std::isfinite(pitch) || !std::isfinite(yaw))
    {
        return result;
    }

    // ------------------------------------------------------------------------
    // 第 1 步：像素 -> 相机射线
    // ------------------------------------------------------------------------
    // 得到相机坐标系下的方向 D_c。
    const cv::Vec3d d_camera = pixelRayCamera(Px, Py);

    // ------------------------------------------------------------------------
    // 第 2 步：相机射线 -> 世界射线
    // ------------------------------------------------------------------------
    // T_cw 把相机坐标系下的点/方向变换到世界坐标系。
    const cv::Matx44d T_cw = totalTransform(pitch, yaw);

    // 方向向量最后一维为 0，只受旋转影响，不受平移影响。
    const cv::Vec3d d_world = transformDirection(T_cw, d_camera);

    // 相机光心在相机坐标系是原点 (0,0,0)。
    // 用 transformPoint 升到齐次坐标 [0,0,0,1]，得到它在世界系的位置。
    const cv::Vec3d camera_origin_world =
        transformPoint(T_cw, cv::Vec3d(0.0, 0.0, 0.0));

    // ------------------------------------------------------------------------
    // 第 3 步：世界射线与目标平面求交
    // ------------------------------------------------------------------------
    // 平面方程：n · X = d。
    // 先把 n 归一化，避免 d 的量纲不一致。
    cv::Vec3d n = config_.target_plane_normal_world;
    const double n_norm = norm3(n);
    if (n_norm < 1e-12)
    {
        return result; // 法向量不能为零
    }
    n /= n_norm;

    // 计算平面常数 d。
    double plane_d = 0.0;
    if (config_.use_plane_point)
    {
        // 用平面点 P0：n · (X - P0) = 0  =>  n · X = n · P0。
        const cv::Vec3d p0(
            config_.target_plane_point_world.x,
            config_.target_plane_point_world.y,
            config_.target_plane_point_world.z);
        plane_d = dot3(n, p0);
    }
    else
    {
        // 原始平面 n_orig · X = d_orig。
        // n 归一化后变成 n = n_orig / ||n_orig||，
        // 所以 d 也要除以 ||n_orig||。
        plane_d = config_.target_plane_offset / n_norm;
    }

    // 射线：P(t) = O + t * D。
    // 代入平面：n · (O + tD) = d  =>  t = (d - n·O) / (n·D)。
    const double denom = dot3(n, d_world);
    if (std::fabs(denom) < 1e-12)
    {
        // 射线与平面平行，无交点。
        return result;
    }

    const double t = (plane_d - dot3(n, camera_origin_world)) / denom;

    // 如果要求交点在相机前方，t 必须大于 0。
    if (config_.require_intersection_in_front && t <= 0.0)
    {
        return result;
    }

    // 交点的世界坐标。
    const cv::Vec3d world = camera_origin_world + t * d_world;

    // 最后检查一次数值有效性。
    if (!std::isfinite(world[0]) ||
        !std::isfinite(world[1]) ||
        !std::isfinite(world[2]))
    {
        return result;
    }

    // ------------------------------------------------------------------------
    // 第 4 步：填结果
    // ------------------------------------------------------------------------
    result.valid = true;
    result.world_point = toPoint3d(world);
    result.camera_origin_world = toPoint3d(camera_origin_world);
    result.ray_direction_world = toPoint3d(d_world);
    result.plane_parameter = t;
    return result;
}

// 重载：只返回世界坐标。
bool PixelToWorld::BackProjection(double Px,
                                  double Py,
                                  double pitch,
                                  double yaw,
                                  cv::Point3d& worldPoint) const
{
    const BackProjectionResult result = BackProjection(Px, Py, pitch, yaw);
    if (!result.valid)
    {
        return false;
    }
    worldPoint = result.world_point;
    return true;
}