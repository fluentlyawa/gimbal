#ifndef PIXEL_TO_WORLD_H_
#define PIXEL_TO_WORLD_H_

// OpenCV 核心头文件：提供 cv::Matx33d、cv::Matx44d、cv::Vec3d、cv::Point3d 等。
#include <opencv2/core.hpp>

// ============================================================================
// PixelToWorldConfig：所有“可调参数”集中在这里。
// 你可以把它理解成“标定表 + 坐标系约定表”。
// 改参数只改这个结构体，不用动算法代码。
// ============================================================================
struct PixelToWorldConfig
{
    // ------------------------------------------------------------------------
    // 1. 相机内参
    // ------------------------------------------------------------------------
    // 针孔相机模型：
    //   u = fx * (Xc / Zc) + cx
    //   v = fy * (Yc / Zc) + cy
    // 反过来，已知像素 (u,v)，可以求相机坐标系下的一条射线：
    //   Xc = (u - cx) / fx
    //   Yc = (v - cy) / fy
    //   Zc = 1
    // 所以 fx、fy 是焦距（单位：像素），cx、cy 是光心（单位：像素）。
    double fx;
    double fy;
    double cx;
    double cy;

    // ------------------------------------------------------------------------
    // 2. 相机 C -> 云台 G 的齐次变换
    // ------------------------------------------------------------------------
    // 数学表达：
    //   [P_G; 1] = T_camera_to_gimbal * [P_C; 1]
    // 其中 T = [R t; 0 1]，R 是 3x3 旋转，t 是 3x1 平移。
    //
    // 为什么用 4x4？
    //   因为平移不能直接用 3x3 矩阵乘点表示，
    //   升到齐次坐标后，旋转和平移可以合并成一次矩阵乘法。
    //
    // 默认值 R = diag(1, -1, 1)：
    //   OpenCV 相机坐标系：x 向右，y 向下，z 向前。
    //   云台 G 坐标系：x 向右，y 向上，z 向前。
    //   两者 y 轴相反，所以 R 的 y 分量取 -1。
    cv::Matx44d T_camera_to_gimbal;

    // ------------------------------------------------------------------------
    // 3. G 原点在 World 坐标系中的固定位置
    // ------------------------------------------------------------------------
    // G 坐标系原点（pitch/yaw 轴交点）在世界系里是不动的。
    // 数学表达：
    //   P_W = R_gw(pitch,yaw) * P_G + O_gw
    // 其中 O_gw 就是 gimbal_origin_in_world。
    cv::Vec3d gimbal_origin_in_world;

    // ------------------------------------------------------------------------
    // 4. 目标平面：n · X = d
    // ------------------------------------------------------------------------
    // 单目相机没有深度，必须给一个几何约束。
    // 你这里“炮台到能量机关距离 2 m”不是相机光轴深度，
    // 而是目标所在平面到世界原点的距离。
    //
    // 平面方程：
    //   n · X = d
    // 例如 n = (0,0,1)，d = 2.0，表示世界系下 Z = 2 m 的竖直平面。
    //
    // 射线与平面求交：
    //   P = O + t * D
    //   n · (O + t * D) = d
    //   t = (d - n·O) / (n·D)
    // 求出 t 后，P 就是唯一的世界坐标。
    cv::Vec3d target_plane_normal_world;
    double target_plane_offset;

    // 如果你更想用“平面上一个点 P0”来定义平面，就把 use_plane_point 设为 true。
    // 此时平面方程变成：
    //   n · (X - P0) = 0
    // 等价于：
    //   n · X = n · P0
    bool use_plane_point;
    cv::Vec3d target_plane_point_world;

    // ------------------------------------------------------------------------
    // 5. 角度零偏和正方向修正
    // ------------------------------------------------------------------------
    // 下位机反馈的 pitch/yaw 可能和几何定义有零偏。
    // 例如机械零位不是水平，或者正方向相反。
    //
    // 实际参与计算的角度：
    //   effectiveYaw   = yaw_sign   * (yaw   + yaw_offset_deg)
    //   effectivePitch = pitch_sign * (pitch + pitch_offset_deg)
    //
    // 单位固定为 deg，内部再转 rad。
    double yaw_offset_deg;
    double pitch_offset_deg;
    double yaw_sign;
    double pitch_sign;

    // 输入 pitch/yaw 是否为角度制。默认 true。
    bool angles_in_degrees;

    // 是否要求射线与平面交点在相机前方 t > 0。
    // 如果交点在相机背后，通常说明平面参数或角度符号错了。
    bool require_intersection_in_front;

    // 构造函数：给所有参数一个默认值，方便先跑通再标定。
    PixelToWorldConfig();
};

// ============================================================================
// BackProjectionResult：反投影结果。
// 不只返回世界坐标，还返回射线信息，方便调试“为什么算错了”。
// ============================================================================
struct BackProjectionResult
{
    bool valid;                       // 本次反投影是否成功
    cv::Point3d world_point;          // 射线与目标平面的交点，即世界坐标
    cv::Point3d camera_origin_world;  // 相机光心在世界系中的位置
    cv::Point3d ray_direction_world;  // 世界系下的射线方向（未归一化）
    double plane_parameter;           // P = O + t * D 中的 t

    BackProjectionResult();
};

// ============================================================================
// PixelToWorld：核心类。
// 对外只需要一个 BackProjection(Px, Py, pitch, yaw)。
// 内部按 Pixel -> Camera -> Gimbal -> World 的齐次变换链计算。
// ============================================================================
class PixelToWorld
{
public:
    // 构造时可以传入配置；不传就用默认配置。
    explicit PixelToWorld(const PixelToWorldConfig& config = PixelToWorldConfig());

    // 修改配置。
    void setConfig(const PixelToWorldConfig& config);

    // 读取配置（const 版本供只读使用，非 const 版本供临时修改字段）。
    const PixelToWorldConfig& config() const;
    PixelToWorldConfig& config();

    // ------------------------------------------------------------------------
    // 主函数：像素 + pitch + yaw -> 世界坐标
    // ------------------------------------------------------------------------
    // 输入：
    //   Px, Py : 像素坐标，单位 pixel
    //   pitch  : 云台俯仰角
    //   yaw    : 云台偏航角
    // 输出：
    //   BackProjectionResult，包含 valid、world_point、射线信息。
    //
    // 内部步骤：
    //   1) 像素 -> 相机射线
    //   2) 相机射线 -> 世界射线（齐次变换）
    //   3) 世界射线与目标平面求交
    //   4) 返回交点
    BackProjectionResult BackProjection(double Px,
                                        double Py,
                                        double pitch,
                                        double yaw) const;

    // 重载：只关心世界坐标时用这个，成功返回 true。
    bool BackProjection(double Px,
                        double Py,
                        double pitch,
                        double yaw,
                        cv::Point3d& worldPoint) const;

    // ------------------------------------------------------------------------
    // 以下子函数拆开每一步，方便单元测试和调试。
    // ------------------------------------------------------------------------

    // 像素 -> 相机坐标系射线。
    // 返回 ( (Px-cx)/fx, (Py-cy)/fy, 1 )。
    cv::Vec3d pixelRayCamera(double Px, double Py) const;

    // G -> W 的齐次变换 T_gw = [R_gw(pitch,yaw)  O_gw; 0 1]。
    cv::Matx44d gimbalToWorldTransform(double pitch, double yaw) const;

    // C -> W 的齐次变换 T_cw = T_gw * T_cg。
    cv::Matx44d totalTransform(double pitch, double yaw) const;

    // ------------------------------------------------------------------------
    // 齐次变换工具函数（静态，不需要对象也能调）。
    // ------------------------------------------------------------------------

    // 用 R 和 t 组装 4x4 齐次矩阵：
    //   T = [R t]
    //       [0 1]
    static cv::Matx44d makeRigidTransform(const cv::Matx33d& R, const cv::Vec3d& t);

    // 利用刚体变换的正交性求逆，不调用通用 4x4 求逆：
    //   T^{-1} = [R^T  -R^T t]
    //            [0    1    ]
    static cv::Matx44d invertRigidTransform(const cv::Matx44d& T);

    // 点变换：P' = T * [P; 1]。
    // 齐次坐标最后一维为 1，表示“点”，会受平移影响。
    static cv::Vec3d transformPoint(const cv::Matx44d& T, const cv::Vec3d& p);

    // 方向变换：D' = T * [D; 0]。
    // 齐次坐标最后一维为 0，表示“方向”，不受平移影响。
    static cv::Vec3d transformDirection(const cv::Matx44d& T, const cv::Vec3d& d);

    // 基本旋转矩阵，输入弧度。
    static cv::Matx33d rotationX(double rad);
    static cv::Matx33d rotationY(double rad);
    static cv::Matx33d rotationZ(double rad);

private:
    // 角度转弧度：根据 angles_in_degrees 决定是否乘 pi/180。
    double angleToRad(double angle) const;

    // 根据 pitch/yaw 生成 G -> W 的旋转矩阵 R_gw。
    cv::Matx33d gimbalToWorldRotation(double pitch, double yaw) const;

    PixelToWorldConfig config_;
};

#endif // PIXEL_TO_WORLD_H_