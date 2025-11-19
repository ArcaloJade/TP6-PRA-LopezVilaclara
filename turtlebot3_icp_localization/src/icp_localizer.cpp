#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <laser_geometry/laser_geometry.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// Add for std::abs
#include <cmath>

// Mis includes
#include <Eigen/Dense>
#include <pcl/kdtree/kdtree_flann.h>
#include <vector>
#include <limits>

class IcpLocalizer : public rclcpp::Node
{
public:
    IcpLocalizer()
    : Node("icp_localizer"), first_scan_(true)
    {
        // Subscribers
        scan_subscriber_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10, std::bind(&IcpLocalizer::scan_callback, this, std::placeholders::_1));

        // Publishers
        pose_publisher_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("/pose_with_covariance", 10);
        current_cloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/current_cloud", 10);
        stable_cloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/stable_cloud", 10);

        // TF Broadcaster
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        // Initialize pose
        current_pose_.header.frame_id = "odom";
        current_pose_.pose.pose.orientation.w = 1.0;
    }

private:
    void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
    /// #TODO: Ejecuten ICP cada vez que se reciba un nuevo scan: mantendrán stable_cloud como la convergencia de la nube de puntos por ICP
    /// y actualizarán current_pose_ y current_cloud_publisher_ con la nueva convergencia para filtrar odometría y observar el correcto funcionamiento
    /// de la nube de puntos en Rviz2. 
    {
        // Convert LaserScan to PointCloud
        sensor_msgs::msg::PointCloud2 cloud_msg;
        projector_.projectLaser(*msg, cloud_msg);
        cloud_msg.header.frame_id = "base_link";
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr current_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(cloud_msg, *current_cloud);

        if (first_scan_)
        {
            stable_cloud_ = current_cloud;
            first_scan_ = false;
            return;
        }

        // Publish clouds for visualization
        sensor_msgs::msg::PointCloud2 stable_cloud_msg;
        pcl::toROSMsg(*stable_cloud_, stable_cloud_msg);
        stable_cloud_msg.header.stamp = this->get_clock()->now();
        stable_cloud_msg.header.frame_id = "odom";
        stable_cloud_publisher_->publish(stable_cloud_msg);

        sensor_msgs::msg::PointCloud2 curr_cloud_msg;
        pcl::toROSMsg(*current_cloud, curr_cloud_msg);
        curr_cloud_msg.header.stamp = this->get_clock()->now();
        curr_cloud_msg.header.frame_id = "odom";
        current_cloud_publisher_->publish(curr_cloud_msg);

        // ICP
        // Parámetros
        const int max_iters = 20;
        const float tolerance = 1e-4f;
        const float max_correspondence_distance = 0.6f; // metros
        const float max_rotation_rad = 0.6f; // si la rotación total excede esto no updateo (0.6f ≈ 34° mas o menos)
        const float max_translation = 1.0f; // si la traslación excede esto no updateo tpoco

        // Uso árboles kd
        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud(stable_cloud_);

        // copia de la CurrentCloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud(new pcl::PointCloud<pcl::PointXYZ>(*current_cloud));
        Eigen::Matrix4f cumulative_transform = Eigen::Matrix4f::Identity();

        float prev_error = std::numeric_limits<float>::max();
        for (int iter = 0; iter < max_iters; ++iter)
        {
            std::vector<std::pair<int,int>> correspondences;
            std::vector<Eigen::Vector3f> src_pts;
            std::vector<Eigen::Vector3f> tgt_pts;

            // Para cada punto en nuestra copia, encuentro el vecino más cercano en la StableCloud
            std::vector<int> pointIdxNKNSearch(1);
            std::vector<float> pointNKNSquaredDistance(1);

            for (size_t i = 0; i < src_cloud->points.size(); ++i)
            {
                const pcl::PointXYZ &p = src_cloud->points[i];
                if (!std::isfinite(p.x) || !std::isfinite(p.y) || std::isnan(p.x)) continue;

                if (kdtree.nearestKSearch(p, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
                {
                    float dist_sq = pointNKNSquaredDistance[0];
                    if (dist_sq <= max_correspondence_distance * max_correspondence_distance)
                    {
                        int tgt_idx = pointIdxNKNSearch[0];
                        const pcl::PointXYZ &q = stable_cloud_->points[tgt_idx];

                        src_pts.emplace_back(p.x, p.y, p.z);
                        tgt_pts.emplace_back(q.x, q.y, q.z);
                    }
                }
            }

            if (src_pts.size() < 3)
            {
                // No hay correspondencias suficientes
                break;
            }

            // Centroides
            Eigen::Vector3f centroid_src = Eigen::Vector3f::Zero();
            Eigen::Vector3f centroid_tgt = Eigen::Vector3f::Zero();
            for (size_t i = 0; i < src_pts.size(); ++i)
            {
                centroid_src += src_pts[i];
                centroid_tgt += tgt_pts[i];
            }
            centroid_src /= static_cast<float>(src_pts.size());
            centroid_tgt /= static_cast<float>(tgt_pts.size());

            // Matriz de cross-covariance (H)
            Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
            for (size_t i = 0; i < src_pts.size(); ++i)
            {
                Eigen::Vector3f ps = src_pts[i] - centroid_src;
                Eigen::Vector3f pt = tgt_pts[i] - centroid_tgt;
                H += ps * pt.transpose();
            }

            // hago el SVD
            Eigen::JacobiSVD<Eigen::Matrix3f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3f U = svd.matrixU();
            Eigen::Matrix3f V = svd.matrixV();

            Eigen::Matrix3f R = V * U.transpose();
            // Me aseguro q la rotación se haga bien
            if (R.determinant() < 0)
            {
                Eigen::Matrix3f Vt = V;
                Vt.col(2) *= -1.0f;
                R = Vt * U.transpose();
            }

            Eigen::Vector3f t = centroid_tgt - R * centroid_src;

            // Transformación incremental a la nube que guarde como copia de la Current
            for (size_t i = 0; i < src_cloud->points.size(); ++i)
            {
                Eigen::Vector3f p(src_cloud->points[i].x, src_cloud->points[i].y, src_cloud->points[i].z);
                Eigen::Vector3f p_trans = R * p + t;
                src_cloud->points[i].x = p_trans.x();
                src_cloud->points[i].y = p_trans.y();
                src_cloud->points[i].z = p_trans.z();
            }

            // Actualizo el cumulative transform
            Eigen::Matrix4f T_inc = Eigen::Matrix4f::Identity();
            T_inc.block<3,3>(0,0) = R;
            T_inc.block<3,1>(0,3) = t;
            cumulative_transform = T_inc * cumulative_transform;

            // Error medio punto a punto
            float mean_error = 0.0f;
            size_t count_pairs = std::min(src_pts.size(), tgt_pts.size());
            for (size_t i = 0; i < count_pairs; ++i)
            {
                Eigen::Vector3f p_trans = R * src_pts[i] + t; // posición aproximada
                mean_error += (p_trans - tgt_pts[i]).norm();
            }
            mean_error /= static_cast<float>(count_pairs);

            if (std::abs(prev_error - mean_error) < tolerance)
            {
                break; // convergió
            }
            prev_error = mean_error;
        }

        // Evalúo la cumulative y la rechazo si es demasiado grande
        Eigen::Matrix3f R_total = cumulative_transform.block<3,3>(0,0);
        Eigen::Vector3f t_total = cumulative_transform.block<3,1>(0,3);
        Eigen::AngleAxisf angle_axis(R_total);
        float rotation_angle = std::abs(angle_axis.angle());
        float translation_norm = t_total.norm();

        bool accept = true;
        if (!std::isfinite(rotation_angle) || rotation_angle > max_rotation_rad) accept = false;
        if (!std::isfinite(translation_norm) || translation_norm > max_translation) accept = false;

        if (accept)
        {
            update_pose(cumulative_transform);
            publish_transform();

            // Publico pose con covarianza (mantenemos una covarianza ingenua para demostración)
            current_pose_.header.stamp = this->get_clock()->now();
            current_pose_.header.frame_id = "odom";
            
            // esto es una covarianza dummy
            for (int i = 0; i < 36; ++i) current_pose_.pose.covariance[i] = 0.0;
            current_pose_.pose.covariance[0] = 0.01;  // x
            current_pose_.pose.covariance[7] = 0.01;  // y
            current_pose_.pose.covariance[35] = 0.05; // yaw
            pose_publisher_->publish(current_pose_);

            // Publico la CurrentCloud transformada
            pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>(*src_cloud));
            sensor_msgs::msg::PointCloud2 transformed_msg;
            pcl::toROSMsg(*transformed_cloud, transformed_msg);
            transformed_msg.header.stamp = this->get_clock()->now();
            transformed_msg.header.frame_id = "odom";
            current_cloud_publisher_->publish(transformed_msg);

            stable_cloud_ = transformed_cloud;
        }
        else
        {
            // (esto para debug) tiro mensaje si se rechaza
            RCLCPP_WARN(this->get_logger(), "ICP rejected: rotation=%.3f rad translation=%.3f m", rotation_angle, translation_norm);

            // Publico la CurrentCloud pero sin transformar
            sensor_msgs::msg::PointCloud2 fallback_msg;
            pcl::toROSMsg(*current_cloud, fallback_msg);
            fallback_msg.header.stamp = this->get_clock()->now();
            fallback_msg.header.frame_id = "odom";
            current_cloud_publisher_->publish(fallback_msg);
        }
        
    }

    void update_pose(const Eigen::Matrix4f& transform)
    {
        tf2::Matrix3x3 rot_matrix(
            transform(0, 0), transform(0, 1), transform(0, 2),
            transform(1, 0), transform(1, 1), transform(1, 2),
            transform(2, 0), transform(2, 1), transform(2, 2)
        );
        tf2::Vector3 translation(transform(0, 3), transform(1, 3), transform(2, 3));
        tf2::Transform incremental_transform(rot_matrix, translation);

        tf2::Transform current_transform;
        tf2::fromMsg(current_pose_.pose.pose, current_transform);

        tf2::Transform new_transform = current_transform * incremental_transform;

        tf2::toMsg(new_transform, current_pose_.pose.pose);
    }

    void publish_transform()
    {
        geometry_msgs::msg::TransformStamped t;

        t.header.stamp = this->get_clock()->now();
        t.header.frame_id = "odom";
        t.child_frame_id = "pra_123";

        t.transform.translation.x = current_pose_.pose.pose.position.x;
        t.transform.translation.y = current_pose_.pose.pose.position.y;
        t.transform.translation.z = current_pose_.pose.pose.position.z;
        t.transform.rotation = current_pose_.pose.pose.orientation;
        
        tf_broadcaster_->sendTransform(t);
    }

    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_subscriber_;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_publisher_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr current_cloud_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr stable_cloud_publisher_;

    laser_geometry::LaserProjection projector_;
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr stable_cloud_;
    bool first_scan_;
    geometry_msgs::msg::PoseWithCovarianceStamped current_pose_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<IcpLocalizer>();
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    
    rclcpp::shutdown();
    return 0;
}
