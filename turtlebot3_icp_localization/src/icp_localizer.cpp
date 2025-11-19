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
#include <limits>
#include <Eigen/Dense>
#include <Eigen/SVD>

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

        // ICP Algorithm
        const int max_iterations = 20;
        const double convergence_threshold = 0.01; // radianes (~0.57 grados)
        const double rmse_threshold = 0.05; // metros
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr P_cloud = current_cloud;
        Eigen::Matrix4f final_transform = Eigen::Matrix4f::Identity();
        
        for (int iter = 0; iter < max_iterations; ++iter)
        {
            // 1. Closest Point Matching
            pcl::PointCloud<pcl::PointXYZ>::Ptr P_matched(new pcl::PointCloud<pcl::PointXYZ>);
            P_matched->points.resize(stable_cloud_->points.size());
            
            for (size_t i = 0; i < stable_cloud_->points.size(); ++i)
            {
                const auto& x_point = stable_cloud_->points[i];
                double min_dist = std::numeric_limits<double>::max();
                int min_idx = 0;
                
                for (size_t j = 0; j < P_cloud->points.size(); ++j)
                {
                    const auto& p_point = P_cloud->points[j];
                    double dist = std::sqrt(
                        std::pow(x_point.x - p_point.x, 2) +
                        std::pow(x_point.y - p_point.y, 2) +
                        std::pow(x_point.z - p_point.z, 2)
                    );
                    
                    if (dist < min_dist)
                    {
                        min_dist = dist;
                        min_idx = j;
                    }
                }
                
                P_matched->points[i] = P_cloud->points[min_idx];
            }
            
            // 2. Calculate centers of mass
            Eigen::Vector3f mx(0.0f, 0.0f, 0.0f);
            Eigen::Vector3f mp(0.0f, 0.0f, 0.0f);
            
            for (const auto& point : stable_cloud_->points)
            {
                mx[0] += point.x;
                mx[1] += point.y;
                mx[2] += point.z;
            }
            mx /= static_cast<float>(stable_cloud_->points.size());
            
            for (const auto& point : P_matched->points)
            {
                mp[0] += point.x;
                mp[1] += point.y;
                mp[2] += point.z;
            }
            mp /= static_cast<float>(P_matched->points.size());
            
            // 3. Subtract centers of mass
            Eigen::MatrixXf X_prime(3, stable_cloud_->points.size());
            Eigen::MatrixXf P_prime(3, P_matched->points.size());
            
            for (size_t i = 0; i < stable_cloud_->points.size(); ++i)
            {
                X_prime(0, i) = stable_cloud_->points[i].x - mx[0];
                X_prime(1, i) = stable_cloud_->points[i].y - mx[1];
                X_prime(2, i) = stable_cloud_->points[i].z - mx[2];
                
                P_prime(0, i) = P_matched->points[i].x - mp[0];
                P_prime(1, i) = P_matched->points[i].y - mp[1];
                P_prime(2, i) = P_matched->points[i].z - mp[2];
            }
            
            // 4. SVD
            Eigen::Matrix3f W = X_prime * P_prime.transpose();
            Eigen::JacobiSVD<Eigen::Matrix3f> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3f U = svd.matrixU();
            Eigen::Matrix3f V = svd.matrixV();
            
            // 5. Calculate rotation and translation
            Eigen::Matrix3f R = U * V.transpose();
            
            // Correction for reflection
            if (R.determinant() < 0)
            {
                V.col(2) *= -1;
                R = U * V.transpose();
            }
            
            Eigen::Vector3f t = mx - R * mp;
            
            // 6. Build transformation matrix
            Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
            transform.block<3, 3>(0, 0) = R;
            transform.block<3, 1>(0, 3) = t;
            
            // 7. Calculate RMSE
            double rmse = 0.0;
            for (size_t i = 0; i < stable_cloud_->points.size(); ++i)
            {
                Eigen::Vector3f x_vec(stable_cloud_->points[i].x,
                                      stable_cloud_->points[i].y,
                                      stable_cloud_->points[i].z);
                Eigen::Vector3f p_vec(P_matched->points[i].x,
                                      P_matched->points[i].y,
                                      P_matched->points[i].z);
                
                Eigen::Vector3f p_transformed = R * p_vec + t;
                rmse += (x_vec - p_transformed).squaredNorm();
            }
            rmse = std::sqrt(rmse / stable_cloud_->points.size());
            
            // 8. Apply transformation to P_cloud for next iteration
            pcl::PointCloud<pcl::PointXYZ>::Ptr P_transformed(new pcl::PointCloud<pcl::PointXYZ>);
            P_transformed->points.resize(P_cloud->points.size());
            
            for (size_t i = 0; i < P_cloud->points.size(); ++i)
            {
                Eigen::Vector3f p(P_cloud->points[i].x,
                                  P_cloud->points[i].y,
                                  P_cloud->points[i].z);
                Eigen::Vector3f p_new = R * p + t;
                
                P_transformed->points[i].x = p_new[0];
                P_transformed->points[i].y = p_new[1];
                P_transformed->points[i].z = p_new[2];
            }
            
            P_cloud = P_transformed;
            final_transform = transform * final_transform;
            
            // 9. Check convergence
            if (rmse < rmse_threshold)
            {
                break;
            }
        }
        
        // 10. Filter convergence: check rotation angle
        Eigen::Matrix3f R_final = final_transform.block<3, 3>(0, 0);
        double rotation_angle = std::acos((R_final.trace() - 1.0) / 2.0);
        
        if (std::abs(rotation_angle) < convergence_threshold)
        {
            update_pose(final_transform);
            publish_transform();
        }
        
        stable_cloud_ = P_cloud; // Con la convergencia aprobada, actualizar la nube estable.
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
