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

// Agregado por mí
#include <Eigen/Dense>
#include <vector>
#include <limits>
#include <pcl/kdtree/kdtree_flann.h>


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

        // A partir de acá, esto de scan_callback lo hice yo
        bool converged;
        Eigen::Matrix4f transform = run_icp(current_cloud, stable_cloud_, 0.25f, converged);

        if (converged) {
            update_pose(transform);
            publish_transform();

            stable_cloud_ = current_cloud;
        }
               
    }

    // Hecho por mí
    Eigen::Matrix4f run_icp(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& src,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& dst,
    float max_corr_dist,
    bool& converged)
`   {
        converged = false;
        
        // 1. Configuración de parámetros de iteración
        int max_iterations = 20;      // Más iteraciones porque es muy rápido
        float tolerance = 1e-6;       // Tolerancia fina para convergencia
        float min_error_change = 1e-5; 

        // Matriz de transformación final acumulada (la que retornaremos)
        Eigen::Matrix4f final_T = Eigen::Matrix4f::Identity();

        // Copia de puntos fuente para transformarlos iterativamente
        // (Usamos vector de Eigen para facilitar operaciones matemáticas)
        std::vector<Eigen::Vector3f> active_src_pts;
        active_src_pts.reserve(src->points.size());
        
        // OPTIMIZACIÓN: Downsampling simple (usar 1 de cada 2 puntos para ir volando)
        // Si quieres precisión extrema, cambia step a 1.
        int step = 2; 
        for (size_t i = 0; i < src->points.size(); i += step) {
            const auto& p = src->points[i];
            // Filtramos NaNs para evitar crashes
            if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)) {
                active_src_pts.emplace_back(p.x, p.y, p.z);
            }
        }

        // 2. Construir KD-Tree UNA sola vez con la nube destino (Target)
        // Esto es lo que acelera el proceso masivamente.
        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud(dst);

        float prev_mean_error = std::numeric_limits<float>::max();

        // --- BUCLE ICP ---
        for (int iter = 0; iter < max_iterations; ++iter)
        {
            std::vector<Eigen::Vector3f> corr_src;
            std::vector<Eigen::Vector3f> corr_dst;
            double current_sum_error = 0.0;

            // 3. Buscar Correspondencias usando KD-Tree
            for (const auto& p_src : active_src_pts)
            {
                pcl::PointXYZ searchPoint;
                searchPoint.x = p_src.x();
                searchPoint.y = p_src.y();
                searchPoint.z = p_src.z();

                std::vector<int> pointIdxNKNSearch(1);
                std::vector<float> pointNKNSquaredDistance(1);

                // Buscar el vecino más cercano (K=1)
                if (kdtree.nearestKSearch(searchPoint, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
                {
                    // Verificar umbral de distancia máxima (al cuadrado para evitar sqrt)
                    if (pointNKNSquaredDistance[0] < max_corr_dist * max_corr_dist)
                    {
                        corr_src.push_back(p_src);
                        
                        // Recuperamos el punto del destino usando el índice del KDTree
                        const auto& p_dst_pcl = dst->points[pointIdxNKNSearch[0]];
                        corr_dst.push_back(Eigen::Vector3f(p_dst_pcl.x, p_dst_pcl.y, p_dst_pcl.z));
                        
                        current_sum_error += pointNKNSquaredDistance[0];
                    }
                }
            }

            // Seguridad: Si perdemos tracking o no hay overlap
            if (corr_src.size() < 10) {
                // Si falla en la primera iteración, retornamos identidad
                if (iter == 0) return Eigen::Matrix4f::Identity(); 
                break; 
            }

            // 4. Calcular Centroides
            Eigen::Vector3f c_src = Eigen::Vector3f::Zero();
            Eigen::Vector3f c_dst = Eigen::Vector3f::Zero();

            for (size_t i = 0; i < corr_src.size(); i++) {
                c_src += corr_src[i];
                c_dst += corr_dst[i];
            }
            c_src /= corr_src.size();
            c_dst /= corr_dst.size();

            // 5. Matriz de Covarianza H
            Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
            for (size_t i = 0; i < corr_src.size(); i++) {
                Eigen::Vector3f a = corr_src[i] - c_src;
                Eigen::Vector3f b = corr_dst[i] - c_dst;
                H += a * b.transpose();
            }

            // 6. SVD (Singular Value Decomposition)
            Eigen::JacobiSVD<Eigen::Matrix3f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3f R = svd.matrixV() * svd.matrixU().transpose();

            // Corregir reflexión (si determinante es -1)
            if (R.determinant() < 0) {
                Eigen::Matrix3f V = svd.matrixV();
                V.col(2) *= -1;
                R = V * svd.matrixU().transpose();
            }

            Eigen::Vector3f t = c_dst - R * c_src;

            // 7. Actualizar transformación global acumulada
            Eigen::Matrix4f delta_T = Eigen::Matrix4f::Identity();
            delta_T.block<3,3>(0,0) = R;
            delta_T.block<3,1>(0,3) = t;
            
            final_T = delta_T * final_T;

            // 8. Aplicar la transformación a los puntos fuente "activos"
            // para que en la próxima iteración busquen mejores vecinos
            for (auto& p : active_src_pts) {
                p = R * p + t;
            }

            // 9. Chequeo de Convergencia
            float mean_error = current_sum_error / corr_src.size();
            if (std::abs(mean_error - prev_mean_error) < tolerance) {
                converged = true; // Cambio de error despreciable
                break;
            }
            prev_mean_error = mean_error;
        }

        // Consideramos convergencia si logramos terminar el bucle con suficientes puntos
        converged = true;
        return final_T;
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
