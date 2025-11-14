import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share_dir = get_package_share_directory('turtlebot3_icp_localization')
    rviz_config_file = os.path.join(pkg_share_dir, 'rviz', 'icp.rviz')
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'),
        Node(
            package='turtlebot3_icp_localization',
            executable='icp_localizer',
            name='icp_localizer',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}]
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_file],
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}]
        )
    ])
