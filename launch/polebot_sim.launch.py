import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import Command

def generate_launch_description():
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    pkg_telemetry = get_package_share_directory('polebot_telemetry')
    
    xacro_file = os.path.join(pkg_telemetry, 'urdf', 'polebot.urdf.xacro')

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': Command(['xacro ', xacro_file])}]
        ),
        
        # 3. Luncurkan Gazebo Harmonic
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
            ),
            launch_arguments={'gz_args': '-r ' + os.path.join(pkg_telemetry, 'worlds', 'depot.sdf')}.items(),
        ),
        
        Node(
            package='ros_gz_sim',
            executable='create',
            arguments=['-name', 'polebot_amr', '-topic', 'robot_description', '-z', '0.3'],
            output='screen'
        ),
        
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[
                '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
                '/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry',
                '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
            ],
            output='screen'
        ),

        Node(
            package='polebot_telemetry',
            executable='battery_physics',
            name='battery_physics_sim',
            parameters=[{'scenario': 'NEW'}],
            output='screen'
        ),
    ])