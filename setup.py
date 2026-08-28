from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'polebot_telemetry'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('share', package_name, 'meshes'), glob('meshes/*.stl')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='itsuya',
    maintainer_email='itsuya@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'battery_sim = polebot_telemetry.battery_sim:main',
            'influx_bridge = polebot_telemetry.influx_bridge:main',
            'battery_physics = polebot_telemetry.battery_physics_sim:main',
            'system_recorder = polebot_telemetry.system_recorder:main',
            'telemetry_logger = polebot_telemetry.telemetry_logger:main',
            'scenario_runner  = polebot_telemetry.scenario_runner:main',
            'path_planning_node = polebot_telemetry.path_planning_node:main',
            'energy_path_predictor = polebot_telemetry.energy_path_predictor:main',
        ],
    },
)
