from setuptools import setup

package_name = 'grapheqa_ros2'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/grapheqa.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@todo.todo',
    description='GraphEQA VLM planner adapter for Hydra + ROS2 + Nav2',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'dsg_bridge = grapheqa_ros2.dsg_bridge_node:main',
            'frontiers = grapheqa_ros2.frontier_node:main',
            'visual_memory = grapheqa_ros2.visual_memory_node:main',
            'vlm_planner = grapheqa_ros2.vlm_planner_node:main',
            'executor = grapheqa_ros2.executor_node:main',
        ],
    },
)
