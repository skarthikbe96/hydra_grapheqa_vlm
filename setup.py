from setuptools import setup, find_packages

package_name = 'grapheqa_ros2'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    include_package_data=True,
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/grapheqa.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'dsg_bridge = grapheqa_ros2.dsg_bridge_node:main',
            'frontiers = grapheqa_ros2.frontier_node:main',
            'visual_memory = grapheqa_ros2.visual_memory_node:main',
            'vlm_planner = grapheqa_ros2.vlm_planner_node:main',
            'vlm_planner_bridge = grapheqa_ros2.vlm_planner_bridge:main',
            'executor = grapheqa_ros2.executor_node:main',
            'plan_result_to_decision = grapheqa_ros2.plan_result_to_decision_node:main',
        ],
    },
)
