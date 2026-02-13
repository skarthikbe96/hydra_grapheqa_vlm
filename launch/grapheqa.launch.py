from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='grapheqa_ros2',
            executable='dsg_bridge',
            name='dsg_bridge',
            output='screen',
            parameters=[{
                'dsg_topic': '/hydra/backend/dsg',
                'world_frame': 'map',
                'base_frame': 'base_footprint',
            }]
        ),
        Node(
            package='grapheqa_ros2',
            executable='frontiers',
            name='frontiers',
            output='screen',
            parameters=[{
                'map_topic': '/map',
                'world_frame': 'map',
                'cluster_min_size': 40,
            }]
        ),
        Node(
            package='grapheqa_ros2',
            executable='visual_memory',
            name='visual_memory',
            output='screen',
            parameters=[{
                'image_topic': '/camera_pan_tilt/image',
                'output_dir': '/tmp/grapheqa_images',
                'save_hz': 1.0,
            }]
        ),
        Node(
            package='grapheqa_ros2',
            executable='vlm_planner',
            name='vlm_planner',
            output='screen',
            parameters=[{
                'output_dir': '/tmp/grapheqa_images',
                # GraphEQA VLM config:
                'vlm_model_name': 'gpt-4o-mini',  # change to your model
                'use_image': True,
                'add_history': True,
                # Question inputs (for quick testing):
                'question': 'What room is the couch in?',
                'choices': ['kitchen', 'living room', 'bedroom', 'bathroom'],
                'pred_candidates': ['A', 'B', 'C', 'D'],
                'gt_answer': 'B',
            }]
        ),
        Node(
            package='grapheqa_ros2',
            executable='executor',
            name='executor',
            output='screen',
            parameters=[{
                'world_frame': 'map',
                'nav2_action': 'navigate_to_pose',
                'goal_tolerance_m': 0.8,
            }]
        ),
    ])
