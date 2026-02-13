import json
import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose

class ExecutorNode(Node):
    def __init__(self):
        super().__init__('executor')

        self.declare_parameter('world_frame', 'map')
        self.declare_parameter('goal_tolerance_m', 0.8)

        self._world = self.get_parameter('world_frame').value
        self._tol = float(self.get_parameter('goal_tolerance_m').value)

        self._ac = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._sub = self.create_subscription(String, '/grapheqa/decision', self._on_decision, 10)

        self.get_logger().info("Executor listening on /grapheqa/decision and sending Nav2 goals.")

    def _on_decision(self, msg: String):
        d = json.loads(msg.data)
        if d.get("target_pose_xyz") is None:
            self.get_logger().info(f"No target (maybe confident). Answer={d.get('answer_token')} conf={d.get('confidence')}")
            return

        x, y, z = d["target_pose_xyz"]

        # Optional: nudge goal slightly (basic safety; replace with proper approach pose later)
        # Here we just use the target point directly.
        goal = NavigateToPose.Goal()
        ps = PoseStamped()
        ps.header.frame_id = self._world
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.position.z = 0.0
        ps.pose.orientation.w = 1.0
        goal.pose = ps

        if not self._ac.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn("Nav2 action server not available (navigate_to_pose). Is Nav2 running?")
            return

        self.get_logger().info(f"Sending Nav2 goal to ({x:.2f}, {y:.2f}) action={d.get('action')} target={d.get('target_id')}")
        send_future = self._ac.send_goal_async(goal)
        send_future.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, fut):
        goal_handle = fut.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Nav2 goal rejected.")
            return
        res_fut = goal_handle.get_result_async()
        res_fut.add_done_callback(self._on_result)

    def _on_result(self, fut):
        res = fut.result().result
        self.get_logger().info(f"Nav2 finished with result: {res}")

def main():
    rclpy.init()
    node = ExecutorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
