#!/usr/bin/env python3


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
        self.declare_parameter('min_goal_separation_m', 0.5)
        self.declare_parameter('allow_preempt', False)  # cancel & replace active goal?

        self._world = self.get_parameter('world_frame').value
        self._tol = float(self.get_parameter('goal_tolerance_m').value)
        self._min_sep = float(self.get_parameter("min_goal_separation_m").value)
        self._allow_preempt = bool(self.get_parameter("allow_preempt").value)

        self._ac = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._sub = self.create_subscription(String, '/grapheqa/decision', self._on_decision, 10)

        self._last_goal_xy = None
        self._active_goal_handle = None
        self._active_goal_xy = None

        self.get_logger().info("Executor listening on /grapheqa/decision and sending Nav2 goals.")

    def _on_decision(self, msg: String):
        try:
            d = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"Bad decision JSON: {e}")
            return

        target = d.get("target_pose_xyz")

        # If planner says "no target" (likely confident), cancel any active goal
        if target is None:
            if self._active_goal_handle is not None:
                self.get_logger().info(
                    f"No target (confident). Canceling active goal. Answer={d.get('answer_token')} conf={d.get('confidence')}"
                )
                self._cancel_active_goal()
            else:
                self.get_logger().info(
                    f"No target (maybe confident). Answer={d.get('answer_token')} conf={d.get('confidence')}"
                )
            return

        x, y, z = target
        x, y = float(x), float(y)

        # If there is an active goal:
        if self._active_goal_handle is not None:
            if not self._allow_preempt:
                self.get_logger().debug("Active goal in progress; ignoring new decision.")
                return
            else:
                # preempt only if sufficiently different
                if self._active_goal_xy is not None:
                    dx = x - self._active_goal_xy[0]
                    dy = y - self._active_goal_xy[1]
                    if math.hypot(dx, dy) < self._min_sep:
                        self.get_logger().debug("New goal too close to active; ignoring.")
                        return
                self.get_logger().info("Preempting active goal (cancel + send new).")
                self._cancel_active_goal()

        # avoid spamming same spot repeatedly
        if self._last_goal_xy is not None:
            dx = x - self._last_goal_xy[0]
            dy = y - self._last_goal_xy[1]
            if math.hypot(dx, dy) < self._min_sep:
                self.get_logger().debug("Goal too close to last; ignoring.")
                return

        self._last_goal_xy = (x, y)

        goal = NavigateToPose.Goal()
        ps = PoseStamped()
        ps.header.frame_id = self._world
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.position.z = 0.0
        ps.pose.orientation.w = 1.0
        goal.pose = ps

        if not self._ac.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn("Nav2 action server not available (navigate_to_pose). Is Nav2 running?")
            return

        self.get_logger().info(
            f"Sending Nav2 goal to ({x:.2f}, {y:.2f}) action={d.get('action')} target={d.get('target_id')}"
        )
        send_future = self._ac.send_goal_async(goal)
        send_future.add_done_callback(lambda fut: self._on_goal_response(fut, (x, y)))

    def _on_goal_response(self, fut, goal_xy):
        goal_handle = fut.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Nav2 goal rejected.")
            return

        self._active_goal_handle = goal_handle
        self._active_goal_xy = goal_xy

        res_fut = goal_handle.get_result_async()
        res_fut.add_done_callback(self._on_result)

    def _on_result(self, fut):
        try:
            res = fut.result().result
            self.get_logger().info(f"Nav2 finished with result: {res}")
        except Exception as e:
            self.get_logger().warn(f"Nav2 result error: {e}")

        # clear active
        self._active_goal_handle = None
        self._active_goal_xy = None

    def _cancel_active_goal(self):
        if self._active_goal_handle is None:
            return
        try:
            cancel_fut = self._active_goal_handle.cancel_goal_async()
            cancel_fut.add_done_callback(lambda _: self.get_logger().info("Cancel requested."))
        except Exception as e:
            self.get_logger().warn(f"Cancel failed: {e}")
        finally:
            self._active_goal_handle = None
            self._active_goal_xy = None


def main():
    rclpy.init()
    node = ExecutorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
