#!/usr/bin/env python3

import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class PlanResultToDecision(Node):
    def __init__(self):
        super().__init__("plan_result_to_decision")

        self.sub = self.create_subscription(
            String,
            "/grapheqa/plan_result",
            self.cb,
            10,
        )
        self.pub = self.create_publisher(
            String,
            "/grapheqa/decision",
            10,
        )

        self.get_logger().info(
            "Bridging /grapheqa/plan_result -> /grapheqa/decision"
        )

    def cb(self, msg: String):
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"Invalid JSON: {e}")
            return

        plan = data.get("plan", {})
        goal = plan.get("goal", None)

        decision = {
            "action": "NONE",
            "target_id": None,
            "target_pose_xyz": None,
            "answer_token": plan.get("answer"),
            "confidence": plan.get("confidence", None),
        }

        if goal is not None:
            decision["action"] = "NAVIGATE"

            if isinstance(goal, dict):
                decision["target_pose_xyz"] = [
                    float(goal.get("x", 0.0)),
                    float(goal.get("y", 0.0)),
                    float(goal.get("z", 0.0)),
                ]
            elif isinstance(goal, (list, tuple)) and len(goal) >= 2:
                decision["target_pose_xyz"] = [
                    float(goal[0]),
                    float(goal[1]),
                    float(goal[2]) if len(goal) > 2 else 0.0,
                ]

        self.pub.publish(String(data=json.dumps(decision)))


def main():
    rclpy.init()
    node = PlanResultToDecision()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
