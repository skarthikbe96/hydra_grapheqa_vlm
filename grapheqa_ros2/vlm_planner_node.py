
#!/usr/bin/env python3

import json
import pathlib
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseArray

from .sg_ros_adapter import SceneGraphRosAdapter, NodeData

# Import GraphEQA planner (unchanged)
from grapheqa_ros2.graph_eqa.graph_eqa.planners.vlm_planner_gpt import VLMPlannerEQAGPT
from grapheqa_ros2.graph_eqa.graph_eqa.scene_graph.scene_graph_sim import SceneGraphSim


class VlmPlannerNode(Node):
    def __init__(self):
        super().__init__('vlm_planner')

        self.declare_parameter('output_dir', '/tmp/grapheqa_images')
        self.declare_parameter('vlm_model_name', 'gpt-4o-mini')
        self.declare_parameter('use_image', True)
        self.declare_parameter('add_history', True)

        # quick test params
        self.declare_parameter('question', 'What is the question?')
        self.declare_parameter('choices', ['A', 'B'])
        self.declare_parameter('pred_candidates', ['A', 'B'])
        self.declare_parameter('gt_answer', 'A')

        self._output_dir = pathlib.Path(self.get_parameter('output_dir').value)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._sg = SceneGraphRosAdapter(frontier_to_object_radius_m=2.0)
        self._latest_dsg_snapshot = None
        self._latest_frontiers: List[Tuple[float,float,float]] = []

        # Subscriptions
        self.create_subscription(String, '/grapheqa/dsg_snapshot', self._on_dsg_snapshot, 10)
        self.create_subscription(PoseArray, '/grapheqa/frontiers', self._on_frontiers, 10)

        # Publish planner decision
        self._pub = self.create_publisher(String, '/grapheqa/decision', 10)

        # Create planner
        class Cfg:
            name = self.get_parameter('vlm_model_name').value
            use_image = bool(self.get_parameter('use_image').value)
            add_history = bool(self.get_parameter('add_history').value)

        question = self.get_parameter('question').value
        choices = list(self.get_parameter('choices').value)
        pred_candidates = list(self.get_parameter('pred_candidates').value)
        gt_answer = self.get_parameter('gt_answer').value

        self._planner = VLMPlannerEQAGPT(
            cfg=Cfg,
            sg_sim=self._sg,
            question=question,
            pred_candidates=pred_candidates,
            choices=choices,
            answer=gt_answer,
            output_path=self._output_dir
        )

        # run loop timer
        self._timer = self.create_timer(3.0, self._tick)
        self.get_logger().info("VLM planner node ready. Publishing /grapheqa/decision")

    def _on_dsg_snapshot(self, msg: String):
        self._latest_dsg_snapshot = json.loads(msg.data)

    def _on_frontiers(self, msg: PoseArray):
        self._latest_frontiers = [(p.position.x, p.position.y, p.position.z) for p in msg.poses]

    def _tick(self):
        if not self._latest_dsg_snapshot:
            self.get_logger().warn("No DSG snapshot yet.")
            return

        # Build adapter snapshot
        nodes = {}
        for nid, nd in self._latest_dsg_snapshot["nodes"].items():
            nodes[nid] = NodeData(
                node_id=nd["id"],
                name=nd["name"],
                pos=tuple(nd["pos"]),
                layer=nd["layer"],
            )
        edges = [(e["src"], e["dst"], e["type"]) for e in self._latest_dsg_snapshot["edges"]]

        self._sg.update_from_snapshot(nodes, edges, self._latest_frontiers)

        # Run GraphEQA step
        target_pose, target_id, is_conf, conf, ans_token = self._planner.get_next_action()

        decision = {
            "t": self._planner.t - 1,
            "answer_token": ans_token,
            "is_confident": bool(is_conf),
            "confidence": float(conf),
            "target_id": target_id,
            "target_pose_xyz": list(target_pose) if target_pose else None,
            "action": None if target_id is None else ("GOTO_FRONTIER" if str(target_id).startswith("frontier_") else "GOTO_OBJECT"),
        }
        out = String()
        out.data = json.dumps(decision)
        self._pub.publish(out)

def main():
    rclpy.init()
    node = VlmPlannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
