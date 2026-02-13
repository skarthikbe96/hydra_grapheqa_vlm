# grapheqa_ros2/dsg_bridge_node.py

import json
from typing import Dict, List, Tuple, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from tf2_ros import Buffer, TransformListener


# Try import common Hydra DSG type (adjust if needed)
# If your type differs, change this import to the correct one.
try:
    from hydra_msgs.msg import DynamicSceneGraph
except Exception as e:
    DynamicSceneGraph = None


def _as_xyz(obj) -> Optional[Tuple[float, float, float]]:
    """
    Extract (x,y,z) from common patterns:
      - obj has .x .y .z
      - obj has .position.x .position.y .position.z
      - obj is a tuple/list
    """
    if obj is None:
        return None
    if isinstance(obj, (list, tuple)) and len(obj) >= 3:
        return (float(obj[0]), float(obj[1]), float(obj[2]))
    if hasattr(obj, "x") and hasattr(obj, "y"):
        z = getattr(obj, "z", 0.0)
        return (float(obj.x), float(obj.y), float(z))
    if hasattr(obj, "position") and hasattr(obj.position, "x") and hasattr(obj.position, "y"):
        z = getattr(obj.position, "z", 0.0)
        return (float(obj.position.x), float(obj.position.y), float(z))
    return None


def _guess_layer(name: str) -> str:
    s = (name or "").lower()
    # Common Hydra naming: rooms/places/objects/agents
    if "room" in s:
        return "room"
    if "place" in s or "region" in s or "area" in s:
        return "region"
    if "agent" in s or "robot" in s:
        return "agent"
    # default to object if it has a semantic label
    return "object"


class DsgBridgeNode(Node):
    """
    Subscribes to Hydra DSG and publishes a compact JSON snapshot:
      /grapheqa/dsg_snapshot (std_msgs/String)
    """

    def __init__(self):
        super().__init__("dsg_bridge")

        self.declare_parameter("dsg_topic", "/hydra/backend/dsg")
        self.declare_parameter("world_frame", "map")
        self.declare_parameter("base_frame", "base_footprint")

        self._dsg_topic = self.get_parameter("dsg_topic").value
        self._world_frame = self.get_parameter("world_frame").value
        self._base_frame = self.get_parameter("base_frame").value

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._pub = self.create_publisher(String, "/grapheqa/dsg_snapshot", 10)

        if DynamicSceneGraph is None:
            self.get_logger().error(
                "Could not import hydra_msgs.msg.DynamicSceneGraph. "
                "Fix the import in dsg_bridge_node.py to your correct Hydra DSG msg type."
            )
            self._sub = None
        else:
            self._sub = self.create_subscription(DynamicSceneGraph, self._dsg_topic, self._on_dsg, 10)
            self.get_logger().info(f"Subscribed to {self._dsg_topic} publishing /grapheqa/dsg_snapshot")

        # Publish agent-only snapshot periodically in case DSG is missing
        self._timer = self.create_timer(1.0, self._publish_agent_only_snapshot)

        self._last_snapshot_json: Optional[str] = None

    def _publish_agent_only_snapshot(self):
        # If DSG is already publishing snapshots frequently, do nothing.
        # If not, keep something alive for downstream nodes.
        if self._last_snapshot_json is not None:
            return

        snap = self._build_snapshot(nodes={}, edges=[])
        self._pub.publish(String(data=json.dumps(snap)))

    def _build_snapshot(self, nodes: Dict, edges: List[Tuple[str, str, str]]) -> dict:
        # Always inject agent from TF (map -> base)
        try:
            t = self._tf_buffer.lookup_transform(self._world_frame, self._base_frame, rclpy.time.Time())
            agent_xyz = (
                float(t.transform.translation.x),
                float(t.transform.translation.y),
                float(t.transform.translation.z),
            )
        except Exception:
            agent_xyz = (0.0, 0.0, 0.0)

        if "agent_0" not in nodes:
            nodes["agent_0"] = {"id": "agent_0", "name": "agent", "pos": list(agent_xyz), "layer": "agent"}

        payload = {
            "frame_id": self._world_frame,
            "nodes": nodes,
            "edges": [{"src": s, "dst": d, "type": t} for (s, d, t) in edges],
        }
        return payload

    def _on_dsg(self, msg):
        """
        Hydra message schemas vary. This tries several common layouts.
        If parsing fails, you'll still get an agent node.
        """

        nodes: Dict[str, dict] = {}
        edges: List[Tuple[str, str, str]] = []

        # ---- Attempt 1: msg has .nodes and .edges flat arrays ----
        try:
            if hasattr(msg, "nodes") and hasattr(msg, "edges"):
                # node objects might have: id, layer_id/layer, name/label, position/attributes.position
                for n in msg.nodes:
                    nid = getattr(n, "id", None)
                    if nid is None:
                        continue
                    # layer/name
                    layer_name = getattr(n, "layer", None) or getattr(n, "layer_name", None) or getattr(n, "layer_id", None)
                    layer = _guess_layer(str(layer_name))
                    name = getattr(n, "name", None) or getattr(n, "label", None) or f"{layer}_{nid}"
                    pos = _as_xyz(getattr(n, "position", None)) or _as_xyz(getattr(n, "pos", None)) or _as_xyz(getattr(getattr(n, "attributes", None), "position", None))
                    if pos is None:
                        continue
                    node_id = f"{layer}_{nid}"
                    nodes[node_id] = {"id": node_id, "name": str(name), "pos": [pos[0], pos[1], pos[2]], "layer": layer}

                for e in msg.edges:
                    src = getattr(e, "source", None) or getattr(e, "src", None)
                    dst = getattr(e, "target", None) or getattr(e, "dst", None)
                    et = getattr(e, "type", None) or getattr(e, "edge_type", None) or "link"
                    if src is None or dst is None:
                        continue
                    # We don’t know src/dst layer here; store raw and fix-up later if possible
                    edges.append((str(src), str(dst), str(et)))
        except Exception as ex:
            self.get_logger().warn(f"DSG parse attempt 1 failed: {ex}")

        # ---- Attempt 2: msg has layered graph: msg.layers[*].nodes / msg.layers[*].edges ----
        if not nodes:
            try:
                if hasattr(msg, "layers"):
                    for L in msg.layers:
                        lname = getattr(L, "name", None) or getattr(L, "layer_name", None) or str(getattr(L, "id", "layer"))
                        guessed = _guess_layer(str(lname))
                        # nodes
                        if hasattr(L, "nodes"):
                            for n in L.nodes:
                                nid = getattr(n, "id", None)
                                if nid is None:
                                    continue
                                name = getattr(n, "name", None) or getattr(n, "label", None) or f"{guessed}_{nid}"
                                pos = _as_xyz(getattr(n, "position", None)) or _as_xyz(getattr(n, "pos", None))
                                if pos is None:
                                    continue
                                node_id = f"{guessed}_{nid}"
                                nodes[node_id] = {"id": node_id, "name": str(name), "pos": [pos[0], pos[1], pos[2]], "layer": guessed}
                        # edges (optional)
                        if hasattr(L, "edges"):
                            for e in L.edges:
                                src = getattr(e, "source", None) or getattr(e, "src", None)
                                dst = getattr(e, "target", None) or getattr(e, "dst", None)
                                et = getattr(e, "type", None) or getattr(e, "edge_type", None) or "link"
                                if src is None or dst is None:
                                    continue
                                edges.append((str(src), str(dst), str(et)))
            except Exception as ex:
                self.get_logger().warn(f"DSG parse attempt 2 failed: {ex}")

        # ---- Fix up edge endpoints if they are numeric IDs ----
        # If edges are like ("12","7"), we don’t know which layer prefix.
        # We will only keep edges that already refer to existing node_ids.
        fixed_edges: List[Tuple[str, str, str]] = []
        for s, d, t in edges:
            if s in nodes and d in nodes:
                fixed_edges.append((s, d, t))
        edges = fixed_edges

        snapshot = self._build_snapshot(nodes=nodes, edges=edges)
        js = json.dumps(snapshot)
        self._last_snapshot_json = js

        self._pub.publish(String(data=js))


def main():
    rclpy.init()
    node = DsgBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
