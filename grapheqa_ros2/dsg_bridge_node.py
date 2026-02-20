#!/usr/bin/env python3
# grapheqa_ros2/dsg_bridge_node.py

import json
import os
import sys
import glob
import tempfile
from typing import Dict, List, Tuple, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

# Hydra message type
try:
    from hydra_msgs.msg import DsgUpdate
except Exception:
    DsgUpdate = None


_LAYER_PRIORITY = ["object", "place", "room", "region", "building", "agent", "frontier"]


def _pick_best_layer(node_ids: List[str]) -> str:
    """Choose a node_id from a list using a layer priority heuristic."""
    if len(node_ids) == 1:
        return node_ids[0]

    def score(nid: str) -> int:
        layer = nid.split("_", 1)[0]
        try:
            return _LAYER_PRIORITY.index(layer)
        except ValueError:
            return len(_LAYER_PRIORITY) + 1

    return sorted(node_ids, key=score)[0]


def _patch_sys_path_for_spark_dsg(logger) -> bool:
    """
    Make Spark-DSG python bindings importable by adding ROS install prefixes
    to sys.path (especially <ws>/install/spark_dsg/lib/pythonX.Y/site-packages).
    Returns True if we added at least one plausible path.
    """
    added_any = False

    # 1) Try sibling install next to grapheqa_ros2 install:
    # .../install/grapheqa_ros2/lib/pythonX/site-packages/grapheqa_ros2/dsg_bridge_node.py
    # -> go up to .../install, then look for .../install/spark_dsg/lib/python*/site-packages
    here = os.path.abspath(__file__)
    install_root = here
    # climb up until ".../install/<pkg>/..." becomes ".../install"
    # (robust to different depths)
    for _ in range(10):
        install_root = os.path.dirname(install_root)
        if os.path.basename(install_root) == "install":
            break

    candidate_prefixes = []
    if os.path.basename(install_root) == "install":
        candidate_prefixes.append(os.path.join(install_root, "spark_dsg"))
        # also try the underlay install if sourced
        # (we’ll add more prefixes below)
    # 2) Also scan environment prefixes
    for envvar in ("AMENT_PREFIX_PATH", "CMAKE_PREFIX_PATH"):
        val = os.environ.get(envvar, "")
        for p in val.split(os.pathsep):
            p = p.strip()
            if not p:
                continue
            # p may be ".../install/spark_dsg" already OR ".../install"
            candidate_prefixes.append(p)
            candidate_prefixes.append(os.path.join(p, "spark_dsg"))

    # de-dup while preserving order
    seen = set()
    prefixes = []
    for p in candidate_prefixes:
        p = os.path.normpath(p)
        if p in seen:
            continue
        seen.add(p)
        prefixes.append(p)

    # For each prefix, add python site-packages if present
    for prefix in prefixes:
        if not os.path.isdir(prefix):
            continue

        # common ROS python install location
        for sp in glob.glob(os.path.join(prefix, "lib", "python*", "site-packages")):
            if sp not in sys.path:
                sys.path.insert(0, sp)
                added_any = True

        # sometimes people install the python package under "python/"
        # (less common, but harmless to check)
        py_dir = os.path.join(prefix, "python")
        if os.path.isdir(py_dir) and py_dir not in sys.path:
            sys.path.insert(0, py_dir)
            added_any = True

    if added_any:
        logger.info("[DSG] Patched sys.path to search Spark-DSG bindings.")
    return added_any


def _try_import_spark_dsg(logger):
    """
    Try importing spark_dsg. If it fails, attempt to patch sys.path and retry.
    Returns imported module or None.
    """
    try:
        import spark_dsg as sdsg  # noqa: F401
        return sdsg
    except Exception as e1:
        logger.error(f"[DSG] spark_dsg import failed: {type(e1).__name__}: {e1}")

    patched = _patch_sys_path_for_spark_dsg(logger)
    if not patched:
        logger.error("[DSG] No spark_dsg python site-packages folder found to patch sys.path.")
        return None

    try:
        import spark_dsg as sdsg  # noqa: F401
        logger.info("[DSG] spark_dsg import succeeded after sys.path patch.")
        return sdsg
    except Exception as e2:
        logger.error(f"[DSG] spark_dsg import still failed after patch: {type(e2).__name__}: {e2}")
        return None


class DsgBridgeNode(Node):
    """
    Subscribes to Hydra DSG (/hydra/backend/dsg) and publishes a compact JSON snapshot:
      /grapheqa/dsg_snapshot (std_msgs/String)

    Hydra's /hydra/backend/dsg contains a binary-serialized Spark-DSG in msg.layer_contents.
    We decode it using spark_dsg python bindings and export nodes/edges.
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

        if DsgUpdate is None:
            self.get_logger().error(
                "Could not import hydra_msgs.msg.DsgUpdate. "
                "Install/build the hydra_msgs package and source the overlay."
            )
            self._sub = None
        else:
            self._sub = self.create_subscription(DsgUpdate, self._dsg_topic, self._on_dsg, 10)
            self.get_logger().info(f"Subscribed to {self._dsg_topic} publishing /grapheqa/dsg_snapshot")

        self._timer = self.create_timer(1.0, self._publish_agent_only_snapshot)
        self._last_snapshot_json: Optional[str] = None

        # import spark_dsg once and keep it
        self._sdsg = _try_import_spark_dsg(self.get_logger())
        self._warned_no_sdsg = False

    def _publish_agent_only_snapshot(self):
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

        return {
            "frame_id": self._world_frame,
            "nodes": nodes,
            "edges": [{"src": s, "dst": d, "type": t} for (s, d, t) in edges],
        }

    def _on_dsg(self, msg):
        # If spark_dsg isn't available, publish agent-only but don't spam every callback
        if self._sdsg is None:
            if not self._warned_no_sdsg:
                self.get_logger().error(
                    "[DSG] spark_dsg python module not available in this runtime environment. "
                    "Fix by sourcing the workspace that installed spark_dsg, or ensure its "
                    "lib/pythonX.Y/site-packages is on PYTHONPATH."
                )
                self._warned_no_sdsg = True
            snapshot = self._build_snapshot(nodes={}, edges=[])
            js = json.dumps(snapshot)
            self._last_snapshot_json = js
            self._pub.publish(String(data=js))
            return

        if not hasattr(msg, "layer_contents") or msg.layer_contents is None:
            self.get_logger().warn("[DSG] msg has no layer_contents; publishing agent-only snapshot")
            snapshot = self._build_snapshot(nodes={}, edges=[])
            js = json.dumps(snapshot)
            self._last_snapshot_json = js
            self._pub.publish(String(data=js))
            return

        # Convert layer_contents -> bytes
        try:
            raw = bytes(msg.layer_contents)
        except Exception as e:
            self.get_logger().error(f"[DSG] Failed to convert layer_contents to bytes: {e}")
            snapshot = self._build_snapshot(nodes={}, edges=[])
            js = json.dumps(snapshot)
            self._last_snapshot_json = js
            self._pub.publish(String(data=js))
            return

        # Spark-DSG python API expects a filepath for load(), so write a temp file
        # try:
        #     with tempfile.NamedTemporaryFile(prefix="hydra_dsg_", suffix=".sparkdsg", delete=True) as f:
        #         f.write(raw)
        #         f.flush()
        #         DSG = getattr(self._sdsg, "DynamicSceneGraph", None)
        #         if DSG is None:
        #             from spark_dsg import _dsg_bindings
        #             DSG = _dsg_bindings.DynamicSceneGraph
        #         graph = DSG.load(f.name)

        # except Exception as e:
        #     self.get_logger().error(f"[DSG] Spark-DSG deserialize/load failed: {e}")
        #     snapshot = self._build_snapshot(nodes={}, edges=[])
        #     js = json.dumps(snapshot)
        #     self._last_snapshot_json = js
        #     self._pub.publish(String(data=js))
        #     return

        # Decode Hydra DSG bytes (these are Spark-DSG io::binary bytes)
        try:
            DSG = getattr(self._sdsg, "DynamicSceneGraph", None)
            if DSG is None:
                from spark_dsg import _dsg_bindings
                DSG = _dsg_bindings.DynamicSceneGraph

            # Keep a graph object and update it each message (more efficient)
            # Keep a graph object and update it each message
            if not hasattr(self, "_graph") or self._graph is None:
                self._graph = DSG()  # start empty

            # DsgUpdate.layer_contents is an update stream; apply it
            # If full_update is True, remove stale nodes not present in this update
            self._graph.update_from_binary(raw, bool(msg.full_update))
            graph = self._graph            

        except Exception as e:
            self.get_logger().error(f"[DSG] Spark-DSG update_from_binary failed: {e}")
            snapshot = self._build_snapshot(nodes={}, edges=[])
            js = json.dumps(snapshot)
            self._last_snapshot_json = js
            self._pub.publish(String(data=js))
            return



        # Export nodes
        nodes: Dict[str, dict] = {}
        edges: List[Tuple[str, str, str]] = []

        def layer_name_from_id(lid: int) -> str:
            return {
                2: "object",
                3: "place",
                4: "room",
                5: "building",
            }.get(int(lid), f"layer{int(lid)}")

        try:
            layer_ids = list(graph.layers())
        except Exception:
            layer_ids = [2, 3, 4, 5]

        for lid in layer_ids:
            try:
                layer = graph.get_layer(lid)
            except Exception:
                continue

            lname = layer_name_from_id(int(lid))

            # NOTE: Spark-DSG python uses "layer.nodes" (iterable), not "layer.nodes()"
            try:
                for n in layer.nodes:
                    try:
                        raw_id = int(n.id.value)
                    except Exception:
                        try:
                            raw_id = int(n.id)
                        except Exception:
                            continue

                    node_id = f"{lname}_{raw_id}"

                    # position
                    try:
                        p = n.attributes.position
                        pos = [float(p.x), float(p.y), float(p.z)]
                    except Exception:
                        continue

                    # semantic name/label (chair/table/etc.)
                    nm = ""
                    for attr in ("name", "label", "semantic_label"):
                        try:
                            v = getattr(n.attributes, attr)
                            if isinstance(v, str) and v:
                                nm = v
                                break
                        except Exception:
                            pass

                    nodes[node_id] = {
                        "id": node_id,
                        "name": nm or lname,
                        "pos": pos,
                        "layer": lname,
                    }
            except Exception:
                continue

        # Optional: edges (best-effort; different builds expose different iterators)
        try:
            for e in graph.edges:
                try:
                    s = str(int(e.source.value))
                except Exception:
                    s = str(getattr(e, "source", ""))
                try:
                    d = str(int(e.target.value))
                except Exception:
                    d = str(getattr(e, "target", ""))
                if s and d:
                    edges.append((s, d, "link"))
        except Exception:
            pass

        # Map raw numeric edge endpoints -> prefixed node ids
        raw_to_node_ids: Dict[str, List[str]] = {}
        for nid in nodes.keys():
            parts = nid.split("_", 1)
            if len(parts) == 2:
                raw_to_node_ids.setdefault(parts[1], []).append(nid)

        def resolve_endpoint(x: str) -> str:
            if x in nodes:
                return x
            cands = raw_to_node_ids.get(str(x), [])
            if not cands:
                return ""
            return _pick_best_layer(cands)

        fixed_edges: List[Tuple[str, str, str]] = []
        for s, d, t in edges:
            s2 = resolve_endpoint(s)
            d2 = resolve_endpoint(d)
            if s2 and d2:
                fixed_edges.append((s2, d2, "link"))
        edges = fixed_edges

        self.get_logger().info(f"[DSG] Publishing snapshot nodes={len(nodes)} edges={len(edges)}")

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
