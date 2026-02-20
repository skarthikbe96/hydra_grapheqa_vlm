#!/usr/bin/env python3
# grapheqa_ros2/dsg_bridge_node.py

import json
import os
import sys
import glob
from typing import Dict, List, Tuple, Optional, Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

try:
    from hydra_msgs.msg import DsgUpdate
except Exception:
    DsgUpdate = None


_LAYER_PRIORITY = ["object", "place", "room", "region", "building", "agent", "frontier"]


def _pick_best_layer(node_ids: List[str]) -> str:
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
    added_any = False

    here = os.path.abspath(__file__)
    install_root = here
    for _ in range(10):
        install_root = os.path.dirname(install_root)
        if os.path.basename(install_root) == "install":
            break

    candidate_prefixes = []
    if os.path.basename(install_root) == "install":
        candidate_prefixes.append(os.path.join(install_root, "spark_dsg"))

    for envvar in ("AMENT_PREFIX_PATH", "CMAKE_PREFIX_PATH"):
        val = os.environ.get(envvar, "")
        for p in val.split(os.pathsep):
            p = p.strip()
            if not p:
                continue
            candidate_prefixes.append(p)
            candidate_prefixes.append(os.path.join(p, "spark_dsg"))

    seen = set()
    prefixes = []
    for p in candidate_prefixes:
        p = os.path.normpath(p)
        if p in seen:
            continue
        seen.add(p)
        prefixes.append(p)

    for prefix in prefixes:
        if not os.path.isdir(prefix):
            continue

        for sp in glob.glob(os.path.join(prefix, "lib", "python*", "site-packages")):
            if sp not in sys.path:
                sys.path.insert(0, sp)
                added_any = True

        py_dir = os.path.join(prefix, "python")
        if os.path.isdir(py_dir) and py_dir not in sys.path:
            sys.path.insert(0, py_dir)
            added_any = True

    if added_any:
        logger.info("[DSG] Patched sys.path to search Spark-DSG bindings.")
    return added_any


def _try_import_spark_dsg(logger):
    try:
        import spark_dsg as sdsg  # noqa: F401
        logger.info(f"[DSG] spark_dsg imported from: {sdsg.__file__}")
        return sdsg
    except Exception as e1:
        logger.error(f"[DSG] spark_dsg import failed: {type(e1).__name__}: {e1}")

    patched = _patch_sys_path_for_spark_dsg(logger)
    if not patched:
        logger.error("[DSG] No spark_dsg python site-packages folder found to patch sys.path.")
        return None

    try:
        import spark_dsg as sdsg  # noqa: F401
        logger.info(f"[DSG] spark_dsg import succeeded after sys.path patch: {sdsg.__file__}")
        return sdsg
    except Exception as e2:
        logger.error(f"[DSG] spark_dsg import still failed after patch: {type(e2).__name__}: {e2}")
        return None


class DsgBridgeNode(Node):
    """
    Subscribes to Hydra DSG (/hydra/backend/dsg) and publishes JSON snapshot on:
      /grapheqa/dsg_snapshot (std_msgs/String)

    Key detail:
      In your spark_dsg build, graph.layers() returns LayerView objects (not ids).
      So we must iterate LayerView directly.
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

        self._sdsg = _try_import_spark_dsg(self.get_logger())
        self._warned_no_sdsg = False

        self._graph = None

    def _publish_agent_only_snapshot(self):
        if self._last_snapshot_json is not None:
            return
        snap = self._build_snapshot(nodes={}, edges=[])
        self._pub.publish(String(data=json.dumps(snap)))

    def _build_snapshot(self, nodes: Dict, edges: List[Tuple[str, str, str]]) -> dict:
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

    @staticmethod
    def _layer_name_guess(layer_id: int) -> str:
        mapping = {
            1: "mesh",
            2: "object",
            3: "place",
            4: "room",
            5: "building",
            6: "agent",
            7: "frontier",
        }
        return mapping.get(int(layer_id), f"layer{int(layer_id)}")

    @staticmethod
    def _node_raw_id(n) -> Optional[int]:
        try:
            return int(n.id.value)
        except Exception:
            pass
        try:
            return int(n.id)
        except Exception:
            pass
        return None

    @staticmethod
    def _vec3_to_list(v) -> Optional[List[float]]:
        try:
            return [float(v.x), float(v.y), float(v.z)]
        except Exception:
            pass
        try:
            arr = list(v)
            if len(arr) >= 3:
                return [float(arr[0]), float(arr[1]), float(arr[2])]
        except Exception:
            pass
        return None

    def _node_pos(self, n) -> Optional[List[float]]:
        a = getattr(n, "attributes", None)
        if a is None:
            return None

        for path in [
            ("position",),
            ("world_pose", "position"),
            ("pose", "position"),
            ("centroid",),
            ("center",),
            ("bbox", "center"),
            ("bounding_box", "center"),
            ("aabb", "center"),
        ]:
            obj = a
            ok = True
            for key in path:
                if not hasattr(obj, key):
                    ok = False
                    break
                obj = getattr(obj, key)
            if not ok:
                continue
            v = self._vec3_to_list(obj)
            if v is not None:
                return v
        return None

    @staticmethod
    def _node_name(n, fallback: str) -> str:
        a = getattr(n, "attributes", None)
        if a is None:
            return fallback
        for attr in ("name", "label", "semantic_label", "category", "class_name"):
            try:
                v = getattr(a, attr)
                if isinstance(v, str) and v:
                    return v
            except Exception:
                pass
        return fallback

    def _decode_graph(self, msg, raw_bytes: bytes):
        DSG = getattr(self._sdsg, "DynamicSceneGraph", None)
        if DSG is None:
            from spark_dsg import _dsg_bindings
            DSG = _dsg_bindings.DynamicSceneGraph

        if bool(msg.full_update):
            if hasattr(DSG, "from_binary"):
                self._graph = DSG.from_binary(raw_bytes)
            else:
                self._graph = DSG()
                try:
                    self._graph.update_from_binary(raw_bytes)
                except TypeError:
                    self._graph.update_from_binary(raw_bytes, False)
        else:
            if self._graph is None:
                self._graph = DSG()
            try:
                self._graph.update_from_binary(raw_bytes)
            except TypeError:
                self._graph.update_from_binary(raw_bytes, False)

        return self._graph

    def _iter_layers(self, graph) -> List[Any]:
        # 1) layers() method
        if hasattr(graph, "layers"):
            try:
                layers_obj = graph.layers
                if callable(layers_obj):
                    return list(layers_obj())     # graph.layers()
                else:
                    return list(layers_obj)       # graph.layers (property)
            except Exception:
                pass

        # 2) get_layers() method (some builds)
        if hasattr(graph, "get_layers"):
            try:
                return list(graph.get_layers())
            except Exception:
                pass

        # 3) layer_ids + get_layer fallback
        if hasattr(graph, "layer_ids"):
            try:
                lids = list(graph.layer_ids)
                out = []
                for lid in lids:
                    try:
                        out.append(graph.get_layer(lid))
                    except Exception:
                        pass
                return out
            except Exception:
                pass

        return []

    def _iter_nodes_from_layer(self, layer) -> List[Any]:
        """
        LayerView in your build seems to have `.nodes` iterable.
        """
        if hasattr(layer, "nodes"):
            try:
                return list(layer.nodes)
            except Exception:
                pass
        # fallback
        for fn in ("get_nodes", "nodes_iter"):
            if hasattr(layer, fn):
                try:
                    return list(getattr(layer, fn)())
                except Exception:
                    pass
        return []

    def _layer_id_from_layerview(self, layer, fallback_idx: int) -> int:
        """
        Try to extract numeric layer id from LayerView.
        Common patterns: layer.id or layer.layer_id
        """
        for attr in ("id", "layer_id"):
            if hasattr(layer, attr):
                try:
                    return int(getattr(layer, attr))
                except Exception:
                    pass
        # fallback: map based on order (not perfect but better than nothing)
        return fallback_idx + 1

    def _on_dsg(self, msg):
        if self._sdsg is None:
            if not self._warned_no_sdsg:
                self.get_logger().error(
                    "[DSG] spark_dsg python module not available. "
                    "Source the workspace that installed spark_dsg."
                )
                self._warned_no_sdsg = True
            snapshot = self._build_snapshot(nodes={}, edges=[])
            js = json.dumps(snapshot)
            self._last_snapshot_json = js
            self._pub.publish(String(data=js))
            return

        if not hasattr(msg, "layer_contents") or msg.layer_contents is None:
            snapshot = self._build_snapshot(nodes={}, edges=[])
            js = json.dumps(snapshot)
            self._last_snapshot_json = js
            self._pub.publish(String(data=js))
            return

        try:
            raw_bytes = msg.layer_contents.tobytes() if hasattr(msg.layer_contents, "tobytes") else bytes(msg.layer_contents)
        except Exception as e:
            self.get_logger().error(f"[DSG] Failed to convert layer_contents to bytes: {e}")
            snapshot = self._build_snapshot(nodes={}, edges=[])
            js = json.dumps(snapshot)
            self._last_snapshot_json = js
            self._pub.publish(String(data=js))
            return

        try:
            graph = self._decode_graph(msg, raw_bytes)
        except Exception as e:
            self.get_logger().error(f"[DSG] Spark-DSG decode failed: {e}")
            self.get_logger().error(
                f"[DSG] Debug: layer_contents type={type(msg.layer_contents)} len={len(msg.layer_contents)} full_update={bool(msg.full_update)}"
            )
            snapshot = self._build_snapshot(nodes={}, edges=[])
            js = json.dumps(snapshot)
            self._last_snapshot_json = js
            self._pub.publish(String(data=js))
            return

        nodes: Dict[str, dict] = {}
        edges: List[Tuple[str, str, str]] = []

        layers = self._iter_layers(graph)

        try:
            self.get_logger().info(f"[DSG] graph.layers attr type={type(getattr(graph,'layers',None))} callable={callable(getattr(graph,'layers',None))}")
        except Exception:
            pass
        self.get_logger().info(f"[DSG] layers() -> {layers}")

        # Case A: layers are LayerView objects
        if layers and "LayerView" in str(type(layers[0])):
            for i, layer in enumerate(layers):
                layer_id = self._layer_id_from_layerview(layer, i)
                lname = self._layer_name_guess(layer_id)
                ln_nodes = self._iter_nodes_from_layer(layer)

                self.get_logger().info(f"[DSG] layer {layer_id} ({lname}) nodes={len(ln_nodes)}")

                for n in ln_nodes:
                    rid = self._node_raw_id(n)
                    if rid is None:
                        continue
                    pos = self._node_pos(n)
                    if pos is None:
                        continue
                    node_id = f"{lname}_{rid}"
                    nodes[node_id] = {
                        "id": node_id,
                        "name": self._node_name(n, lname),
                        "pos": pos,
                        "layer": lname,
                    }

        # Case B: layers are numeric IDs
        else:
            for lid in layers:
                try:
                    lname = self._layer_name_guess(int(lid))
                except Exception:
                    lname = str(lid)

                try:
                    layer = graph.get_layer(lid)
                except Exception:
                    continue

                ln_nodes = self._iter_nodes_from_layer(layer)
                self.get_logger().info(f"[DSG] layer {lid} ({lname}) nodes={len(ln_nodes)}")

                for n in ln_nodes:
                    rid = self._node_raw_id(n)
                    if rid is None:
                        continue
                    pos = self._node_pos(n)
                    if pos is None:
                        continue
                    node_id = f"{lname}_{rid}"
                    nodes[node_id] = {
                        "id": node_id,
                        "name": self._node_name(n, lname),
                        "pos": pos,
                        "layer": lname,
                    }

        # Edges: best-effort (may differ per binding)
        try:
            for e in getattr(graph, "edges", []):
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

        # Map raw endpoints -> prefixed ids
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
                fixed_edges.append((s2, d2, t))
        edges = fixed_edges

        self.get_logger().info(f"[DSG] Publishing snapshot nodes={len(nodes)} edges={len(edges)} full_update={bool(msg.full_update)}")

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