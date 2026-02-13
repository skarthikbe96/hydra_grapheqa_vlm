# grapheqa_ros2/sg_ros_adapter.py

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx


@dataclass
class NodeData:
    node_id: str
    name: str
    pos: Tuple[float, float, float]
    layer: str  # 'room' | 'region' | 'object' | 'agent' | 'frontier'


def _dist(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


class SceneGraphRosAdapter:
    """
    Minimal compatibility layer to satisfy GraphEQA VLM planners.
    It stores a compact DiGraph + provides the properties/methods the planner expects.
    """

    def __init__(self, frontier_to_object_radius_m: float = 2.0):
        self._g = nx.DiGraph()
        self._frontier_to_object_radius_m = float(frontier_to_object_radius_m)
        self._agent_id: Optional[str] = None

        # Planner-facing fields
        self.object_node_ids: List[str] = []
        self.object_node_names: List[str] = []
        self.frontier_node_ids: List[str] = []
        self.room_node_ids: List[str] = []
        self.room_node_names: List[str] = []
        self.region_node_ids: List[str] = []

    def update_from_snapshot(
        self,
        nodes: Dict[str, NodeData],
        edges: List[Tuple[str, str, str]],
        frontier_points_map: List[Tuple[float, float, float]],
    ) -> None:
        """
        nodes: node_id -> NodeData (already in map frame)
        edges: list of (src_id, dst_id, edge_type)
        frontier_points_map: list of (x,y,z) in map frame
        """
        self._g = nx.DiGraph()
        self._agent_id = None

        # Add DSG nodes
        for nid, nd in nodes.items():
            self._g.add_node(nid, name=nd.name, pos=tuple(nd.pos), layer=nd.layer)
            if nd.layer == "agent":
                self._agent_id = nid

        # Add DSG edges
        for src, dst, etype in edges:
            if src in self._g.nodes and dst in self._g.nodes:
                self._g.add_edge(src, dst, type=etype)

        # Add frontier nodes
        self.frontier_node_ids = []
        for i, p in enumerate(frontier_points_map):
            fid = f"frontier_{i}"
            self.frontier_node_ids.append(fid)
            self._g.add_node(fid, name=fid, pos=tuple(p), layer="frontier")

        # Enrich: connect frontier -> nearby objects
        obj_ids = [nid for nid in self._g.nodes if self._g.nodes[nid].get("layer") == "object"]
        for fid in self.frontier_node_ids:
            fpos = self._g.nodes[fid]["pos"]
            for oid in obj_ids:
                opos = self._g.nodes[oid]["pos"]
                if _dist(fpos, opos) <= self._frontier_to_object_radius_m:
                    self._g.add_edge(fid, oid, type="near")

        # Refresh planner-facing lists
        self.object_node_ids = []
        self.object_node_names = []
        self.room_node_ids = []
        self.room_node_names = []
        self.region_node_ids = []

        for nid in self._g.nodes:
            layer = self._g.nodes[nid].get("layer")
            name = self._g.nodes[nid].get("name", nid)
            if layer == "object":
                self.object_node_ids.append(nid)
                self.object_node_names.append(name)
            elif layer == "room":
                self.room_node_ids.append(nid)
                self.room_node_names.append(name)
            elif layer == "region":
                self.region_node_ids.append(nid)

    @property
    def scene_graph_str(self) -> str:
        # Compact graph encoding that is stable + works well in prompts
        payload = nx.node_link_data(self._g)
        return json.dumps(payload)

    def get_position_from_id(self, node_id: str) -> Tuple[float, float, float]:
        if node_id not in self._g.nodes:
            raise KeyError(f"Unknown node_id: {node_id}")
        return tuple(self._g.nodes[node_id]["pos"])

    def get_current_semantic_state_str(self) -> str:
        """
        GraphEQA expects a short string describing the agent and (optionally) its room.
        We try to infer a room; otherwise we just provide pose.
        """
        if not self._agent_id or self._agent_id not in self._g.nodes:
            return "agent_id: unknown"

        apos = self._g.nodes[self._agent_id]["pos"]

        # Try infer room by hierarchy (room -> region -> agent) if present
        room_id = None
        room_name = None

        preds = list(self._g.predecessors(self._agent_id))
        # predecessor region?
        for p in preds:
            if self._g.nodes[p].get("layer") == "region":
                for pp in self._g.predecessors(p):
                    if self._g.nodes[pp].get("layer") == "room":
                        room_id = pp
                        room_name = self._g.nodes[pp].get("name", pp)
                        break
            if room_id:
                break

        # Fallback: nearest room centroid if rooms exist
        if not room_id and self.room_node_ids:
            best = None
            bestd = 1e18
            for rid in self.room_node_ids:
                rpos = self._g.nodes[rid]["pos"]
                d = _dist(apos, rpos)
                if d < bestd:
                    bestd = d
                    best = rid
            if best:
                room_id = best
                room_name = self._g.nodes[best].get("name", best)

        return f"agent_id: {self._agent_id}, location(map): {apos}, room_id: {room_id}, room_name: {room_name}"
