from collections import deque
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseArray, Pose

class FrontierNode(Node):
    def __init__(self):
        super().__init__('frontiers')

        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('world_frame', 'map')
        self.declare_parameter('cluster_min_size', 30)

        self._map_topic = self.get_parameter('map_topic').value
        self._world_frame = self.get_parameter('world_frame').value
        self._cluster_min_size = int(self.get_parameter('cluster_min_size').value)

        self._pub = self.create_publisher(PoseArray, '/grapheqa/frontiers', 10)
        self._sub = self.create_subscription(OccupancyGrid, self._map_topic, self._on_map, 10)

        self.get_logger().info(f"Listening map: {self._map_topic} publishing /grapheqa/frontiers")

    def _on_map(self, msg: OccupancyGrid):
        w = msg.info.width
        h = msg.info.height
        res = msg.info.resolution
        ox = msg.info.origin.position.x
        oy = msg.info.origin.position.y

        data = msg.data  # row-major

        def idx(x, y): return y * w + x

        # frontier cell: free (0) adjacent to unknown (-1)
        frontier = [[False]*w for _ in range(h)]
        for y in range(1, h-1):
            for x in range(1, w-1):
                v = data[idx(x, y)]
                if v != 0:
                    continue
                # check 4-neighbors for unknown
                if (data[idx(x+1, y)] == -1 or data[idx(x-1, y)] == -1 or
                    data[idx(x, y+1)] == -1 or data[idx(x, y-1)] == -1):
                    frontier[y][x] = True

        visited = [[False]*w for _ in range(h)]
        clusters: List[List[Tuple[int,int]]] = []

        for y in range(h):
            for x in range(w):
                if not frontier[y][x] or visited[y][x]:
                    continue
                q = deque([(x, y)])
                visited[y][x] = True
                cells = []
                while q:
                    cx, cy = q.popleft()
                    cells.append((cx, cy))
                    for nx, ny in [(cx+1,cy),(cx-1,cy),(cx,cy+1),(cx,cy-1)]:
                        if 0 <= nx < w and 0 <= ny < h and frontier[ny][nx] and not visited[ny][nx]:
                            visited[ny][nx] = True
                            q.append((nx, ny))
                if len(cells) >= self._cluster_min_size:
                    clusters.append(cells)

        out = PoseArray()
        out.header.frame_id = self._world_frame
        out.header.stamp = msg.header.stamp

        for cells in clusters:
            mx = sum(c[0] for c in cells) / len(cells)
            my = sum(c[1] for c in cells) / len(cells)
            wx = ox + (mx + 0.5) * res
            wy = oy + (my + 0.5) * res

            p = Pose()
            p.position.x = float(wx)
            p.position.y = float(wy)
            p.position.z = 0.0
            p.orientation.w = 1.0
            out.poses.append(p)

        self._pub.publish(out)

def main():
    rclpy.init()
    node = FrontierNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
