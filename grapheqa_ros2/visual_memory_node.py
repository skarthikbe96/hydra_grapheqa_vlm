#!/usr/bin/env python3

import os
import time
import glob
import re

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from cv_bridge import CvBridge
import cv2


class VisualMemoryNode(Node):
    def __init__(self):
        super().__init__('visual_memory')

        self.declare_parameter('image_topic', '/camera_pan_tilt/image')
        self.declare_parameter('output_dir', '/tmp/grapheqa_images')
        self.declare_parameter('save_hz', 1.0)
        self.declare_parameter('max_images', 200)

        self._topic = self.get_parameter('image_topic').value
        self._out_dir = self.get_parameter('output_dir').value
        self._save_period = 1.0 / float(self.get_parameter('save_hz').value)
        self._max_images = int(self.get_parameter('max_images').value)

        os.makedirs(self._out_dir, exist_ok=True)

        self._bridge = CvBridge()
        self._last_save_t = 0.0
        self._counter = self._init_counter_from_disk()

        self._pub = self.create_publisher(String, '/grapheqa/latest_image_path', 10)
        self._sub = self.create_subscription(Image, self._topic, self._on_img, 10)

        self.get_logger().info(
            f"Saving images from {self._topic} into {self._out_dir} as current_img_<i>.png"
        )

    def _init_counter_from_disk(self) -> int:
        """
        Continue numbering from existing current_img_*.png if present.
        """
        files = glob.glob(os.path.join(self._out_dir, "current_img_*.png"))
        if not files:
            return 0
        max_i = -1
        for f in files:
            m = re.search(r"current_img_(\d+)\.png$", os.path.basename(f))
            if m:
                max_i = max(max_i, int(m.group(1)))
        return max_i + 1 if max_i >= 0 else 0

    def _prune(self):
        """
        Keep only the latest max_images by index.
        """
        files = sorted(glob.glob(os.path.join(self._out_dir, "current_img_*.png")))
        if len(files) <= self._max_images:
            return
        for f in files[: len(files) - self._max_images]:
            try:
                os.remove(f)
            except Exception:
                pass

    def _on_img(self, msg: Image):
        now = time.time()

        # throttle saving
        if now - self._last_save_t < self._save_period:
            return
        self._last_save_t = now

        # convert
        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"[VM] cv_bridge failed: {e}")
            return

        # save with GraphEQA-compatible name
        fname = f"current_img_{self._counter}.png"
        path = os.path.join(self._out_dir, fname)

        ok = cv2.imwrite(path, cv_img)
        if not ok:
            self.get_logger().error(f"[VM] cv2.imwrite failed: {path}")
            return

        self._counter += 1

        # publish latest path (optional, but useful for debugging)
        self._pub.publish(String(data=path))

        # prune old images
        self._prune()


def main():
    rclpy.init()
    node = VisualMemoryNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
