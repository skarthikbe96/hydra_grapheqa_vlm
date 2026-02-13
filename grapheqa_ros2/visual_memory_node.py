import os
import time

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

        self._topic = self.get_parameter('image_topic').value
        self._out_dir = self.get_parameter('output_dir').value
        self._save_period = 1.0 / float(self.get_parameter('save_hz').value)

        os.makedirs(self._out_dir, exist_ok=True)

        self._bridge = CvBridge()
        self._last_save_t = 0.0
        self._last_path = ''

        self._pub = self.create_publisher(String, '/grapheqa/latest_image_path', 10)
        self._sub = self.create_subscription(Image, self._topic, self._on_img, 10)

        self.get_logger().info(f"Saving images from {self._topic} into {self._out_dir}")

    def _on_img(self, msg: Image):
        now = time.time()
        if (now - self._last_save_t) < self._save_period:
            return

        cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        fname = f"current_{int(now*1000)}.jpg"
        path = os.path.join(self._out_dir, fname)
        cv2.imwrite(path, cv_img)

        self._last_save_t = now
        self._last_path = path

        out = String()
        out.data = path
        self._pub.publish(out)

def main():
    rclpy.init()
    node = VisualMemoryNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
