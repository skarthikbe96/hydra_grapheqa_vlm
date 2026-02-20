#!/usr/bin/env python3
"""
grapheqa_ros2/vlm_planner_bridge.py

Drop-in corrected + very verbose debug version.

What it does:
- Runs as a ROS2 node in the ROS python env
- Spawns grapheqa_infer_server.py using a conda env python
- Talks JSONL over stdin/stdout
- Subscribes to:
    /grapheqa/latest_image_path  (std_msgs/String)
    /grapheqa/dsg_snapshot       (std_msgs/String)
- Publishes:
    /grapheqa/plan_result        (std_msgs/String)  (JSON string with server response)

Notes:
- Keeps ALL debug logs (you can later downgrade to .debug())
- Uses unbuffered python (-u) and PYTHONUNBUFFERED=1
- Robust to server stdout chatter, reads JSON-only lines
"""

import json
import os
import subprocess
import threading
import queue
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseArray


@dataclass
class PendingReq:
    req: Dict[str, Any]
    t0: float
    rid: int


class GrapheqaVLMPlannerBridge(Node):
    def __init__(self):
        super().__init__("grapheqa_vlm_planner_bridge")

        self.get_logger().info("[BRIDGE] Initializing VLM Planner Bridge")

        # ---------- parameters ----------
        # IMPORTANT: conda_base must be the anaconda root, not the conda binary
        self.conda_base = self.declare_parameter("conda_base", "/home/rebellion/anaconda3").value
        self.conda_env = self.declare_parameter("conda_env", "grapheqa").value
        self.env_python = self.declare_parameter("env_python", "").value  # optional override

        self.server_script = self.declare_parameter(
            "server_script",
            "/home/rebellion/mobile_robotics/graph_eqa_ws/src/grapheqa_ros2/grapheqa_ros2/grapheqa_infer_server.py",
        ).value

        self.output_dir = self.declare_parameter("output_dir", "/tmp/grapheqa_images").value
        self.vlm_model_name = self.declare_parameter("vlm_model_name", "gpt-4o-mini").value
        self.use_image = bool(self.declare_parameter("use_image", True).value)
        self.add_history = bool(self.declare_parameter("add_history", True).value)

        self.question = self.declare_parameter("question", "Find the fridge in the kitchen and navigate near it").value
        self.choices = self.declare_parameter(
            "choices", ["kitchen", "living room", "bedroom", "bathroom"]
        ).value

        self.timeout_s = float(self.declare_parameter("server_timeout_s", 20.0).value)

        self.result_topic = self.declare_parameter("result_topic", "/grapheqa/plan_result").value
        self.result_pub = self.create_publisher(String, self.result_topic, 10)

        self.run_hz = float(self.declare_parameter("run_hz", 0.5).value)

        # ---------- internal state ----------
        self._latest_image_path: Optional[str] = None
        self._latest_dsg: Optional[str] = None
        self._rid = 0

        self._req_q: "queue.Queue[PendingReq]" = queue.Queue(maxsize=10)
        self._resp_q: "queue.Queue[dict]" = queue.Queue(maxsize=10)

        self._proc: Optional[subprocess.Popen] = None
        self._proc_lock = threading.Lock()

        # ---------- subscriptions ----------
        self.create_subscription(String, "/grapheqa/latest_image_path", self._on_image_path, 10)
        self.create_subscription(String, "/grapheqa/dsg_snapshot", self._on_dsg, 10)

        # ---------- start server + IO thread ----------
        self._start_server()

        self._io_thread = threading.Thread(target=self._io_loop, daemon=True)
        self._io_thread.start()

        # ---------- tick timer ----------
        self.timer = self.create_timer(1.0 / max(0.1, self.run_hz), self._tick)

        self.get_logger().info("[BRIDGE] GraphEQA VLM planner bridge ready.")

        self._latest_frontiers = []  # IMPORTANT: initialize so tick won't crash

        self.create_subscription(PoseArray, "/grapheqa/frontiers", self._on_frontiers, 10)
       
        
    def _on_frontiers(self, msg: PoseArray):
        # store as a simple list of xyz dicts (JSON-friendly)
        self._latest_frontiers = [
            {"x": p.position.x, "y": p.position.y, "z": p.position.z}
            for p in msg.poses
        ]


    # ===================== ROS callbacks =====================
    def _on_image_path(self, msg: String):
        self._latest_image_path = msg.data
        exists = os.path.exists(msg.data)
        self.get_logger().info(f"[BRIDGE] Got image path: {msg.data} exists={exists}")

    def _on_dsg(self, msg: String):
        self._latest_dsg = msg.data
        self.get_logger().info(f"[BRIDGE] Got DSG snapshot: {len(msg.data)} bytes")

    # ===================== server process helpers =====================
    def _pick_python(self) -> str:
        py = (self.env_python or "").strip()
        if py:
            py = os.path.expanduser(py)
            return py

        # conda_base must be root like /home/rebellion/anaconda3
        conda_base = os.path.expanduser(self.conda_base)
        return os.path.join(conda_base, "envs", self.conda_env, "bin", "python")

    def _start_server(self):
        if not self.server_script:
            raise RuntimeError("Parameter 'server_script' must be set to absolute path of grapheqa_infer_server.py")

        py = self._pick_python()

        self.get_logger().info(f"[BRIDGE] Launching infer server with python={py}")
        self.get_logger().info(f"[BRIDGE] Server script: {self.server_script}")

        if not os.path.isfile(py):
            raise RuntimeError(f"Env python not found: {py}")
        if not os.path.isfile(self.server_script):
            raise RuntimeError(f"Server script not found: {self.server_script}")

        cmd = [
            py, "-u",
            self.server_script,
            "--vlm_model_name", str(self.vlm_model_name),
        ]
        if self.use_image:
            cmd.append("--use_image")
        if self.add_history:
            cmd.append("--add_history")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        self.get_logger().info("[BRIDGE] Starting GraphEQA infer server cmd:\n  " + " ".join(cmd))

        with self._proc_lock:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
            )

        self.get_logger().info("[BRIDGE] Infer server process started")

        # drain stderr so server logs show up in ROS logs
        def _drain_stderr(p: subprocess.Popen):
            assert p.stderr is not None
            for line in p.stderr:
                self.get_logger().info("[SERVER STDERR] " + line.rstrip())

        threading.Thread(target=_drain_stderr, args=(self._proc,), daemon=True).start()

        ready = self._read_server_json_line()
        self.get_logger().info(f"[BRIDGE] Ready handshake: {ready}")
        if not ready.get("ok", False) or ready.get("type") != "ready":
            raise RuntimeError(f"GraphEQA infer server did not start correctly. Got: {ready}")

    def _read_server_json_line(self, max_lines: int = 500) -> dict:
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("Infer server not running.")

        for _ in range(max_lines):
            with self._proc_lock:
                line = self._proc.stdout.readline()

            if not line:
                raise RuntimeError("Infer server stdout closed.")

            s = line.strip()
            if not s:
                continue

            # show everything while debugging
            self.get_logger().info("[BRIDGE] server stdout: " + s[:300])

            if not s.startswith("{"):
                continue

            try:
                return json.loads(s)
            except json.JSONDecodeError:
                self.get_logger().warn("[BRIDGE] server stdout bad json: " + s[:300])
                continue

        raise RuntimeError("Infer server did not produce JSON within max_lines.")

    def _server_send(self, obj: dict):
        with self._proc_lock:
            if self._proc is None or self._proc.stdin is None:
                raise RuntimeError("Infer server not running.")
            self._proc.stdin.write(json.dumps(obj) + "\n")
            self._proc.stdin.flush()

    def _server_recv(self, max_lines: int = 500) -> dict:
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("Infer server not running.")

        for _ in range(max_lines):
            with self._proc_lock:
                line = self._proc.stdout.readline()

            if not line:
                raise RuntimeError("Infer server stdout closed.")

            self.get_logger().info("[BRIDGE] Raw server stdout: " + line.rstrip())

            s = line.strip()
            if not s:
                continue

            if not s.startswith("{"):
                continue

            try:
                return json.loads(s)
            except json.JSONDecodeError:
                self.get_logger().warn("[BRIDGE] bad json: " + s[:300])
                continue

        raise RuntimeError("Infer server did not produce JSON within max_lines.")

    def _proc_alive(self) -> bool:
        p = getattr(self, "_proc", None)
        return p is not None and (p.poll() is None)

    def _restart_server(self, why: str):
        self.get_logger().error(f"[BRIDGE] Restarting infer server: {why}")
        try:
            with self._proc_lock:
                if self._proc is not None:
                    self._proc.terminate()
        except Exception as e:
            self.get_logger().warn(f"[BRIDGE] terminate failed: {e}")

        time.sleep(0.3)
        self._start_server()

    def _io_loop(self):
        self.get_logger().info("[BRIDGE] IO thread started")
        while rclpy.ok():
            try:
                pending = self._req_q.get(timeout=0.2)
            except queue.Empty:
                if not self._proc_alive():
                    self._restart_server("process exited")
                continue

            if not self._proc_alive():
                self._restart_server("process not alive before request")

            try:
                self.get_logger().info(f"[BRIDGE] (rid={pending.rid}) sending to server")
                self._server_send(pending.req)

                resp = self._server_recv()
                resp["_rid"] = pending.rid
                resp["_latency_s"] = time.time() - pending.t0

                self._resp_q.put(resp, timeout=0.2)

            except Exception as e:
                self.get_logger().error(f"[BRIDGE] IPC exception: {e}")
                try:
                    self._resp_q.put({"ok": False, "error": f"ipc_exception:{e}", "_rid": pending.rid})
                except Exception:
                    pass
                self._restart_server(str(e))

    # ===================== periodic tick =====================
    def _tick(self):
        self.get_logger().info("[BRIDGE] Tick triggered")

        have_image = self._latest_image_path is not None
        have_dsg = self._latest_dsg is not None

        self.get_logger().info(f"[BRIDGE] State: have_image={have_image} have_dsg={have_dsg}")

        if not have_image:
            self.get_logger().warn("[BRIDGE] Tick skipped: no latest_image_path yet")
            return
        if not have_dsg:
            self.get_logger().warn("[BRIDGE] Tick skipped: no dsg_snapshot yet")
            return

        # build request
        self._rid += 1
        rid = self._rid

        req = {
            "type": "plan",
            "question": str(self.question),
            "choices": list(self.choices) if isinstance(self.choices, (list, tuple)) else [],
            "images_dir": str(self.output_dir),
            "context": {
                "latest_image_path": self._latest_image_path,
                "dsg_snapshot": self._latest_dsg,
                "frontiers": self._latest_frontiers,     # list of frontier points/clusters
                "confidence_threshold": 0.7,             # ser
            },
        }

        self.get_logger().info("[BRIDGE] Sending request:\n" + json.dumps(req, indent=2))

        try:
            self._req_q.put(PendingReq(req=req, t0=time.time(), rid=rid), timeout=0.05)
        except queue.Full:
            self.get_logger().warn("[BRIDGE] Request queue full; dropping request.")
            return

        try:
            resp = self._resp_q.get(timeout=self.timeout_s)
        except queue.Empty:
            self.get_logger().error("[BRIDGE] Timeout waiting for response")
            self._restart_server("timeout")
            return

        self.get_logger().info(f"[BRIDGE] Got response rid={resp.get('_rid')} ok={resp.get('ok')} latency={resp.get('_latency_s')}")

        if not resp.get("ok", False):
            self.get_logger().error(f"[BRIDGE] Server error: {resp.get('error')}")
            tb = resp.get("traceback")
            if tb:
                self.get_logger().error("[BRIDGE] Server traceback:\n" + tb)
            return

        # publish
        out = String()
        out.data = json.dumps(resp)
        self.get_logger().info("[BRIDGE] Publishing plan_result on " + self.result_topic)
        self.result_pub.publish(out)

    # ===================== lifecycle =====================
    def destroy_node(self):
        self.get_logger().info("[BRIDGE] Shutting down bridge")
        try:
            with self._proc_lock:
                if self._proc is not None:
                    self._proc.terminate()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = GrapheqaVLMPlannerBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()