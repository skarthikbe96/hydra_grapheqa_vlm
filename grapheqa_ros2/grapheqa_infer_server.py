#!/usr/bin/env python3
"""
JSONL stdin/stdout infer server that runs inside a Conda env.

Protocol:
- On startup prints: {"type":"ready","ok":true}
- Reads one JSON object per line from stdin
- For each request prints one JSON object per line to stdout

Request:
{"type":"plan","question":"...","choices":["kitchen","living room"],"images_dir":"/tmp/grapheqa_images","context":{...}}

Response:
{"type":"plan_result","ok":true,"plan":{"answer":"living room","goal":null,"debug":{...}}}

or (exploration step):
{"type":"plan_result","ok":true,"plan":{"answer":"","goal":{"frame":"map","x":1.0,"y":2.0,"yaw":0.0},"debug":{...}}}
"""
import argparse
import json
import math
import os
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _print_json(obj: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _load_latest_image_paths(images_dir: str, k: int = 5) -> List[str]:
    if not images_dir or not os.path.isdir(images_dir):
        return []
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

    pairs: List[Tuple[float, str]] = []
    for fn in os.listdir(images_dir):
        if not fn.lower().endswith(exts):
            continue
        p = os.path.join(images_dir, fn)
        try:
            pairs.append((os.path.getmtime(p), p))
        except FileNotFoundError:
            continue

    pairs.sort()
    return [p for _, p in pairs][-k:]


def _safe_json_loads(s: Any) -> Optional[dict]:
    if not isinstance(s, str) or not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def _yaw_to_quat_z_w(yaw: float) -> Tuple[float, float]:
    # for debug only; executor likely uses yaw directly
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


@dataclass
class Goal:
    frame: str
    x: float
    y: float
    yaw: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"frame": self.frame, "x": float(self.x), "y": float(self.y), "yaw": float(self.yaw)}


class GraphEQAServer:
    """
    Drop-in server that can:
    - answer (placeholder)
    - or output a navigation goal (frontier-driven), which is required for "embodied" behavior.
    """

    def __init__(self, vlm_model_name: str, use_image: bool, add_history: bool):
        self.vlm_model_name = vlm_model_name
        self.use_image = use_image
        self.add_history = add_history

        # You will replace these with real GraphEQA imports later.
        self.real_planner = None

        print("[SERVER] Infer server started", file=sys.stderr, flush=True)

    # ---------- Core helpers ----------

    def _pick_frontier_goal(self, frontiers: List[Dict[str, Any]], world_frame: str = "map") -> Optional[Goal]:
        """
        Choose a frontier goal.
        Expecting something like:
          frontiers = [{"x":..., "y":...}, ...]
        or
          frontiers = [{"centroid":[x,y], ...}, ...]
        """
        if not frontiers:
            return None

        # Simple heuristic: pick the first valid frontier.
        for f in frontiers:
            if isinstance(f, dict):
                if "x" in f and "y" in f:
                    return Goal(frame=world_frame, x=float(f["x"]), y=float(f["y"]), yaw=0.0)
                c = f.get("centroid")
                if isinstance(c, (list, tuple)) and len(c) >= 2:
                    return Goal(frame=world_frame, x=float(c[0]), y=float(c[1]), yaw=0.0)

        return None

    def _vlm_answer_placeholder(self, question: str, choices: List[str], image_paths: List[str], dsg: Optional[dict]) -> Tuple[str, float]:
        """
        Placeholder for your real VLM+GraphEQA reasoning.

        Returns:
          (answer_choice_value, confidence in [0,1])
        """
        # CURRENT BEHAVIOR IN YOUR CODE WAS: always "A".
        # This placeholder does something slightly less wrong:
        # - if 'living' appears in question or choices, bias to 'living room'
        # Replace with actual model call.
        q = (question or "").lower()

        if not choices:
            return "", 0.0

        # naive heuristic example
        for c in choices:
            if "living" in c.lower() and ("couch" in q or "sofa" in q):
                return c, 0.55

        # fallback: first choice with low confidence
        return choices[0], 0.25

    # ---------- Main plan ----------

    def plan(self, question: str, choices: List[str], images_dir: str, context: Dict[str, Any]) -> Dict[str, Any]:
        image_paths = _load_latest_image_paths(images_dir, k=5) if self.use_image else []

        # Parse the DSG snapshot that your bridge sends as a JSON STRING.
        dsg_snapshot_str = context.get("dsg_snapshot")
        dsg = _safe_json_loads(dsg_snapshot_str)

        world_frame = "map"
        if isinstance(dsg, dict) and isinstance(dsg.get("frame_id"), str):
            world_frame = dsg["frame_id"]

        # OPTIONAL: if you later add frontiers into context, we can use them.
        # Example desired context field:
        # context["frontiers"] = [{"x":..., "y":...}, ...]
        frontiers = context.get("frontiers")
        if not isinstance(frontiers, list):
            frontiers = []

        # --- Step 1: Try to answer (placeholder) ---
        answer, conf = self._vlm_answer_placeholder(
            question=question,
            choices=choices,
            image_paths=image_paths,
            dsg=dsg,
        )

        # --- Step 2: Confidence gating: if not confident, output a goal ---
        # This is the key missing behavior in your current implementation.
        confidence_threshold = float(context.get("confidence_threshold", 0.7))

        goal: Optional[Goal] = None
        if conf < confidence_threshold:
            goal = self._pick_frontier_goal(frontiers=frontiers, world_frame=world_frame)

        # If we found a goal, we *do not* answer yet (embodied step).
        # If we didn't find a goal, we answer anyway (best-effort).
        if goal is not None:
            final_answer = ""
            final_goal = goal.to_dict()
        else:
            final_answer = answer
            final_goal = None

        # logs
        print(f"[SERVER] Using {len(image_paths)} images", file=sys.stderr, flush=True)
        for p in image_paths[-3:]:
            print(f"[SERVER]   {p}", file=sys.stderr, flush=True)

        if goal is not None:
            print(f"[SERVER] Not confident (conf={conf:.2f} < {confidence_threshold:.2f}); sending GOAL {final_goal}",
                  file=sys.stderr, flush=True)
        else:
            print(f"[SERVER] Answering (conf={conf:.2f} >= {confidence_threshold:.2f} or no goal): {final_answer}",
                  file=sys.stderr, flush=True)

        return {
            "answer": final_answer,
            "goal": final_goal,
            "debug": {
                "vlm_model_name": self.vlm_model_name,
                "use_image": self.use_image,
                "add_history": self.add_history,
                "confidence": conf,
                "confidence_threshold": confidence_threshold,
                "num_images_used": len(image_paths),
                "images_used": image_paths[-2:],  # keep small
                "have_dsg": bool(dsg),
                "frontiers_in_context": len(frontiers),
            },
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm_model_name", default="gpt-4o-mini")
    ap.add_argument("--use_image", action="store_true")
    ap.add_argument("--add_history", action="store_true")
    args = ap.parse_args()

    try:
        server = GraphEQAServer(
            vlm_model_name=args.vlm_model_name,
            use_image=bool(args.use_image),
            add_history=bool(args.add_history),
        )
    except Exception:
        _print_json(
            {
                "type": "ready",
                "ok": False,
                "error": "Failed to initialize server",
                "traceback": traceback.format_exc(),
            }
        )
        return 2

    _print_json({"type": "ready", "ok": True})

    for line in sys.stdin:
        s = (line or "").strip()
        if not s:
            continue

        try:
            req = json.loads(s)
        except json.JSONDecodeError:
            _print_json({"type": "error", "ok": False, "error": "bad_json"})
            continue

        print("[SERVER] Received request:\n" + json.dumps(req, indent=2),
              file=sys.stderr, flush=True)

        rtype = req.get("type", "")
        try:
            if rtype == "ping":
                _print_json({"type": "pong", "ok": True})
                continue

            if rtype != "plan":
                _print_json({"type": "error", "ok": False, "error": f"unknown_type:{rtype}"})
                continue

            question = (req.get("question") or "").strip()
            choices = req.get("choices") or []
            images_dir = req.get("images_dir") or ""
            context = req.get("context") or {}

            plan = server.plan(question=question, choices=choices, images_dir=images_dir, context=context)

            _print_json({"type": "plan_result", "ok": True, "plan": plan})
            print("[SERVER] Sent plan_result", file=sys.stderr, flush=True)

        except Exception:
            _print_json(
                {
                    "type": "plan_result",
                    "ok": False,
                    "error": "exception",
                    "traceback": traceback.format_exc(),
                }
            )
            print("[SERVER] Exception occurred (see traceback in JSON)",
                  file=sys.stderr, flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())