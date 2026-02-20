#!/usr/bin/env python3
"""
JSONL stdin/stdout "infer server" that runs inside a Conda env.

Protocol:
- On startup prints: {"type":"ready","ok":true}
- Reads one JSON object per line from stdin
- For each request prints one JSON object per line to stdout

Request example:
{"type":"plan","question":"...","choices":["A","B"],"images_dir":"/tmp/grapheqa_images","context":{...}}

Response example:
{"type":"plan_result","ok":true,"plan":{"answer":"B","goal":{"frame":"map","x":1.0,"y":2.0,"yaw":0.0},"debug":{...}}}
"""
import argparse
import json
import os
import sys
import traceback
from typing import Any, Dict


def _print_json(obj: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _load_latest_image_paths(images_dir: str, k: int = 5):
    if not images_dir or not os.path.isdir(images_dir):
        return []
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

    pairs = []
    for fn in os.listdir(images_dir):
        if not fn.lower().endswith(exts):
            continue
        p = os.path.join(images_dir, fn)
        try:
            pairs.append((os.path.getmtime(p), p))
        except FileNotFoundError:
            continue

    pairs.sort()
    paths = [p for _, p in pairs]
    return paths[-k:]



class GraphEQAServer:
    """
    Keep model/planner objects alive across requests to avoid reloading.
    This is where you import graph_eqa and build your planner.
    """

    def __init__(self, vlm_model_name: str, use_image: bool, add_history: bool):
        self.vlm_model_name = vlm_model_name
        self.use_image = use_image
        self.add_history = add_history

        print("[SERVER] Infer server started", file=sys.stderr, flush=True)


        # --- Import graph_eqa lazily here so Conda env provides deps ---
        # Adjust imports to match your actual graph_eqa package API.
        try:
            # Example placeholders – update to your actual classes/functions
            # from graph_eqa.planners.vlm_planner import VLMPlanner
            # self.planner = VLMPlanner(model_name=vlm_model_name, use_image=use_image, add_history=add_history)
            self.planner = None
        except Exception as e:
            raise RuntimeError(f"Failed to import/build GraphEQA planner: {e}") from e

    def plan(self, question: str, choices, images_dir: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a JSON-serializable dict with at least:
          - answer (e.g. "B")
          - (optional) goal (frame,x,y,yaw)
          - debug
        """
        # ---- Minimal working fallback (so plumbing works immediately) ----
        # Replace this with your real call into graph_eqa:
        image_paths = _load_latest_image_paths(images_dir, k=5) if self.use_image else []

        # TODO: Replace this stub with your graph_eqa call:
        # result = self.planner.plan(scene_graph=..., question=question, choices=choices, images=image_paths, context=context)

        # Example dummy result:
        answer = "A" if choices else ""

        print(f"[SERVER] Using {len(image_paths)} images", file=sys.stderr, flush=True)
        for p in image_paths[-3:]:
            print(f"[SERVER]   {p}", file=sys.stderr, flush=True)

        return {
            "answer": answer,
            "goal": context.get("goal"),  # optionally passed in
            "debug": {
                "vlm_model_name": self.vlm_model_name,
                "use_image": self.use_image,
                "add_history": self.add_history,
                "num_images_used": len(image_paths),
                "images_used": image_paths[-2:],  # keep small
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

    # Handshake (UIAP-style) :contentReference[oaicite:1]{index=1}
    _print_json({"type": "ready", "ok": True})

    # Main loop
    for line in sys.stdin:
        s = (line or "").strip()
        if not s:
            continue

        try:
            req = json.loads(s)
        except json.JSONDecodeError:
            _print_json({"type": "error", "ok": False, "error": "bad_json"})
            continue

        # ✅ Log once, right after parse
        print(
            "[SERVER] Received request:\n" + json.dumps(req, indent=2),
            file=sys.stderr,
            flush=True
        )

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

            # (optional) log response on stderr
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
            print("[SERVER] Exception occurred (see traceback in JSON)", file=sys.stderr, flush=True)



    return 0


if __name__ == "__main__":
    raise SystemExit(main())