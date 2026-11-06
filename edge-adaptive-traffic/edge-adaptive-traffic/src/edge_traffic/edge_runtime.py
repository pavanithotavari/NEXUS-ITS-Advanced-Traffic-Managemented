"""Edge runtime: sense -> decide -> actuate -> log (-> publish).

  python -m edge_traffic.edge_runtime --source sim
  python -m edge_traffic.edge_runtime --source video --video path/or/0
"""
import argparse
import csv
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen
from .config import load_config
from .simulator import Intersection
from .controller import QLearningController, ActuatedController

PHASE_NAME = {0: "NS_GREEN", 1: "EW_GREEN"}


def make_publisher(cfg):
    if not cfg["mqtt"]["enabled"]:
        return lambda payload: None
    import paho.mqtt.client as mqtt
    cl = mqtt.Client()
    cl.connect(cfg["mqtt"]["host"], cfg["mqtt"]["port"])
    cl.loop_start()
    return lambda payload: cl.publish(cfg["mqtt"]["topic"], json.dumps(payload))


def post_telemetry(api_url, payload):
    request = Request(
        f"{api_url.rstrip('/')}/api/telemetry",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=5) as response:
        if response.status != 201:
            raise RuntimeError(f"Telemetry API returned unexpected status {response.status}.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["sim", "video"], default="sim")
    p.add_argument("--video", default="0")
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--policy", choices=["qlearning", "actuated"], default="qlearning")
    p.add_argument("--q", default="models/q_table.npy")
    p.add_argument("--realtime", action="store_true", help="sleep one step_seconds per step")
    p.add_argument("--api-url", help="POST telemetry to a running API, e.g. http://127.0.0.1:8000")
    p.add_argument("--intersection-id", default="junction01")
    p.add_argument("--config")
    a = p.parse_args()

    cfg = load_config(a.config)
    if a.policy == "qlearning" and Path(a.q).exists():
        ctrl = QLearningController(cfg).load(a.q)
    else:
        ctrl = ActuatedController(cfg)
    print(f"Policy: {ctrl.name}")
    publish = make_publisher(cfg)
    Path("data").mkdir(exist_ok=True)
    log_file = open("data/runtime_log.csv", "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["t", "phase", "elapsed", "qN", "qS", "qE", "qW", "action"])

    if a.source == "sim":
        env = Intersection(cfg, seed=7, start_hour=8.0)
        obs = env.reset()
    else:
        import cv2
        from .detector import YoloDetector
        det = YoloDetector(cfg)
        cap = cv2.VideoCapture(int(a.video) if a.video.isdigit() else a.video)
        state = {"phase": 0, "elapsed": 0}

    try:
        for t in range(a.steps):
            if a.source == "sim":
                queues, phase, elapsed = obs
                action = ctrl.act(queues, phase, elapsed)
                obs, _, _, _ = env.step(action)
            else:
                ok, frame = cap.read()
                if not ok:
                    break
                queues = det.count(frame)
                phase, elapsed = state["phase"], state["elapsed"]
                action = ctrl.act(queues, phase, elapsed)
                if action == 1 and elapsed >= cfg["controller"]["min_green_steps"]:
                    state["phase"], state["elapsed"] = 1 - phase, 0
                else:
                    state["elapsed"] += 1
                # TODO: drive real signal hardware here (GPIO / relay / NTCIP controller)
            log.writerow([t, phase, elapsed, *map(int, queues), action])
            payload = {
                "intersection_id": a.intersection_id,
                "t": t,
                "phase": PHASE_NAME[phase],
                "queues": dict(zip(("N", "S", "E", "W"), map(int, queues))),
                "action": action,
            }
            publish(payload)
            if a.api_url:
                post_telemetry(a.api_url, payload)
            if t % 20 == 0:
                print(f"t={t:4d} {PHASE_NAME[phase]:9s} q(N,S,E,W)={list(map(int, queues))} action={'SWITCH' if action else 'hold'}")
            if a.realtime:
                time.sleep(cfg["sim"]["step_seconds"])
    finally:
        log_file.close()
        if a.source == "video":
            cap.release()
    print("Log written to data/runtime_log.csv")


if __name__ == "__main__":
    main()
