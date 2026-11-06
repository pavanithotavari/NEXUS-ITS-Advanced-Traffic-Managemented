"""Compare fixed-time vs actuated vs tabular Q-learning vs Deep Q-Network."""
import argparse
from pathlib import Path
import numpy as np
from .config import load_config
from .simulator import Intersection
from .controller import (
    FixedTimeController,
    ActuatedController,
    QLearningController,
    GreenWaveController,
)
from .deep_rl import DeepQLearningController


def run_episode(cfg, ctrl, seed, hour):
    env = Intersection(cfg, seed=seed, start_hour=hour)
    obs = env.reset()
    qsum, served = 0.0, 0
    for _ in range(cfg["sim"]["episode_steps"]):
        obs, _, _, info = env.step(ctrl.act(*obs))
        qsum += obs[0].sum()
        served += info["served"].sum()
    return qsum / cfg["sim"]["episode_steps"], served


def evaluate(cfg, q_path="models/q_table.npy", dqn_path="models/dqn_model.npz", n=20):
    ctrls = [FixedTimeController(cfg), ActuatedController(cfg)]

    if Path(q_path).exists():
        ctrls.append(QLearningController(cfg).load(q_path))
    else:
        ctrls.append(QLearningController(cfg))

    if Path(dqn_path).exists():
        ctrls.append(DeepQLearningController(cfg).load(dqn_path))
    else:
        ctrls.append(DeepQLearningController(cfg))

    ctrls.append(GreenWaveController(cfg))

    print("=" * 76)
    print(" NEXUS-ITS MULTI-CONTROLLER BENCHMARK EVALUATION")
    print("=" * 76)
    print(f"{'Controller':<18}{'Avg Queue (veh)':>16}{'Throughput (v/h)':>18}{'CO2 Cut vs Base':>20}")
    print("-" * 76)

    out = {}
    base_q = None

    for c in ctrls:
        res = [run_episode(cfg, c, seed=1000 + i, hour=6 + (i * 14 / n)) for i in range(n)]
        aq = float(np.mean([r[0] for r in res]))
        th = float(np.mean([r[1] for r in res]))
        out[c.name] = (aq, th)

        if base_q is None:
            base_q = aq
            reduction_str = "BASELINE"
        else:
            pct = ((base_q - aq) / max(base_q, 1e-3)) * 100.0
            reduction_str = f"{pct:+.1f}%"

        print(f"{c.name:<18}{aq:>16.2f}{th:>18.0f}{reduction_str:>20}")

    print("=" * 76)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--q", default="models/q_table.npy")
    p.add_argument("--dqn", default="models/dqn_model.npz")
    p.add_argument("-n", type=int, default=15)
    a = p.parse_args()
    evaluate(load_config(a.config), a.q, a.dqn, a.n)
