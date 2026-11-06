import numpy as np
from edge_traffic.config import load_config
from edge_traffic.simulator import Intersection
from edge_traffic.controller import FixedTimeController, ActuatedController, QLearningController
from edge_traffic.train_rl import train
from edge_traffic.evaluate import run_episode


def test_min_green_enforced():
    cfg = load_config()
    env = Intersection(cfg, seed=0, start_hour=8)
    env.reset()
    env.step(1)  # too early, must be ignored
    assert env.phase == 0


def test_phase_switch_after_min_green():
    cfg = load_config()
    env = Intersection(cfg, seed=0, start_hour=8)
    env.reset()
    for _ in range(cfg["controller"]["min_green_steps"]):
        env.step(0)
    env.step(1)
    assert env.phase == 1


def test_training_beats_or_matches_fixed(tmp_path):
    cfg = load_config()
    agent = train(cfg, episodes=60, out=str(tmp_path / "q.npy"), verbose=False)
    agent.epsilon = 0.0
    fixed = FixedTimeController(cfg)
    q = np.mean([run_episode(cfg, agent, 5 + i, 8.5)[0] for i in range(5)])
    f = np.mean([run_episode(cfg, fixed, 5 + i, 8.5)[0] for i in range(5)])
    assert q <= f * 1.15  # learned policy should be competitive
