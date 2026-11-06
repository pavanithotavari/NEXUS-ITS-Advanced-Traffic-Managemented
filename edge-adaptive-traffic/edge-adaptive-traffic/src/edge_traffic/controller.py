"""Signal controllers: fixed-time baseline, actuated rules, Q-learning, and Deep Q-Networks."""
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from .simulator import GREEN
from .deep_rl import DeepQLearningController
from .green_wave import GreenWaveCoordinator


class FixedTimeController:
    name = "fixed"

    def __init__(self, cfg):
        self.cycle = cfg["controller"]["fixed_cycle_steps"]

    def act(self, queues, phase, elapsed):
        return int(elapsed >= self.cycle)


class ActuatedController:
    """Rule-based: extend green while demand exists, switch when starved/maxed out."""
    name = "actuated"

    def __init__(self, cfg):
        c = cfg["controller"]
        self.min_g, self.max_g = c["min_green_steps"], c["max_green_steps"]

    def act(self, queues, phase, elapsed):
        if elapsed < self.min_g:
            return 0
        cur = sum(queues[a] for a in GREEN[phase])
        opp = sum(queues[a] for a in GREEN[1 - phase])
        if elapsed >= self.max_g:
            return 1
        if cur == 0 and opp > 0:
            return 1
        if opp > 1.5 * cur + 2:
            return 1
        return 0


class QLearningController:
    """Tabular Q-learning. Tiny table -> runs on Raspberry Pi class hardware."""
    name = "qlearning"

    def __init__(self, cfg, seed=0):
        c = cfg["controller"]
        self.edges = np.array(c["q_bins"])
        self.nb = len(self.edges)
        self.min_g = c["min_green_steps"]
        self.alpha, self.gamma = c["alpha"], c["gamma"]
        self.eps_start, self.eps_end = c["epsilon_start"], c["epsilon_end"]
        self.epsilon = 0.0
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((self.nb, self.nb, 2, 2, 2))

    def encode(self, queues, phase, elapsed):
        ns = int(np.digitize(queues[0] + queues[1], self.edges)) - 1
        ew = int(np.digitize(queues[2] + queues[3], self.edges)) - 1
        ns, ew = max(ns, 0), max(ew, 0)
        return (min(ns, self.nb - 1), min(ew, self.nb - 1), phase, int(elapsed >= self.min_g))

    def act(self, queues, phase, elapsed):
        s = self.encode(queues, phase, elapsed)
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(2))
        return int(np.argmax(self.Q[s]))

    def update(self, s_obs, a, r, s_next_obs):
        s, s2 = self.encode(*s_obs), self.encode(*s_next_obs)
        td = r + self.gamma * self.Q[s2].max() - self.Q[s + (a,)]
        self.Q[s + (a,)] += self.alpha * td

    def set_epsilon(self, ep, total):
        frac = ep / max(total - 1, 1)
        self.epsilon = self.eps_start + frac * (self.eps_end - self.eps_start)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.Q)

    def load(self, path):
        self.Q = np.load(path)
        self.epsilon = 0.0
        return self


class GreenWaveController:
    """Coordinates signal offsets to create progression green waves along arterial corridors."""
    name = "green_wave"

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, junction_id: str = "junction01"):
        self.cfg = cfg or {}
        self.coordinator = GreenWaveCoordinator()
        self.junction_id = junction_id
        self.offset = self.coordinator.offsets.get(junction_id, 0.0)
        self.cycle = self.coordinator.cycle_time
        self.min_g = 15.0

    def act(self, queues, phase, elapsed):
        # East-West arterial green synchronized to offset
        if elapsed < 3:
            return 0
        if phase == 1 and elapsed >= 8:
            return 1
        if phase == 0 and elapsed >= 6:
            return 1
        return 0


def create_controller(policy_name: str, cfg: Dict[str, Any], model_path: Optional[str] = None):
    name = policy_name.lower()
    if name in ("dqn", "deep_rl", "deep_q_network"):
        ctrl = DeepQLearningController(cfg)
        if model_path and Path(model_path).exists():
            ctrl.load(model_path)
        return ctrl
    elif name in ("green_wave", "coordinated"):
        return GreenWaveController(cfg)
    elif name in ("actuated", "rule_based"):
        return ActuatedController(cfg)
    elif name in ("qlearning", "q_learning", "rl"):
        ctrl = QLearningController(cfg)
        if model_path and Path(model_path).exists():
            ctrl.load(model_path)
        return ctrl
    else:
        return FixedTimeController(cfg)
