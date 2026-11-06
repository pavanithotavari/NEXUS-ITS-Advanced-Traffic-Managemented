"""Lightweight 4-approach intersection simulator (N, S, E, W).

Phase 0 = North/South green, Phase 1 = East/West green.
Runs on any machine, so you can develop and train without a camera or SUMO.
"""
import numpy as np

N, S, E, W = 0, 1, 2, 3
GREEN = {0: (N, S), 1: (E, W)}


def arrival_rates(hour: float) -> np.ndarray:
    """Vehicles/second per approach. Morning peak favours N/S, evening peak E/W."""
    g = lambda mu, sd: np.exp(-0.5 * ((hour - mu) / sd) ** 2)
    base = 0.06
    ns = base + 0.20 * g(8.5, 1.2) + 0.08 * g(17.5, 1.2)
    ew = base + 0.08 * g(8.5, 1.2) + 0.20 * g(17.5, 1.2)
    return np.array([ns, ns, ew, ew])


class Intersection:
    def __init__(self, cfg, seed=0, start_hour=None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.dt = cfg["sim"]["step_seconds"]
        self.sat = cfg["sim"]["saturation_per_step"]
        self.lost_steps = cfg["sim"]["lost_steps"]
        self.min_green = cfg["controller"]["min_green_steps"]
        self.start_hour = start_hour
        self.reset()

    def reset(self):
        self.queues = np.zeros(4, dtype=int)
        self.phase = 0
        self.elapsed = 0
        self.lost = 0
        self.t = 0
        self.hour0 = (self.start_hour if self.start_hour is not None
                      else float(self.rng.uniform(6, 20)))
        return self.observe()

    @property
    def hour(self):
        return self.hour0 + self.t * self.dt / 3600.0

    def observe(self):
        return self.queues.copy(), self.phase, self.elapsed

    def step(self, action: int):
        """action: 0 = keep phase, 1 = request switch (ignored before min green)."""
        if action == 1 and self.elapsed >= self.min_green and self.lost == 0:
            self.phase = 1 - self.phase
            self.elapsed = 0
            self.lost = self.lost_steps
        else:
            self.elapsed += 1

        served = np.zeros(4, dtype=int)
        if self.lost > 0:
            self.lost -= 1
        else:
            for a in GREEN[self.phase]:
                cap = int(self.rng.binomial(self.sat, 0.5))
                served[a] = min(self.queues[a], cap)
        self.queues -= served

        arrivals = self.rng.poisson(arrival_rates(self.hour) * self.dt)
        self.queues += arrivals
        self.t += 1
        reward = -float(self.queues.sum())
        info = {"arrivals": arrivals, "served": served}
        return self.observe(), reward, False, info
