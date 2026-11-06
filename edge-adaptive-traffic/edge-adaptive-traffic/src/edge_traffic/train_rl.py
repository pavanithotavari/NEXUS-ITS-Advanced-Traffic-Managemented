"""Train the Q-learning signal controller in simulation."""
import argparse
import numpy as np
from .config import load_config
from .simulator import Intersection
from .controller import QLearningController


def train(cfg, episodes=None, out="models/q_table.npy", verbose=True):
    episodes = episodes or cfg["training"]["episodes"]
    seed = cfg["training"]["seed"]
    env = Intersection(cfg, seed=seed)
    agent = QLearningController(cfg, seed=seed)
    for ep in range(episodes):
        agent.set_epsilon(ep, episodes)
        obs = env.reset()
        total = 0.0
        for _ in range(cfg["sim"]["episode_steps"]):
            a = agent.act(*obs)
            nxt, r, _, _ = env.step(a)
            agent.update(obs, a, r, nxt)
            obs, total = nxt, total + r
        if verbose and (ep % 25 == 0 or ep == episodes - 1):
            print(f"episode {ep:4d}  eps={agent.epsilon:.2f}  avg_queue={-total / cfg['sim']['episode_steps']:.2f}")
    agent.save(out)
    return agent


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int)
    p.add_argument("--config")
    a = p.parse_args()
    train(load_config(a.config), a.episodes)
