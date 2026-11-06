"""Train Deep Q-Network (DQN) on adaptive intersection control."""
import argparse
import time
from pathlib import Path
import numpy as np
from .config import load_config
from .simulator import Intersection
from .deep_rl import DeepQLearningController


def train_dqn(cfg, episodes: int = 150, out: str = "models/dqn_model.npz", verbose: bool = True):
    env = Intersection(cfg, seed=cfg.get("training", {}).get("seed", 42))
    agent = DeepQLearningController(cfg)
    total_steps = cfg["sim"]["episode_steps"]
    rewards_history = []

    print(f"[*] Training Deep Q-Network for {episodes} episodes...")
    start_t = time.time()

    for ep in range(episodes):
        obs = env.reset()
        queues, phase, elapsed = obs
        state = agent.encode_state(list(queues), phase, elapsed)
        ep_reward = 0.0

        # Linear exploration decay
        agent.epsilon = max(0.02, 0.40 * (1.0 - ep / episodes))

        for t in range(total_steps):
            action = agent.act(list(queues), phase, elapsed, state_vec=state)
            next_obs, r, done, _ = env.step(action)
            n_queues, n_phase, n_elapsed = next_obs
            next_state = agent.encode_state(list(n_queues), n_phase, n_elapsed)

            agent.update(state, action, r, next_state, done=done)

            state = next_state
            queues, phase, elapsed = next_obs
            ep_reward += r

        rewards_history.append(ep_reward)
        if verbose and (ep + 1) % 15 == 0:
            avg_r = np.mean(rewards_history[-15:])
            print(f" Episode {ep + 1:3d}/{episodes:3d} | Avg Loss/Delay: {avg_r:8.1f} | Epsilon: {agent.epsilon:.3f}")

    agent.save(out)
    elapsed_time = time.time() - start_t
    print(f"[OK] DQN Model successfully trained and saved to {out} in {elapsed_time:.1f}s")
    return agent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--out", default="models/dqn_model.npz")
    p.add_argument("--config")
    a = p.parse_args()
    cfg = load_config(a.config)
    train_dqn(cfg, episodes=a.episodes, out=a.out)


if __name__ == "__main__":
    main()
