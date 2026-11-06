"""Deep Q-Network (DQN) Reinforcement Learning Engine for Edge Traffic Signals.

Provides deep neural network control with experience replay, target networks,
and adaptive exploration. Implemented with high-performance vectorized NumPy
for zero-dependency edge hardware deployment (Raspberry Pi, Jetson, PC).
"""
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self.idx = 0

    def push(self, s: np.ndarray, a: int, r: float, s_next: np.ndarray, done: bool):
        if len(self.buffer) < self.capacity:
            self.buffer.append((s, a, r, s_next, done))
        else:
            self.buffer[self.idx] = (s, a, r, s_next, done)
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in indices:
            s, a, r, s_n, d = self.buffer[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(s_n)
            dones.append(d)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=bool),
        )

    def __len__(self):
        return len(self.buffer)


class NeuralNetwork:
    """Vectorized 3-layer neural network with Adam optimizer."""

    def __init__(self, in_dim: int = 20, hidden_dim: int = 64, out_dim: int = 2, lr: float = 0.001):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.lr = lr

        # He / Kaiming normal initialization
        self.W1 = np.random.randn(in_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / in_dim)
        self.b1 = np.zeros((1, hidden_dim), dtype=np.float32)
        self.W2 = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, hidden_dim), dtype=np.float32)
        self.W3 = np.random.randn(hidden_dim, out_dim).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b3 = np.zeros((1, out_dim), dtype=np.float32)

        # Adam optimizer state
        self.mW1, self.vW1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        self.mb1, self.vb1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        self.mW2, self.vW2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        self.mb2, self.vb2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        self.mW3, self.vW3 = np.zeros_like(self.W3), np.zeros_like(self.W3)
        self.mb3, self.vb3 = np.zeros_like(self.b3), np.zeros_like(self.b3)
        self.t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8

    def forward(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.maximum(0.01 * self.z1, self.z1)  # LeakyReLU
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = np.maximum(0.01 * self.z2, self.z2)
        self.out = np.dot(self.a2, self.W3) + self.b3
        return self.out

    def train_step(self, X: np.ndarray, y_target: np.ndarray) -> float:
        """One step of backpropagation MSE loss."""
        batch_size = X.shape[0]
        preds = self.forward(X)
        error = preds - y_target  # shape (B, out_dim)
        loss = float(np.mean(error ** 2))

        # Gradient of MSE wrt output
        d_out = (2.0 / batch_size) * error

        # Layer 3 grads
        dW3 = np.dot(self.a2.T, d_out)
        db3 = np.sum(d_out, axis=0, keepdims=True)

        # Layer 2 grads
        d_a2 = np.dot(d_out, self.W3.T)
        d_z2 = d_a2 * np.where(self.z2 > 0, 1.0, 0.01)
        dW2 = np.dot(self.a1.T, d_z2)
        db2 = np.sum(d_z2, axis=0, keepdims=True)

        # Layer 1 grads
        d_a1 = np.dot(d_z2, self.W2.T)
        d_z1 = d_a1 * np.where(self.z1 > 0, 1.0, 0.01)
        dW1 = np.dot(X.T, d_z1)
        db1 = np.sum(d_z1, axis=0, keepdims=True)

        # Adam updates
        self.t += 1
        for param, grad, m, v in [
            (self.W1, dW1, self.mW1, self.vW1),
            (self.b1, db1, self.mb1, self.vb1),
            (self.W2, dW2, self.mW2, self.vW2),
            (self.b2, db2, self.mb2, self.vb2),
            (self.W3, dW3, self.mW3, self.vW3),
            (self.b3, db3, self.mb3, self.vb3),
        ]:
            m[:] = self.beta1 * m + (1.0 - self.beta1) * grad
            v[:] = self.beta2 * v + (1.0 - self.beta2) * (grad ** 2)
            m_hat = m / (1.0 - self.beta1 ** self.t)
            v_hat = v / (1.0 - self.beta2 ** self.t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        return loss

    def copy_weights_to(self, target: "NeuralNetwork"):
        target.W1 = self.W1.copy()
        target.b1 = self.b1.copy()
        target.W2 = self.W2.copy()
        target.b2 = self.b2.copy()
        target.W3 = self.W3.copy()
        target.b3 = self.b3.copy()


class DeepQLearningController:
    """Deep Q-Network Agent for smart traffic signal control."""
    name = "deep_q_network"

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, in_dim: int = 20, hidden_dim: int = 64):
        self.cfg = cfg or {}
        ctrl_cfg = self.cfg.get("controller", {})
        self.gamma = ctrl_cfg.get("gamma", 0.95)
        self.epsilon = ctrl_cfg.get("epsilon_start", 0.20)
        self.min_green = ctrl_cfg.get("min_green_steps", 3)
        self.batch_size = 32
        self.target_update_freq = 50
        self.train_step_count = 0

        self.policy_net = NeuralNetwork(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=2, lr=0.001)
        self.target_net = NeuralNetwork(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=2)
        self.policy_net.copy_weights_to(self.target_net)

        self.buffer = ReplayBuffer(capacity=8000)

    def encode_state(
        self,
        queues: List[int],
        phase: int,
        elapsed: int,
        moving_counts: Optional[List[int]] = None,
        emergency_approach: Optional[str] = None,
        weather: str = "clear"
    ) -> np.ndarray:
        """Construct normalized feature vector (20-dimensional)."""
        # Queues normalized (max 50)
        q_norm = [min(q, 50) / 50.0 for q in queues]
        # Moving vehicles (max 30)
        mv = moving_counts or [0, 0, 0, 0]
        mv_norm = [min(m, 30) / 30.0 for m in mv]
        # Active phase one-hot
        p_oh = [1.0 if phase == 0 else 0.0, 1.0 if phase == 1 else 0.0]
        # Elapsed time normalized
        el_norm = [min(elapsed, 20) / 20.0]
        # Emergency approach one-hot: None, N, S, E, W
        emg_map = {None: 0, "N": 1, "S": 2, "E": 3, "W": 4}
        emg_idx = emg_map.get(emergency_approach, 0)
        emg_oh = [1.0 if i == emg_idx else 0.0 for i in range(5)]
        # Weather one-hot: clear, rain, fog, night
        w_map = {"clear": 0, "rain": 1, "fog": 2, "night": 3}
        w_idx = w_map.get(weather.lower(), 0)
        w_oh = [1.0 if i == w_idx else 0.0 for i in range(4)]

        features = q_norm + mv_norm + p_oh + el_norm + emg_oh + w_oh
        return np.array(features, dtype=np.float32)

    def act(self, queues, phase, elapsed, state_vec: Optional[np.ndarray] = None) -> int:
        """Choose action: 0 = hold phase, 1 = switch."""
        if elapsed < self.min_green:
            return 0  # Enforce safety minimum green

        if np.random.random() < self.epsilon:
            return int(np.random.randint(2))

        if state_vec is None:
            state_vec = self.encode_state(queues, phase, elapsed)

        q_values = self.policy_net.forward(state_vec)[0]
        return int(np.argmax(q_values))

    def update(self, s: np.ndarray, a: int, r: float, s_next: np.ndarray, done: bool = False):
        self.buffer.push(s, a, r, s_next, done)

        if len(self.buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Target Q calculation with target network
        next_q = self.target_net.forward(next_states)
        max_next_q = np.max(next_q, axis=1)
        targets = self.policy_net.forward(states)

        for i in range(self.batch_size):
            target_val = rewards[i] if dones[i] else rewards[i] + self.gamma * max_next_q[i]
            targets[i, actions[i]] = target_val

        loss = self.policy_net.train_step(states, targets)
        self.train_step_count += 1

        if self.train_step_count % self.target_update_freq == 0:
            self.policy_net.copy_weights_to(self.target_net)

        return loss

    def save(self, filepath: str = "models/dqn_model.npz"):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            filepath,
            W1=self.policy_net.W1, b1=self.policy_net.b1,
            W2=self.policy_net.W2, b2=self.policy_net.b2,
            W3=self.policy_net.W3, b3=self.policy_net.b3,
        )

    def load(self, filepath: str = "models/dqn_model.npz"):
        if Path(filepath).exists():
            data = np.load(filepath)
            self.policy_net.W1 = data["W1"]
            self.policy_net.b1 = data["b1"]
            self.policy_net.W2 = data["W2"]
            self.policy_net.b2 = data["b2"]
            self.policy_net.W3 = data["W3"]
            self.policy_net.b3 = data["b3"]
            self.policy_net.copy_weights_to(self.target_net)
            self.epsilon = 0.02
        return self
