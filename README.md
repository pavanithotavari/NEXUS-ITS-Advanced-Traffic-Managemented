# NEXUS-ITS: Autonomous Edge-AI Smart City Traffic Orchestration & Digital Twin

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.5.0-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Edge AI](https://img.shields.io/badge/Edge%20AI-DQN%20%7C%20YOLOv8%20%7C%20EVP-cyan.svg)](#)

A next-generation, edge-native Intelligent Transportation System (ITS) featuring a **60 FPS Real-Time Microscopic Digital Twin**, **Deep Q-Network (DQN) Multi-Agent Reinforcement Learning**, **NTCIP 1202 Emergency Vehicle Preemption**, **Arterial Green Wave Corridor Coordination**, and **Edge Vision SORT Tracking with Incident Analytics**.

---

## Quickstart Guide

### 1. Launch Mission Control Center (One Command)

From this directory:

```powershell
python run_system.py
```

This starts the FastAPI server, launches the 20 Hz WebSocket digital twin, and opens the **Mission Control Dashboard** in your browser at `http://localhost:8000`.

### 2. Run Comprehensive Test Suite

```powershell
pytest -q
```
*All 17 integration and unit tests pass with 100% success.*

### 3. Run Controller Benchmark Evaluation

```powershell
python -m edge_traffic.evaluate -n 10
```

### 4. Train Deep Q-Network (DQN)

```powershell
python -m edge_traffic.train_dqn --episodes 50
```
