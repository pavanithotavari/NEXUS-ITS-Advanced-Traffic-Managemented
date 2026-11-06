"""Universal One-Click Launcher for NEXUS-ITS Autonomous Traffic Management & Digital Twin."""
import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path
import uvicorn

BANNER = r"""
  _  _ _____  ___   _ ___      ___ _____ ___ 
 | \| | __\ \/ / | | / __|___ |_ _|_   _/ __|
 | .` | _| >  <| |_| \__ \___| | |  | | \__ \
 |_|\_|___/_/\_\\___/|___/     |___| |_| |___/
 =========================================================================
 AUTONOMOUS EDGE-AI INTELLIGENT TRANSPORTATION SYSTEM & DIGITAL TWIN
 Version: 2.5.0 | Multi-Agent RL | Emergency Preemption | Arterial Green Wave
 =========================================================================
"""

def open_browser(url: str, delay: float = 1.2):
    time.sleep(delay)
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main():
    p = argparse.ArgumentParser(description="NEXUS-ITS Autonomous Traffic System Launcher")
    p.add_argument("--host", default="127.0.0.1", help="API host (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8000, help="API port (default: 8000)")
    p.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    p.add_argument("--reload", action="store_true", help="Enable uvicorn hot reloading")
    a = p.parse_args()

    print(BANNER)
    server_url = f"http://{a.host}:{a.port}"
    print(f"[*] Starting Edge AI Mission Control Center...")
    print(f"[*] Dashboard UI:             {server_url}/")
    print(f"[*] Real-Time WebSocket Hub:  ws://{a.host}:{a.port}/ws/live")
    print(f"[*] OpenAPI Interactive Docs: {server_url}/docs")
    print(f"[*] Health & Telemetry API:   {server_url}/health")
    print("-" * 73)
    print("[*] Storage Status: Hybrid Edge Resilient (SQLite + MongoDB Atlas Sync)")
    print("[*] Neural Core:    Vectorized Deep Q-Network + Tabular Q-Learning + Green Wave")
    print("[*] Simulation:     Microscopic IDM Multi-Junction Corridor Active")
    print("=" * 73)

    if not a.no_browser:
        threading.Thread(target=open_browser, args=(server_url,), daemon=True).start()

    uvicorn.run(
        "edge_traffic.api:app",
        host=a.host,
        port=a.port,
        reload=a.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
