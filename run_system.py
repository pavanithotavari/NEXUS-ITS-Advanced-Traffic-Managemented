#!/usr/bin/env python3
"""Root convenience script to launch the NEXUS-ITS system from parent directory."""
import sys
from pathlib import Path

# Add src to path
proj_dir = Path(__file__).resolve().parent / "edge-adaptive-traffic" / "edge-adaptive-traffic"
src_dir = proj_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from edge_traffic.launch import main

if __name__ == "__main__":
    main()
