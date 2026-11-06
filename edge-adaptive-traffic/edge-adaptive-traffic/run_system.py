#!/usr/bin/env python3
"""Root convenience script to launch the NEXUS-ITS system."""
import sys
from pathlib import Path

# Add src to path
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from edge_traffic.launch import main

if __name__ == "__main__":
    main()
