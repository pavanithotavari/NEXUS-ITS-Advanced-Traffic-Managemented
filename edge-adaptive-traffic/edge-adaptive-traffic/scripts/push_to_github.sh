#!/usr/bin/env bash
# Usage: ./scripts/push_to_github.sh https://github.com/<your-username>/edge-adaptive-traffic.git
set -e
git init
git add .
git commit -m "Initial commit: Edge AI adaptive traffic management"
git branch -M main
git remote add origin "$1"
git push -u origin main
