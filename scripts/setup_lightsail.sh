#!/bin/bash
set -euo pipefail

# Setup script for Polymarket data collector on AWS Lightsail
# Usage: bash scripts/setup_lightsail.sh

echo "=== Polymarket Data Collector — Lightsail Setup ==="

# 1. System dependencies
echo "[1/4] Installing system packages..."
if command -v dnf &>/dev/null; then
    sudo dnf install -y python3 python3-pip git
elif command -v apt-get &>/dev/null; then
    sudo apt-get update
    sudo apt-get install -y python3 python3-pip python3-venv git
else
    echo "ERROR: unsupported package manager"
    exit 1
fi

# 2. Python venv
echo "[2/4] Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
echo "[3/4] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Create data directory
echo "[4/4] Creating data directory..."
mkdir -p data logs

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To run the data collector:"
echo "  source .venv/bin/activate"
echo "  python scripts/data_collector.py"
echo ""
echo "To run in background (survives SSH disconnect):"
echo "  source .venv/bin/activate"
echo "  nohup python scripts/data_collector.py > logs/collector.log 2>&1 &"
echo "  tail -f logs/collector.log"
echo ""
