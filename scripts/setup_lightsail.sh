#!/bin/bash
set -euo pipefail

# Setup script for Polymarket data collector on AWS Lightsail
# Usage: bash scripts/setup_lightsail.sh

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$REPO_DIR/.venv"
SERVICE_NAME="polymarket-collector"

echo "=== Polymarket Data Collector — Lightsail Setup ==="
echo "  Repo: $REPO_DIR"

# 1. System dependencies
echo "[1/5] Installing system packages..."
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
echo "[2/5] Creating Python virtual environment..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# 3. Install dependencies
echo "[3/5] Installing Python dependencies..."
pip install --upgrade pip
pip install -r "$REPO_DIR/requirements.txt"

# 4. Create data directory
echo "[4/5] Creating data and logs directories..."
mkdir -p "$REPO_DIR/data" "$REPO_DIR/logs"

# 5. Install systemd service
echo "[5/5] Installing systemd service..."
sudo tee /etc/systemd/system/${SERVICE_NAME}.service > /dev/null << EOF
[Unit]
Description=Polymarket Data Collector
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$REPO_DIR
ExecStart=$VENV_DIR/bin/python scripts/data_collector.py
Restart=always
RestartSec=10
StartLimitIntervalSec=300
StartLimitBurst=5
Environment=PYTHONUNBUFFERED=1
StandardOutput=journal
StandardError=journal
SyslogIdentifier=$SERVICE_NAME

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Start the collector:"
echo "  sudo systemctl start ${SERVICE_NAME}"
echo ""
echo "Useful commands:"
echo "  sudo systemctl status ${SERVICE_NAME}     # check status"
echo "  sudo journalctl -u ${SERVICE_NAME} -f     # live logs"
echo "  sudo systemctl restart ${SERVICE_NAME}     # restart"
echo "  sudo systemctl stop ${SERVICE_NAME}        # stop"
echo ""
