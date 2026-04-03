#!/usr/bin/env bash
# run.sh — connect NordVPN to South Korea then launch bot.py
# Usage: ./run.sh [bot.py args...]
set -euo pipefail

COUNTRY="South_Korea"

connect_vpn() {
  if command -v nordvpn &>/dev/null; then
    echo "[VPN] Connecting to NordVPN — $COUNTRY..."
    nordvpn connect "$COUNTRY"
    sleep 2
    nordvpn status
  else
    echo "[VPN] nordvpn CLI not found in WSL2. Checking if Windows NordVPN is routing traffic..."
    IP_INFO=$(curl -s --max-time 5 https://ipinfo.io/json || echo "{}")
    COUNTRY_CODE=$(echo "$IP_INFO" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('country','??'))" 2>/dev/null || echo "??")
    echo "[VPN] Detected country code: $COUNTRY_CODE"
    if [[ "$COUNTRY_CODE" == "US" ]]; then
      echo "[VPN] ERROR: IP appears to be US-based. Connect NordVPN to South Korea from Windows first."
      exit 1
    fi
    echo "[VPN] IP looks non-US ($COUNTRY_CODE). Proceeding."
  fi
}

connect_vpn
echo "[BOT] Starting bot..."
python3 bot.py "$@"
