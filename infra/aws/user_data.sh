#!/bin/bash
set -euxo pipefail
exec > /var/log/polymarket-setup.log 2>&1

echo "=== Polymarket data collector bootstrap ==="

# 1. System packages
dnf update -y
dnf install -y python3.12 python3.12-pip git

# 2. Create service user and directories
useradd -r -m -d /opt/polymarket -s /bin/bash polymarket || true
mkdir -p /opt/polymarket/{data,logs,app}

# 3. Clone repo
if [ ! -d /opt/polymarket/app/.git ]; then
    git clone --branch master https://github.com/krisseo0202/polymarket_trading.git /opt/polymarket/app
fi

# 4. Python venv + dependencies
python3.12 -m venv /opt/polymarket/venv
/opt/polymarket/venv/bin/pip install --upgrade pip
/opt/polymarket/venv/bin/pip install -r /opt/polymarket/app/requirements.txt

# 5. Symlink data and logs so scripts write to persistent paths
if [ ! -L /opt/polymarket/app/data ]; then
    rm -rf /opt/polymarket/app/data
    ln -sf /opt/polymarket/data /opt/polymarket/app/data
fi
if [ ! -L /opt/polymarket/app/logs ]; then
    rm -rf /opt/polymarket/app/logs
    ln -sf /opt/polymarket/logs /opt/polymarket/app/logs
fi

# 6. Create .env.example as a template
cat > /opt/polymarket/app/.env.example << 'ENVEOF'
HOST=https://clob.polymarket.com
CHAIN_ID=137
PRIVATE_KEY=<your-private-key>
PROXY_FUNDER=<your-proxy-funder-address>
API_KEY=<your-api-key>
API_SECRET=<your-api-secret>
API_PASSPHRASE=<your-api-passphrase>
ENVEOF

# 7. Install systemd units
cp /opt/polymarket/app/infra/aws/systemd/*.service /etc/systemd/system/
cp /opt/polymarket/app/infra/aws/systemd/*.timer /etc/systemd/system/
systemctl daemon-reload

# 8. Enable services (they won't start until .env exists due to ConditionPathExists)
systemctl enable polymarket-btc-recorder.service
systemctl enable polymarket-snapshots.service
systemctl enable polymarket-history.timer

# 9. Set up daily S3 backup cron (runs but is a no-op if aws cli isn't configured)
cat > /etc/cron.daily/polymarket-s3-backup << 'CRONEOF'
#!/bin/bash
# Skip if aws cli is not configured
if ! sudo -u polymarket aws sts get-caller-identity &>/dev/null; then
    exit 0
fi
BUCKET=$(sudo -u polymarket aws s3 ls 2>/dev/null | grep polymarket-collector-data | awk '{print $3}')
if [ -n "$BUCKET" ]; then
    sudo -u polymarket aws s3 sync /opt/polymarket/data/ "s3://$BUCKET/data/" --exclude "*.tmp"
fi
CRONEOF
chmod +x /etc/cron.daily/polymarket-s3-backup

# 10. Fix ownership
chown -R polymarket:polymarket /opt/polymarket

echo "=== Bootstrap complete. SSH in, create .env, then start services. ==="
