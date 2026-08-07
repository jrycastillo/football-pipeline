# Deployment — systemd service files

> **Full end-to-end setup (clone → models → config → DB → run) is in
> [`docs/DEPLOYMENT.md`](../docs/DEPLOYMENT.md).** This directory just holds the
> systemd unit + installer referenced there.

This directory contains files for deploying the Football Pipeline as a systemd service on Linux servers (Ubuntu/Debian).

## Quick Start

### 1. Prerequisites

Ensure you have:
- Ubuntu 20.04+ or Debian 11+
- Python 3.8+
- sudo/root access
- Project cloned to `/home/ubuntu/football` (or adjust paths in service file)

### 2. Setup Environment

```bash
cd /home/ubuntu/football

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
nano .env  # Edit with your credentials
```

### 3. Install Service

```bash
# Make installer executable
chmod +x deployment/install-service.sh

# Run installer (requires sudo)
sudo deployment/install-service.sh
```

The service will:
- ✅ Auto-start on system boot
- ✅ Restart automatically if it crashes
- ✅ Poll for new videos every 60 seconds
- ✅ Process 3 videos in parallel
- ✅ Log to `/home/ubuntu/football/logs/`

## Service Management

### Check Status
```bash
sudo systemctl status football-pipeline
```

### View Logs (Live)
```bash
# Application logs
tail -f /home/ubuntu/football/logs/pipeline.log

# Error logs
tail -f /home/ubuntu/football/logs/pipeline-error.log

# Systemd journal
journalctl -u football-pipeline -f
```

### Stop/Start/Restart
```bash
sudo systemctl stop football-pipeline
sudo systemctl start football-pipeline
sudo systemctl restart football-pipeline
```

### Disable Auto-Start
```bash
sudo systemctl disable football-pipeline
```

### Re-enable Auto-Start
```bash
sudo systemctl enable football-pipeline
```

## Configuration

### Adjust Service Parameters

Edit the service file before installation:

```bash
nano deployment/football-pipeline.service
```

Key parameters:
- `--parallel 3`: Number of concurrent video processing jobs
- `--poll_interval 60`: Seconds between API polls
- `--tracking_mode bytetrack`: Tracker to use (bytetrack/botsort/sam2)
- `--locking_mode 2`: Internal processing mode

### Change Working Directory

If your project is NOT in `/home/ubuntu/football`, update these lines in `football-pipeline.service`:

```ini
WorkingDirectory=/your/project/path
EnvironmentFile=/your/project/path/.env
ExecStart=/your/project/path/venv/bin/python orchestrator.py ...
StandardOutput=append:/your/project/path/logs/pipeline.log
```

Also update `PROJECT_DIR` in `install-service.sh`.

### Change User

If running as a different user (not `ubuntu`), update:

```ini
User=youruser
Group=youruser
```

## Health Monitoring

The service automatically saves health snapshots every 10 processing cycles:

```bash
# View health snapshot
cat /home/ubuntu/football/output/health_snapshot.json
```

Health metrics include:
- Videos processed/failed/running
- Processing performance (FPS, avg time)
- Database query stats
- API call stats
- Recent errors

## Troubleshooting

### Service Won't Start

1. Check logs:
   ```bash
   journalctl -u football-pipeline -n 50
   ```

2. Verify .env file exists and has correct permissions:
   ```bash
   ls -la /home/ubuntu/football/.env
   ```

3. Test manually:
   ```bash
   cd /home/ubuntu/football
   source venv/bin/activate
   python orchestrator.py --poll --parallel 1
   ```

### Database Connection Errors

- Verify `MYSQL_PASSWORD` is set in `.env`
- Test database connection:
  ```bash
  mysql -h $MYSQL_HOST -P $MYSQL_PORT -u $MYSQL_USER -p$MYSQL_PASSWORD $MYSQL_DB
  ```

### Out of Memory

Reduce parallel workers:
```bash
# Edit service file
sudo nano /etc/systemd/system/football-pipeline.service

# Change --parallel 3 to --parallel 1
# Then reload:
sudo systemctl daemon-reload
sudo systemctl restart football-pipeline
```

### Permission Denied Errors

Ensure correct ownership:
```bash
sudo chown -R ubuntu:ubuntu /home/ubuntu/football
sudo chmod +x /home/ubuntu/football/venv/bin/python
```

## Updating the Service

After changing code or configuration:

```bash
# Pull latest code
cd /home/ubuntu/football
git pull

# Update dependencies if needed
source venv/bin/activate
pip install -r requirements.txt

# Restart service
sudo systemctl restart football-pipeline
```

## Uninstall

To completely remove the service:

```bash
# Stop and disable
sudo systemctl stop football-pipeline
sudo systemctl disable football-pipeline

# Remove service file
sudo rm /etc/systemd/system/football-pipeline.service

# Reload systemd
sudo systemctl daemon-reload
```

## Advanced: Process Monitoring

### Set Up Monitoring Alerts

You can use systemd's `OnFailure` directive to send alerts:

Add to `football-pipeline.service`:
```ini
[Unit]
OnFailure=alert-email@%n.service
```

### Resource Usage

Monitor CPU/Memory:
```bash
# Real-time
systemctl status football-pipeline

# Historical
journalctl -u football-pipeline | grep -i "memory\|cpu"
```

## Production Checklist

Before deploying to production:

- [ ] `.env` file configured with production credentials
- [ ] Database accessible from server
- [ ] Sufficient disk space for output (100GB+ recommended)
- [ ] GPU drivers installed (if using GPU)
- [ ] Model files downloaded to `models/` directory
- [ ] Test run completed successfully
- [ ] Monitoring/alerting configured
- [ ] Log rotation set up (see below)

### Log Rotation

Create `/etc/logrotate.d/football-pipeline`:

```
/home/ubuntu/football/logs/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0644 ubuntu ubuntu
    sharedscripts
    postrotate
        systemctl reload football-pipeline > /dev/null 2>&1 || true
    endscript
}
```

---

**Need Help?** Check the main [README.md](../README.md) or [PIPELINE_GUIDE.md](../docs/PIPELINE_GUIDE.md)
