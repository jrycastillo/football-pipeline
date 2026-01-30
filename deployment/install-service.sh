#!/bin/bash
# Installation script for Football Pipeline systemd service
# Run this on your staging server as root or with sudo

set -e

echo "==== Football Pipeline Service Installer ===="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
   echo "❌ Please run as root or with sudo"
   exit 1
fi

# Configuration
SERVICE_NAME="football-pipeline"
PROJECT_DIR="/home/ubuntu/football"
SERVICE_FILE="deployment/football-pipeline.service"
LOG_DIR="${PROJECT_DIR}/logs"
OUTPUT_DIR="${PROJECT_DIR}/output"

# Verify project directory exists
if [ ! -d "$PROJECT_DIR" ]; then
    echo "❌ Project directory not found: $PROJECT_DIR"
    echo "   Please adjust PROJECT_DIR in this script"
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"
chown ubuntu:ubuntu "$LOG_DIR"
chown ubuntu:ubuntu "$OUTPUT_DIR"

# Verify .env file exists
if [ ! -f "${PROJECT_DIR}/.env" ]; then
    echo "⚠️  WARNING: .env file not found!"
    echo "   Copy .env.example to .env and configure it:"
    echo "   cp ${PROJECT_DIR}/.env.example ${PROJECT_DIR}/.env"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Copy service file
echo "📋 Installing service file..."
cp "${PROJECT_DIR}/${SERVICE_FILE}" "/etc/systemd/system/${SERVICE_NAME}.service"

# Reload systemd
echo "🔄 Reloading systemd..."
systemctl daemon-reload

# Enable service (auto-start on boot)
echo "✅ Enabling service..."
systemctl enable "$SERVICE_NAME"

# Start service
echo "🚀 Starting service..."
systemctl start "$SERVICE_NAME"

# Wait a moment for service to start
sleep 2

# Check status
echo ""
echo "==== Service Status ===="
systemctl status "$SERVICE_NAME" --no-pager || true

echo ""
echo "==== Installation Complete ===="
echo ""
echo "Useful commands:"
echo "  • View logs:       journalctl -u ${SERVICE_NAME} -f"
echo "  • Check status:    systemctl status ${SERVICE_NAME}"
echo "  • Stop service:    systemctl stop ${SERVICE_NAME}"
echo "  • Start service:   systemctl start ${SERVICE_NAME}"
echo "  • Restart service: systemctl restart ${SERVICE_NAME}"
echo "  • Disable service: systemctl disable ${SERVICE_NAME}"
echo ""
echo "Log files:"
echo "  • Main log:  ${LOG_DIR}/pipeline.log"
echo "  • Error log: ${LOG_DIR}/pipeline-error.log"
echo ""
