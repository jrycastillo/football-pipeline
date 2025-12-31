#!/bin/bash
#
# Football Video Processing Pipeline - Startup Script
# Starts the polling service with 3 parallel workers and DB mode enabled
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
PARALLEL_WORKERS=3
POLL_INTERVAL=30
JNR_STRIDE=60
LOG_FILE="polling_service.log"
STATUS_SERVER_LOG="status_server.log"

echo "============================================"
echo " Football Video Processing Pipeline"
echo "============================================"
echo ""

# Kill any existing processes
echo "[1/4] Stopping existing processes..."
pkill -9 -f "orchestrator.py" 2>/dev/null || true
pkill -9 -f "pipeline_consolidated.py" 2>/dev/null || true
pkill -9 -f "status_server.py" 2>/dev/null || true
sleep 2

# Clean up temp files
echo "[2/4] Cleaning up temp files..."
rm -rf /tmp/pf_* /tmp/football_* 2>/dev/null || true

# Start status server
echo "[3/4] Starting Status API server on port 8080..."
nohup python3 status_server.py > "$STATUS_SERVER_LOG" 2>&1 &
STATUS_PID=$!
sleep 2

# Check if status server started
if ps -p $STATUS_PID > /dev/null 2>&1; then
    echo "       ✅ Status API running (PID: $STATUS_PID)"
else
    echo "       ⚠️  Status API failed to start, check $STATUS_SERVER_LOG"
fi

# Start polling service
echo "[4/4] Starting Polling Service..."
echo "       - Workers: $PARALLEL_WORKERS"
echo "       - Poll Interval: ${POLL_INTERVAL}s"
echo "       - JNR Stride: $JNR_STRIDE"
echo "       - DB Mode: Enabled"
echo ""

PYTHONUNBUFFERED=1 nohup python3 orchestrator.py \
    --poll \
    --poll_interval $POLL_INTERVAL \
    --jnr_stride $JNR_STRIDE \
    --parallel $PARALLEL_WORKERS \
    > "$LOG_FILE" 2>&1 &

POLL_PID=$!
sleep 3

# Verify polling service started
if ps -p $POLL_PID > /dev/null 2>&1; then
    echo "✅ Polling Service started (PID: $POLL_PID)"
    echo ""
    echo "============================================"
    echo " Services Running:"
    echo "   - Polling Service: $LOG_FILE"
    echo "   - Status API: http://localhost:8080/status"
    echo ""
    echo " Commands:"
    echo "   Monitor logs: tail -f $LOG_FILE"
    echo "   Check status: curl http://localhost:8080/status"
    echo "   Stop all:     pkill -f orchestrator.py"
    echo "============================================"
else
    echo "❌ Failed to start polling service!"
    echo "Check logs: cat $LOG_FILE"
    exit 1
fi
