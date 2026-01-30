#!/bin/bash
# Restart orchestrator with proper .env credentials

echo "🔄 Restarting Football Pipeline Orchestrator..."
echo ""

# Stop existing processes
echo "1. Stopping existing orchestrator..."
pkill -f "orchestrator.py --poll" 2>/dev/null
sleep 2

# Verify .env exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "Run: cp .env.example .env and configure it"
    exit 1
fi

echo "✅ Found .env file"
echo ""

# Verify credentials are set
source .env
if [ -z "$MYSQL_PASSWORD" ]; then
    echo "❌ Error: MYSQL_PASSWORD not set in .env"
    exit 1
fi

echo "✅ MYSQL_PASSWORD is configured"
echo ""

# Clear old polling log
echo "2. Clearing old polling log..."
> polling_service.log

# Start orchestrator in background
echo "3. Starting orchestrator..."
nohup python -u orchestrator.py --poll --parallel 3 >> polling_service.log 2>&1 &
ORCH_PID=$!

echo "✅ Started with PID: $ORCH_PID"
echo ""

# Wait for startup
echo "4. Waiting for startup (8 seconds)..."
sleep 8

# Check if running
if ps -p $ORCH_PID > /dev/null; then
    echo "✅ Orchestrator is running!"
else
    echo "❌ Orchestrator failed to start. Check polling_service.log"
    tail -20 polling_service.log
    exit 1
fi

# Show initial log
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Initial Log Output:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
head -30 polling_service.log
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ Orchestrator restarted successfully!"
echo ""
echo "Monitor with:"
echo "  • tail -f polling_service.log"
echo "  • ./monitor_processing.sh"
echo "  • python check_db_status.py"
echo ""
