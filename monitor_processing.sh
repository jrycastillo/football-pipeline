#!/bin/bash
# Live monitoring script for pipeline processing

clear
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "         FOOTBALL PIPELINE - LIVE MONITORING"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    # Clear previous output (keep header)
    tput cup 6 0
    tput ed

    # Current time
    echo "🕐 Time: $(date '+%H:%M:%S')"
    echo ""

    # Active processes
    ACTIVE=$(ps aux | grep pipeline_consolidated | grep -v grep | wc -l | tr -d ' ')
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 PROCESSING STATUS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "   Active pipelines: $ACTIVE"
    echo ""

    # Frame progress
    LATEST_FRAME=$(tail -50 /Users/ronan/Babak/output/pipeline.log 2>/dev/null | grep "Processing Frame" | tail -1 | grep -o "Frame [0-9]*" | grep -o "[0-9]*")
    if [ -n "$LATEST_FRAME" ]; then
        echo "   Latest frame: $LATEST_FRAME"

        # Estimate total (assume 25fps, 10min = 15000 frames)
        PERCENT=$((LATEST_FRAME * 100 / 15000))
        echo "   Progress: ~$PERCENT% (assuming 10min video)"

        # Progress bar
        BARS=$((PERCENT / 2))
        printf "   ["
        for i in $(seq 1 50); do
            if [ $i -le $BARS ]; then
                printf "█"
            else
                printf "░"
            fi
        done
        printf "] $PERCENT%%\n"
    else
        echo "   Frame: Waiting..."
    fi
    echo ""

    # Recent activity
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔍 RECENT ACTIVITY (Last 10 seconds)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Jersey detections
    JERSEYS=$(tail -200 /Users/ronan/Babak/output/pipeline.log 2>/dev/null | grep -c "Registered Jersey")
    echo "   Jersey recognitions: $JERSEYS (total in recent logs)"

    # Latest jerseys
    LATEST_JERSEY=$(tail -100 /Users/ronan/Babak/output/pipeline.log 2>/dev/null | grep "Registered Jersey" | tail -1 | grep -o "#[0-9]*")
    if [ -n "$LATEST_JERSEY" ]; then
        echo "   Latest detected: $LATEST_JERSEY"
    fi
    echo ""

    # Completions
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ COMPLETED VIDEOS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Find completed videos in last hour
    COMPLETED=$(find /Users/ronan/Babak/output -name "player_stats.json" -mmin -60 2>/dev/null | wc -l | tr -d ' ')
    echo "   Completed in last hour: $COMPLETED"

    if [ "$COMPLETED" -gt 0 ]; then
        echo ""
        echo "   Recent completions:"
        find /Users/ronan/Babak/output -name "player_stats.json" -mmin -60 -exec sh -c 'DIR=$(dirname {}); ID=$(basename "$DIR"); echo "      • Video $ID"' \; 2>/dev/null | head -5
    fi
    echo ""

    # Health check
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "💚 HEALTH METRICS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ -f /Users/ronan/Babak/output/health_snapshot.json ]; then
        PROCESSED=$(cat /Users/ronan/Babak/output/health_snapshot.json 2>/dev/null | grep "videos_processed" | grep -o "[0-9]*" | head -1)
        FAILED=$(cat /Users/ronan/Babak/output/health_snapshot.json 2>/dev/null | grep "videos_failed" | grep -o "[0-9]*" | head -1)
        RUNNING=$(cat /Users/ronan/Babak/output/health_snapshot.json 2>/dev/null | grep "videos_running" | grep -o "[0-9]*" | head -1)

        echo "   Processed: ${PROCESSED:-0}"
        echo "   Failed: ${FAILED:-0}"
        echo "   Running: ${RUNNING:-0}"
    fi
    echo ""

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "   Refreshing in 5 seconds... (Ctrl+C to exit)"
    echo ""

    sleep 5
done
