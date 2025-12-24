#!/bin/bash
# Wait for the current batch processor (PID 5841) to finish
echo "[chain] Waiting for PID 5841 to finish..."
tail --pid=5841 -f /dev/null

echo "[chain] Previous batch finished. Starting Final Batch..."
nohup python3 -u batch_processor.py > batch_run_final.log 2>&1 &
echo "[chain] Final Batch PID: $!"
