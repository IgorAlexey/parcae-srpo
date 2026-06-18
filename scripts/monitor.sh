#!/bin/bash
# Monitor training on vast.ai instance and fetch checkpoints.
# Run: bash monitor.sh

INSTANCE="ssh3.vast.ai"
PORT="32403"
CKPT_DIR="/home/igoralexey/Downloads/super-pi/checkpoints"
mkdir -p "$CKPT_DIR"

while true; do
    clear
    echo "=== $(date '+%H:%M:%S') ==="

    # Step count and latest metrics
    SSH_CMD="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -p $PORT root@$INSTANCE"
    STEPS=$($SSH_CMD 'grep -c "^step " /workspace/train.log' 2>/dev/null || echo 0)
    LAST=$($SSH_CMD 'grep "^step " /workspace/train.log | tail -1' 2>/dev/null || echo "no data")
    TOTAL=$($SSH_CMD 'grep "total_steps" /workspace/real_train2.py 2>/dev/null | grep -o "[0-9]\+"' 2>/dev/null || echo "?")
    GPU=$($SSH_CMD 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1' 2>/dev/null || echo "offline")
    ERR=$($SSH_CMD 'tail -3 /workspace/train_err.log 2>/dev/null' 2>/dev/null)
    CKPT=$($SSH_CMD 'ls /workspace/checkpoints/ 2>/dev/null | head -5' 2>/dev/null || echo "none")

    echo "Steps: $STEPS / $TOTAL"
    echo "Last:  $LAST"
    echo "GPU:   $GPU"
    echo "CKPT:  $CKPT"
    if [ -n "$ERR" ]; then
        echo "ERR:   $ERR"
    fi

    # Fetch new checkpoints
    REMOTE_CKPTS=$($SSH_CMD 'ls /workspace/checkpoints/step_*.pt 2>/dev/null' 2>/dev/null)
    for rc in $REMOTE_CKPTS; do
        local_name="$CKPT_DIR/$(basename $rc)"
        if [ ! -f "$local_name" ]; then
            scp -o StrictHostKeyChecking=no -P $PORT "root@$INSTANCE:$rc" "$local_name" 2>/dev/null
            echo "Fetched: $(basename $rc)"
        fi
    done

    sleep 300  # every 5 min
done
