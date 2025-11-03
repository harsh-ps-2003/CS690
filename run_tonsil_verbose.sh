#!/bin/bash

echo "=================================================================="
echo "Running scmodal_tonsil.py with verbose error reporting"
echo "=================================================================="
echo ""
echo "Setting CUDA_LAUNCH_BLOCKING=1 to get better CUDA error messages"
echo "This makes CUDA operations synchronous so we can see exact error locations"
echo ""

# Set environment variables for better error reporting
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run with Python unbuffered output (-u flag) so we see messages immediately
python -u scmodal_tonsil.py

# Capture exit code
EXIT_CODE=$?

echo ""
echo "=================================================================="
if [ $EXIT_CODE -eq 137 ]; then
    echo "Process was KILLED (exit code 137 = SIGKILL from OOM killer)"
    echo "This means the OS killed the process due to resource exhaustion"
    echo ""
    echo "Check system logs with: dmesg | tail -50"
    echo "or: journalctl -n 100 | grep -i kill"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS! Training completed without errors"
else
    echo "Process exited with code: $EXIT_CODE"
fi
echo "=================================================================="

exit $EXIT_CODE

