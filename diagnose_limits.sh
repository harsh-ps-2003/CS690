#!/bin/bash

echo "=================================================================="
echo "Diagnosing System Resource Limits"
echo "=================================================================="
echo ""

echo "1. Shell Resource Limits (ulimit):"
echo "-----------------------------------"
ulimit -a
echo ""

echo "2. Cgroup v2 Memory Limits:"
echo "-----------------------------------"
# Check if we're in a cgroup v2 system
if [ -f /sys/fs/cgroup/memory.max ]; then
    echo "Cgroup v2 detected"
    echo "Memory limit: $(cat /sys/fs/cgroup/memory.max)"
    echo "Memory current: $(cat /sys/fs/cgroup/memory.current)"
    echo "Memory high (soft limit): $(cat /sys/fs/cgroup/memory.high 2>/dev/null || echo 'N/A')"
    echo "Swap limit: $(cat /sys/fs/cgroup/memory.swap.max 2>/dev/null || echo 'N/A')"
elif [ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
    echo "Cgroup v1 detected"
    echo "Memory limit: $(cat /sys/fs/cgroup/memory/memory.limit_in_bytes) bytes"
    echo "Memory usage: $(cat /sys/fs/cgroup/memory/memory.usage_in_bytes) bytes"
else
    echo "No cgroup memory limits found (or no access)"
fi
echo ""

echo "3. Check /dev/shm (PyTorch shared memory):"
echo "-----------------------------------"
df -h /dev/shm
echo ""

echo "4. Check /tmp space:"
echo "-----------------------------------"
df -h /tmp
echo ""

echo "5. System Memory Info:"
echo "-----------------------------------"
free -h
echo ""

echo "6. Check for Slurm/PBS job limits:"
echo "-----------------------------------"
if command -v squeue &> /dev/null; then
    echo "Slurm detected - checking job info:"
    squeue -u $USER -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %.10l %.10m %R"
    echo ""
    if [ -n "$SLURM_JOB_ID" ]; then
        echo "Current job limits:"
        scontrol show job $SLURM_JOB_ID | grep -E "(TimeLimit|Memory|CPU)"
    fi
elif command -v qstat &> /dev/null; then
    echo "PBS/Torque detected - checking job info:"
    qstat -u $USER
else
    echo "No job scheduler detected (Slurm/PBS)"
fi
echo ""

echo "7. Process-specific cgroup info:"
echo "-----------------------------------"
echo "Cgroup membership:"
cat /proc/self/cgroup
echo ""

# Try to find the actual cgroup path
CGROUP_PATH=$(cat /proc/self/cgroup | grep -E '^0:' | cut -d: -f3)
if [ -n "$CGROUP_PATH" ]; then
    echo "Trying to read cgroup limits from: /sys/fs/cgroup${CGROUP_PATH}"
    if [ -f "/sys/fs/cgroup${CGROUP_PATH}/memory.max" ]; then
        echo "Memory max: $(cat /sys/fs/cgroup${CGROUP_PATH}/memory.max)"
        echo "Memory current: $(cat /sys/fs/cgroup${CGROUP_PATH}/memory.current 2>/dev/null || echo 'N/A')"
    elif [ -f "/sys/fs/cgroup${CGROUP_PATH}/memory.limit_in_bytes" ]; then
        LIMIT=$(cat /sys/fs/cgroup${CGROUP_PATH}/memory.limit_in_bytes)
        echo "Memory limit: $LIMIT bytes ($(numfmt --to=iec $LIMIT 2>/dev/null || echo $LIMIT))"
    else
        echo "No memory limits in this cgroup path"
    fi
fi
echo ""

echo "8. Check systemd resource limits:"
echo "-----------------------------------"
if command -v systemctl &> /dev/null; then
    # Get the systemd unit we're running under
    UNIT=$(systemctl status $$ 2>/dev/null | grep -oP '(?<=unit:)[^()]+' | tr -d ' ' | head -1)
    if [ -n "$UNIT" ]; then
        echo "Running under systemd unit: $UNIT"
        echo "Unit properties:"
        systemctl show --property=MemoryLimit --property=MemoryMax --property=MemoryHigh "$UNIT" 2>/dev/null || echo "Cannot read unit properties"
    else
        echo "Not running under a systemd unit (or no access)"
    fi
else
    echo "systemd not available"
fi
echo ""

echo "9. File descriptor limits:"
echo "-----------------------------------"
echo "Soft limit: $(ulimit -Sn)"
echo "Hard limit: $(ulimit -Hn)"
echo "Currently open (for this shell): $(ls /proc/$$/fd | wc -l)"
echo ""

echo "10. Environment variables that might indicate limits:"
echo "-----------------------------------"
env | grep -iE '(MEM|LIMIT|QUOTA|SLURM|PBS|CGROUP)' || echo "No relevant env vars found"
echo ""

echo "=================================================================="
echo "Diagnosis complete!"
echo "=================================================================="

