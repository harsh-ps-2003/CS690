#!/bin/bash
echo "=== System Memory Information ==="
free -h

echo -e "\n=== Resource Limits (ulimit -a) ==="
ulimit -a

echo -e "\n=== Virtual Memory Limit Specifically ==="
ulimit -v

echo -e "\n=== Check /proc/self/limits ==="
if [ -f /proc/self/limits ]; then
    cat /proc/self/limits
else
    echo "/proc/self/limits not found (not on Linux?)"
fi

echo -e "\n=== Check systemd memory limits ==="
# Check cgroup v1
if [ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
    limit=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
    echo "CGroup v1 memory limit: $limit bytes ($(echo $limit | numfmt --to=iec-i --suffix=B))"
fi

# Check cgroup v2
if [ -f /sys/fs/cgroup/memory.max ]; then
    limit=$(cat /sys/fs/cgroup/memory.max)
    echo "CGroup v2 memory limit: $limit"
fi

# Check if under systemd
if systemctl status $$ &>/dev/null; then
    echo -e "\n=== systemd-cgls output ==="
    systemd-cgls | grep -A5 -B5 $$
fi
