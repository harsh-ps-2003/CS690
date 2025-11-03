#!/bin/bash

echo "=================================================================="
echo "Checking Systemd User Slice Limits"
echo "=================================================================="
echo ""

# Get your user ID
USER_ID=$(id -u)
USER_NAME=$(whoami)

echo "User: $USER_NAME (UID: $USER_ID)"
echo ""

echo "1. Check user slice properties:"
echo "-----------------------------------"
systemctl show user-${USER_ID}.slice | grep -iE '(Memory|Task|CPU|IO)' || echo "Cannot read user slice properties"
echo ""

echo "2. Check current session scope properties:"
echo "-----------------------------------"
# Get current session scope
SESSION_SCOPE=$(systemctl status $$ 2>/dev/null | grep -oP 'session-[0-9]+\.scope' | head -1)
if [ -n "$SESSION_SCOPE" ]; then
    echo "Current session scope: $SESSION_SCOPE"
    systemctl show user.slice/user-${USER_ID}.slice/${SESSION_SCOPE} | grep -iE '(Memory|Task)' || echo "Cannot read session scope"
else
    echo "Not in a systemd session scope"
fi
echo ""

echo "3. Check user@ service limits:"
echo "-----------------------------------"
systemctl --user show | grep -iE '(Memory|Default)' || echo "Cannot query user service manager"
echo ""

echo "4. Check if there's a user-specific override:"
echo "-----------------------------------"
if [ -d "/etc/systemd/system/user-${USER_ID}.slice.d/" ]; then
    echo "Found user-specific overrides:"
    ls -la /etc/systemd/system/user-${USER_ID}.slice.d/
    cat /etc/systemd/system/user-${USER_ID}.slice.d/*.conf 2>/dev/null || echo "Cannot read override files"
else
    echo "No user-specific slice overrides found"
fi
echo ""

echo "5. Check global user slice configuration:"
echo "-----------------------------------"
if [ -f "/etc/systemd/system/user-.slice.d/override.conf" ]; then
    echo "Found global user slice override:"
    cat /etc/systemd/system/user-.slice.d/override.conf
else
    echo "No global user slice override"
fi
echo ""

echo "6. Check /etc/systemd/system.conf for default limits:"
echo "-----------------------------------"
grep -E '^(DefaultMemory|DefaultTasksMax)' /etc/systemd/system.conf 2>/dev/null || echo "No defaults in system.conf (or no access)"
echo ""

echo "7. Check OOM score of current shell:"
echo "-----------------------------------"
if [ -f "/proc/$$/oom_score" ]; then
    echo "OOM score: $(cat /proc/$$/oom_score)"
    echo "OOM score adj: $(cat /proc/$$/oom_score_adj)"
    echo "(Higher score = more likely to be killed. 1000 = always killed first)"
else
    echo "Cannot read OOM score"
fi
echo ""

echo "8. Check for PAM limits:"
echo "-----------------------------------"
if [ -f "/etc/security/limits.conf" ]; then
    echo "Checking /etc/security/limits.conf for user limits:"
    grep -v '^#' /etc/security/limits.conf | grep -E "(^$USER_NAME|^@|^\*)" || echo "No limits found for user"
else
    echo "No PAM limits file"
fi
echo ""

if [ -d "/etc/security/limits.d/" ]; then
    echo "Checking /etc/security/limits.d/:"
    for f in /etc/security/limits.d/*.conf; do
        if [ -f "$f" ]; then
            echo "File: $f"
            grep -v '^#' "$f" | grep -E "(^$USER_NAME|^@|^\*)" || echo "  (no relevant limits)"
        fi
    done
else
    echo "No limits.d directory"
fi
echo ""

echo "9. Check actual cgroup controllers enabled:"
echo "-----------------------------------"
CGROUP_PATH="/sys/fs/cgroup/user.slice/user-${USER_ID}.slice/session-${SESSION_SCOPE}"
if [ -d "$CGROUP_PATH" ]; then
    echo "Cgroup path: $CGROUP_PATH"
    echo "Controllers:"
    cat ${CGROUP_PATH}/cgroup.controllers 2>/dev/null || echo "Cannot read controllers"
    echo ""
    echo "Memory stats:"
    cat ${CGROUP_PATH}/memory.stat 2>/dev/null | head -20 || echo "Cannot read memory stats"
else
    echo "Cgroup path not found or no access"
fi
echo ""

echo "10. Test: Try to allocate 3GB of memory:"
echo "-----------------------------------"
echo "Running Python to allocate memory..."
python3 -c "
import sys
print('Attempting to allocate 3GB...')
try:
    # Allocate 3GB
    big_list = [0] * (3 * 1024 * 1024 * 1024 // 8)  # 3GB of integers
    print('✅ Successfully allocated 3GB')
    sys.exit(0)
except MemoryError as e:
    print(f'❌ MemoryError: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ Error: {type(e).__name__}: {e}')
    sys.exit(1)
"
ALLOC_RESULT=$?
if [ $ALLOC_RESULT -eq 0 ]; then
    echo "Memory allocation test: PASSED"
else
    echo "Memory allocation test: FAILED (exit code: $ALLOC_RESULT)"
    echo "This suggests there IS a hidden memory limit!"
fi
echo ""

echo "=================================================================="
echo "Diagnosis complete!"
echo "=================================================================="

