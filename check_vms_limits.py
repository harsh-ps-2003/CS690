#!/usr/bin/env python3
"""Check system VMS limits and memory configuration"""

import os
import resource
import psutil

def format_bytes(bytes_val):
    """Format bytes to human readable"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"

def check_limits():
    print("=== System Memory Information ===")
    vm = psutil.virtual_memory()
    print(f"Total RAM: {format_bytes(vm.total)}")
    print(f"Available RAM: {format_bytes(vm.available)}")
    print(f"Used RAM: {format_bytes(vm.used)} ({vm.percent:.1f}%)")
    
    print("\n=== Process Resource Limits ===")
    
    # Check various resource limits
    limits_to_check = [
        (resource.RLIMIT_AS, "RLIMIT_AS (Virtual Memory)"),
        (resource.RLIMIT_DATA, "RLIMIT_DATA (Data Segment)"),
        (resource.RLIMIT_STACK, "RLIMIT_STACK (Stack Size)"),
        (resource.RLIMIT_RSS, "RLIMIT_RSS (Resident Set Size)"),
    ]
    
    for limit_type, name in limits_to_check:
        try:
            soft, hard = resource.getrlimit(limit_type)
            soft_str = "unlimited" if soft == resource.RLIM_INFINITY else format_bytes(soft)
            hard_str = "unlimited" if hard == resource.RLIM_INFINITY else format_bytes(hard)
            print(f"{name}:")
            print(f"  Soft limit: {soft_str}")
            print(f"  Hard limit: {hard_str}")
        except Exception as e:
            print(f"{name}: Error - {e}")
    
    print("\n=== Current Process Memory ===")
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"RSS (Resident Set Size): {format_bytes(mem_info.rss)}")
    print(f"VMS (Virtual Memory Size): {format_bytes(mem_info.vms)}")
    
    # Check systemd memory limits if available
    print("\n=== Checking systemd limits ===")
    try:
        # Check if we're running under systemd
        cgroup_path = f"/proc/{os.getpid()}/cgroup"
        if os.path.exists(cgroup_path):
            with open(cgroup_path, 'r') as f:
                cgroup_content = f.read()
                if 'systemd' in cgroup_content:
                    print("Process appears to be running under systemd")
                    # Try to check memory limits
                    limit_files = [
                        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
                        "/sys/fs/cgroup/memory.max",  # cgroup v2
                    ]
                    for limit_file in limit_files:
                        if os.path.exists(limit_file):
                            try:
                                with open(limit_file, 'r') as f:
                                    limit = f.read().strip()
                                    if limit.isdigit():
                                        limit_val = int(limit)
                                        if limit_val < (1 << 62):  # Less than huge number
                                            print(f"Memory limit from {limit_file}: {format_bytes(limit_val)}")
                                    else:
                                        print(f"Memory limit from {limit_file}: {limit}")
                            except:
                                pass
                else:
                    print("Not running under systemd")
    except Exception as e:
        print(f"Could not check systemd limits: {e}")
    
    # Check ulimit
    print("\n=== Shell ulimit -a equivalent ===")
    import subprocess
    try:
        result = subprocess.run(['sh', '-c', 'ulimit -a'], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
    except:
        print("Could not run ulimit -a")

if __name__ == "__main__":
    check_limits()
