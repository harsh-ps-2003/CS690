#!/usr/bin/env python3
"""Test script to verify VMS memory management improvements"""

import os
import sys
import gc
import numpy as np
import time

# Import the malloc_trim function
try:
    import ctypes
    _libc = ctypes.CDLL("libc.so.6")
    def malloc_trim():
        """Return freed memory to OS to reduce VMS"""
        try:
            _libc.malloc_trim(0)
            return True
        except Exception:
            return False
except Exception:
    def malloc_trim():
        return False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil not available. Install with: pip install psutil")
    sys.exit(1)

def get_memory_stats():
    """Get current memory statistics"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    return {
        'rss_mb': mem_info.rss / 1e6,
        'vms_mb': mem_info.vms / 1e6,
        'rss_gb': mem_info.rss / 1e9,
        'vms_gb': mem_info.vms / 1e9,
    }

def print_memory(prefix=""):
    """Print memory statistics"""
    stats = get_memory_stats()
    print(f"{prefix}RSS: {stats['rss_mb']:.1f}MB ({stats['rss_gb']:.2f}GB) | VMS: {stats['vms_mb']:.1f}MB ({stats['vms_gb']:.2f}GB)")
    return stats

def test_memory_allocation():
    """Test memory allocation and cleanup behavior"""
    print("=== Testing Memory Allocation and VMS Management ===\n")
    
    print_memory("Initial: ")
    
    # Allocate large array
    print("\n1. Allocating 1GB array...")
    data1 = np.random.rand(1024 * 1024 * 128).astype(np.float32)  # ~1GB
    stats1 = print_memory("After 1GB alloc: ")
    
    # Allocate another large array
    print("\n2. Allocating another 1GB array...")
    data2 = np.random.rand(1024 * 1024 * 128).astype(np.float32)  # ~1GB
    stats2 = print_memory("After 2GB total: ")
    
    # Delete arrays
    print("\n3. Deleting arrays...")
    del data1
    del data2
    stats3 = print_memory("After del (no GC): ")
    
    # Run garbage collection
    print("\n4. Running gc.collect()...")
    gc.collect()
    stats4 = print_memory("After gc.collect(): ")
    
    # Run malloc_trim
    print("\n5. Running malloc_trim()...")
    success = malloc_trim()
    if success:
        print("   malloc_trim() executed successfully")
    else:
        print("   malloc_trim() not available on this system")
    stats5 = print_memory("After malloc_trim: ")
    
    # Analysis
    print("\n=== Analysis ===")
    print(f"VMS after allocation: {stats2['vms_gb']:.2f}GB")
    print(f"VMS after del+gc: {stats4['vms_gb']:.2f}GB (Δ{stats4['vms_gb']-stats2['vms_gb']:+.2f}GB)")
    print(f"VMS after malloc_trim: {stats5['vms_gb']:.2f}GB (Δ{stats5['vms_gb']-stats4['vms_gb']:+.2f}GB)")
    
    if stats5['vms_gb'] < stats4['vms_gb']:
        print("\n✅ SUCCESS: malloc_trim() reduced VMS!")
    else:
        print("\n⚠️  malloc_trim() did not reduce VMS (may not be supported on this system)")
    
    # Test repeated allocations
    print("\n\n=== Testing Repeated Allocations ===")
    print("This simulates the training loop behavior...\n")
    
    initial_stats = get_memory_stats()
    
    for i in range(5):
        print(f"\nIteration {i+1}:")
        # Allocate
        data = np.random.rand(1024 * 1024 * 64).astype(np.float32)  # 512MB
        stats_alloc = print_memory(f"  After alloc: ")
        
        # Process (simulate computation)
        result = np.sum(data)
        
        # Cleanup
        del data
        gc.collect()
        malloc_trim()
        stats_clean = print_memory(f"  After cleanup: ")
        
        time.sleep(0.5)  # Brief pause
    
    final_stats = get_memory_stats()
    print(f"\n=== Final Analysis ===")
    print(f"Initial VMS: {initial_stats['vms_gb']:.2f}GB")
    print(f"Final VMS: {final_stats['vms_gb']:.2f}GB")
    print(f"VMS Growth: {final_stats['vms_gb']-initial_stats['vms_gb']:+.2f}GB")
    
    if final_stats['vms_gb'] - initial_stats['vms_gb'] < 1.0:
        print("\n✅ Good: VMS growth is controlled (<1GB)")
    else:
        print("\n⚠️  Warning: Significant VMS growth detected")

if __name__ == "__main__":
    test_memory_allocation()
