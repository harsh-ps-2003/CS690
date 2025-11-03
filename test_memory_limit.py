#!/usr/bin/env python3
"""
Test to find the actual memory limit by gradually allocating more memory.
This will help us find if there's a hidden limit we can't see via diagnostics.
"""

import os
import gc
import time
import psutil
import numpy as np

def print_memory():
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / 1e9
    vm = psutil.virtual_memory()
    print(f"  Process: {mem_gb:.2f}GB | System available: {vm.available/1e9:.1f}GB")
    return mem_gb

print("="*70)
print("Memory Limit Test")
print("="*70)
print("This will gradually allocate memory to find the limit.")
print("If there's a hidden 2GB limit, it should get killed around 2GB.")
print("")

arrays = []
chunk_size_gb = 0.1  # Allocate 100MB at a time

try:
    for i in range(50):  # Try to allocate up to 5GB
        target_gb = (i + 1) * chunk_size_gb
        print(f"\nStep {i+1}: Allocating to {target_gb:.1f}GB total...")
        
        # Allocate 100MB
        arr = np.zeros(int(chunk_size_gb * 1e9 / 8), dtype=np.float64)
        arrays.append(arr)
        
        # Touch the memory to ensure it's actually allocated
        arr[0] = 1.0
        arr[-1] = 1.0
        
        # Force Python GC to get accurate reading
        gc.collect()
        time.sleep(0.1)
        
        mem_gb = print_memory()
        
        # Check if we crossed common limit thresholds
        if mem_gb > 1.5 and mem_gb < 1.6:
            print("  ⚠️  Crossed 1.5GB - if there's a ~1.6GB limit, kill might happen soon...")
        elif mem_gb > 2.0 and mem_gb < 2.1:
            print("  ⚠️  Crossed 2.0GB - if there's a 2GB limit, kill should have happened by now!")
        elif mem_gb > 3.0 and mem_gb < 3.1:
            print("  ✅ Crossed 3.0GB - no 2GB limit exists!")
        
        # Small delay to let system catch up
        time.sleep(0.2)
    
    print("\n" + "="*70)
    print(f"✅ SUCCESS: Allocated {len(arrays) * chunk_size_gb:.1f}GB without being killed!")
    print("This means there's NO hard memory limit preventing allocation.")
    print("The issue must be something else (time limit, OOM score, etc.)")
    print("="*70)

except MemoryError as e:
    print("\n" + "="*70)
    print(f"❌ MemoryError after allocating {len(arrays) * chunk_size_gb:.1f}GB")
    print(f"Error: {e}")
    print("This indicates a real memory limit.")
    print("="*70)

except KeyboardInterrupt:
    print("\n" + "="*70)
    print("Interrupted by user")
    print("="*70)

except Exception as e:
    print("\n" + "="*70)
    print(f"❌ Unexpected error: {type(e).__name__}: {e}")
    print("="*70)

finally:
    # Cleanup
    del arrays
    gc.collect()
    print(f"\nFinal memory: {print_memory()}")

