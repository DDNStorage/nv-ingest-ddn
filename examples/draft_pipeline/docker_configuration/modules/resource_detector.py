#!/usr/bin/env python3
"""
Resource Detection Module
Detects system resources (CPU, RAM, GPU) for optimal Milvus configuration
"""

import os
import subprocess
import psutil
import json
from typing import Dict, List, Any


class ResourceDetector:
    """Detects and analyzes system resources"""
    
    @staticmethod
    def get_cpu_info() -> Dict[str, Any]:
        """Get CPU information"""
        cpu_count = psutil.cpu_count(logical=False)  # Physical cores
        cpu_threads = psutil.cpu_count(logical=True)  # Logical cores
        
        # Get CPU model
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if line.strip().startswith('model name'):
                        cpu_model = line.split(':', 1)[1].strip()
                        break
                else:
                    cpu_model = "Unknown"
        except:
            cpu_model = "Unknown"
        
        return {
            'physical_cores': cpu_count,
            'logical_cores': cpu_threads,
            'model': cpu_model,
            'numa_nodes': 1  # Simplified - would need numactl for accurate detection
        }
    
    @staticmethod
    def get_memory_info() -> Dict[str, Any]:
        """Get memory information"""
        mem = psutil.virtual_memory()
        
        return {
            'total_gb': round(mem.total / (1024**3), 2),
            'available_gb': round(mem.available / (1024**3), 2),
            'used_gb': round(mem.used / (1024**3), 2),
            'percent_used': mem.percent
        }
    
    @staticmethod
    def get_gpu_info() -> List[Dict[str, Any]]:
        """Get GPU information using nvidia-smi"""
        gpus = []
        
        # Check if CPU mode is forced
        if os.environ.get('FORCE_CPU_MODE', '').lower() == 'true':
            return gpus
        
        try:
            # Check if nvidia-smi is available
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free,compute_cap', 
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                check=True
            )
            
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 5:
                        gpus.append({
                            'index': int(parts[0]),
                            'name': parts[1],
                            'memory_total_mb': int(parts[2]),
                            'memory_free_mb': int(parts[3]),
                            'compute_capability': parts[4]
                        })
        except (subprocess.CalledProcessError, FileNotFoundError):
            # No GPUs or nvidia-smi not available
            pass
        
        return gpus
    
    @staticmethod
    def get_storage_info() -> Dict[str, Any]:
        """Get storage information for the current directory"""
        disk = psutil.disk_usage('/')
        
        # Try to detect storage type (SSD vs HDD)
        storage_type = "unknown"
        try:
            # This is a simple heuristic - rotational = 0 usually means SSD
            with open('/sys/block/sda/queue/rotational', 'r') as f:
                storage_type = "HDD" if f.read().strip() == "1" else "SSD"
        except:
            pass
        
        return {
            'total_gb': round(disk.total / (1024**3), 2),
            'free_gb': round(disk.free / (1024**3), 2),
            'used_gb': round(disk.used / (1024**3), 2),
            'percent_used': disk.percent,
            'type': storage_type
        }
    
    @staticmethod
    def estimate_network_bandwidth() -> Dict[str, float]:
        """Estimate network bandwidth (simplified)"""
        # This is a simplified estimation
        # In production, you might want to run actual bandwidth tests
        return {
            'estimated_mbps': 1000,  # Default to 1Gbps
            'note': 'Estimated value - actual bandwidth may vary'
        }
    
    @classmethod
    def detect_all(cls) -> Dict[str, Any]:
        """Detect all system resources"""
        return {
            'cpu': cls.get_cpu_info(),
            'memory': cls.get_memory_info(),
            'gpus': cls.get_gpu_info(),
            'storage': cls.get_storage_info(),
            'network': cls.estimate_network_bandwidth()
        }
    
    @classmethod
    def print_summary(cls, resources: Dict[str, Any]) -> None:
        """Print a summary of detected resources"""
        print("=== System Resources Detected ===")
        print(f"\nCPU:")
        print(f"  Model: {resources['cpu']['model']}")
        print(f"  Physical Cores: {resources['cpu']['physical_cores']}")
        print(f"  Logical Cores: {resources['cpu']['logical_cores']}")
        
        print(f"\nMemory:")
        print(f"  Total: {resources['memory']['total_gb']} GB")
        print(f"  Available: {resources['memory']['available_gb']} GB")
        
        print(f"\nGPUs: {len(resources['gpus'])}")
        for gpu in resources['gpus']:
            print(f"  GPU {gpu['index']}: {gpu['name']}")
            print(f"    Memory: {gpu['memory_total_mb']} MB")
        
        print(f"\nStorage:")
        print(f"  Type: {resources['storage']['type']}")
        print(f"  Free: {resources['storage']['free_gb']} GB")


if __name__ == "__main__":
    # Test the resource detector
    detector = ResourceDetector()
    resources = detector.detect_all()
    detector.print_summary(resources)
    
    # Also save to JSON for debugging
    with open('detected_resources.json', 'w') as f:
        json.dump(resources, f, indent=2)