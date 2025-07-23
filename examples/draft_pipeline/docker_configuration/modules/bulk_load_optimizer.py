#!/usr/bin/env python3
"""
Bulk Load Optimizer Module
Calculates optimal Milvus configuration for bulk loading operations
"""

import math
from typing import Dict, Any, Optional


class BulkLoadOptimizer:
    """Optimizes Milvus configuration for bulk load operations"""
    
    # Constants for calculations
    EMBEDDING_DIM = 2048  # Default embedding dimension
    BYTES_PER_FLOAT = 4
    OVERHEAD_FACTOR = 1.5  # Memory overhead for processing
    
    def __init__(self, resources: Dict[str, Any]):
        """Initialize with detected resources"""
        self.resources = resources
        self.cpu_cores = resources['cpu']['physical_cores']
        self.cpu_threads = resources['cpu']['logical_cores']
        self.total_ram_gb = resources['memory']['total_gb']
        self.available_ram_gb = resources['memory']['available_gb']
        self.gpus = resources['gpus']
        self.gpu_count = len(self.gpus)
    
    def calculate_auto_mode(self) -> Dict[str, Any]:
        """
        Auto mode: Optimized for maximum bulk ingestion and indexing performance
        Focuses resources on data ingestion and index building, minimal query capacity
        """
        config = {}
        
        # System resource analysis
        usable_ram_gb = self.total_ram_gb * 0.9  # Reserve 10% for OS
        usable_cpu_cores = self.cpu_cores
        
        # Determine optimal segment size based on available RAM
        # Larger segments = better bulk loading performance
        if usable_ram_gb < 32:
            segment_size_mb = 1024  # 1GB segments for low RAM systems
            segment_seal_proportion = 0.75
        elif usable_ram_gb < 64:
            segment_size_mb = 2048  # 2GB segments for medium RAM
            segment_seal_proportion = 0.8
        elif usable_ram_gb < 128:
            segment_size_mb = 4096  # 4GB segments for high RAM
            segment_seal_proportion = 0.85
        else:
            segment_size_mb = 8192  # 8GB segments for very high RAM systems
            segment_seal_proportion = 0.9
        
        # Calculate node counts with focus on ingestion/indexing
        if self.gpu_count > 0:
            # GPU-accelerated mode
            # Index nodes: 2-3 per GPU (GPUs are very efficient at indexing)
            config['index_nodes'] = min(self.gpu_count * 3, 16)
            
            # Data nodes: 2x index nodes for optimal pipeline flow
            config['data_nodes'] = min(config['index_nodes'] * 2, 24)
            
            # Query nodes: Absolute minimum
            config['query_nodes'] = 1
            
            # GPU memory configuration
            min_gpu_memory_mb = min(gpu['memory_total_mb'] for gpu in self.gpus)
            config['gpu'] = {
                'init_mem_size_mb': int(min_gpu_memory_mb * 0.7),  # 70% initial for indexing
                'max_mem_size_mb': int(min_gpu_memory_mb * 0.9)    # 90% max utilization
            }
        else:
            # CPU-only mode
            # Index nodes: Limited by CPU compute capability
            config['index_nodes'] = min(max(4, usable_cpu_cores // 4), 16)
            
            # Data nodes: 1.5x index nodes for CPU mode
            config['data_nodes'] = min(int(config['index_nodes'] * 1.5), 24)
            
            # Query nodes: Absolute minimum
            config['query_nodes'] = 1
        
        # Validate total node count against available resources
        total_nodes = config['data_nodes'] + config['index_nodes'] + config['query_nodes']
        
        # Each node needs minimum 2GB RAM, but prefer 4GB for bulk operations
        min_ram_per_node_gb = 2
        preferred_ram_per_node_gb = 4
        max_nodes_by_ram = int(usable_ram_gb / min_ram_per_node_gb)
        preferred_nodes_by_ram = int(usable_ram_gb / preferred_ram_per_node_gb)
        
        # Adjust node counts if necessary
        if total_nodes > max_nodes_by_ram:
            # Critical: Not enough RAM even for minimum configuration
            scale_factor = max_nodes_by_ram / total_nodes
            config['data_nodes'] = max(2, int(config['data_nodes'] * scale_factor))
            config['index_nodes'] = max(1, int(config['index_nodes'] * scale_factor))
            config['query_nodes'] = 1
            print(f"Warning: Scaled down nodes due to memory constraints. Total nodes: {total_nodes} -> {max_nodes_by_ram}")
        elif total_nodes > preferred_nodes_by_ram:
            # Sub-optimal: Can run but with reduced memory per node
            print(f"Warning: Running with reduced memory per node. Consider adding more RAM for optimal performance.")
        
        # Recalculate total nodes after potential adjustment
        total_nodes = config['data_nodes'] + config['index_nodes'] + config['query_nodes']
        
        # Memory allocation strategy for bulk ingestion
        # Prioritize data nodes and index nodes, minimal for query nodes
        data_node_weight = 0.55  # 55% for data ingestion
        index_node_weight = 0.40  # 40% for index building
        query_node_weight = 0.05  # 5% for minimal query capability
        
        # Calculate actual memory allocation
        total_memory_mb = usable_ram_gb * 1024
        
        # Ensure query nodes don't exceed 4GB each
        query_memory_per_node = min(4096, int((total_memory_mb * query_node_weight) / config['query_nodes']))
        query_memory_total = query_memory_per_node * config['query_nodes']
        
        # Redistribute remaining memory to data and index nodes
        remaining_memory = total_memory_mb - query_memory_total
        data_memory_total = int(remaining_memory * (data_node_weight / (data_node_weight + index_node_weight)))
        index_memory_total = remaining_memory - data_memory_total
        
        config['memory_allocation'] = {
            'data_node_memory_mb': int(data_memory_total / config['data_nodes']),
            'index_node_memory_mb': int(index_memory_total / config['index_nodes']),
            'query_node_memory_mb': query_memory_per_node,
            'total_allocated_gb': usable_ram_gb
        }
        
        # For compatibility with existing code
        config['memory_per_node_mb'] = int(total_memory_mb / total_nodes)
        
        # Parallelism settings optimized for bulk operations
        config['parallelism'] = {
            # Data sync parallelism: maximize based on CPU cores
            'data_sync_parallel': min(usable_cpu_cores * 4, 2048),
            
            # Build index parallelism: match index nodes
            'build_index_parallel': config['index_nodes'],
            
            # Flush parallelism: scale with data nodes
            'flush_parallel': min(config['data_nodes'] * 2, 64),
            
            # Compaction: moderate during bulk load
            'compaction_parallel': min(usable_cpu_cores // 2, 16),
            
            # Import concurrent tasks: scale with data nodes
            'import_concurrent_tasks': min(config['data_nodes'] * 2, 32),
            
            # Channel parallelism for data nodes
            'channel_parallel': min(config['data_nodes'], 16)
        }
        
        # Segment configuration optimized for bulk loading
        config['segment'] = {
            'max_size_mb': segment_size_mb,
            'seal_proportion': segment_seal_proportion,
            'insert_buffer_size': 67108864,  # 64MB insert buffer
            'delete_buffer_size': 67108864,  # 64MB delete buffer
            'sync_period': 300,  # 5 minutes sync period during bulk load
            'compaction_interval': 3600  # 1 hour compaction interval
        }
        
        # Bulk insert specific settings
        config['bulk_insert'] = {
            'max_file_size_mb': 16384,  # 16GB max file size
            'batch_size': 100000,  # 100k vectors per batch
            'wait_for_index': False,  # Don't wait for indexing
            'max_import_pending_tasks': config['data_nodes'] * 2,
            'import_task_timeout': 3600  # 1 hour timeout
        }
        
        # Thread pool configuration
        config['thread_pools'] = {
            'knowhere_thread_pool_ratio': min(16, self.cpu_threads // 2),
            'disk_thread_pool_ratio': min(usable_cpu_cores, 16),
            'build_parallel_rate': 1.0,  # Use all available resources
            'search_thread_pool_ratio': 1  # Minimal for bulk loading
        }
        
        # I/O and buffer configurations
        config['io_config'] = {
            'mmap_enabled': usable_ram_gb < 64,  # Use mmap for low RAM systems
            'read_ahead_size_mb': min(256, int(usable_ram_gb * 0.01)),  # 1% of RAM, max 256MB
            'write_buffer_size_mb': 128,  # Write buffer per data node
            'wal_buffer_size_mb': 256,  # WAL buffer size
            'message_queue_size_mb': 512,  # Pulsar message queue buffer
            'disk_io_concurrency': min(32, usable_cpu_cores),
            'io_thread_pool_size': min(64, usable_cpu_cores * 2)
        }
        
        # Index building specific configurations
        config['index_config'] = {
            'index_build_threads': config['index_nodes'] * 2,
            'max_index_file_size_mb': 1024,  # 1GB index files
            'enable_disk_index': self.gpu_count == 0,  # Enable disk index for CPU mode
            'index_cache_size_mb': int(usable_ram_gb * 0.1 * 1024),  # 10% of RAM for index cache
            'index_type': 'IVF_FLAT' if self.gpu_count > 0 else 'IVF_SQ8',  # GPU vs CPU index types
            'nlist': 4096,  # Number of clusters for IVF index
            'nprobe': 16,  # Number of clusters to search
            'enable_gpu_index': self.gpu_count > 0
        }
        
        # Resource limits and quotas
        config['resource_limits'] = {
            'max_insert_rate_mb_per_sec': -1,  # Unlimited during bulk load
            'max_flush_rate_mb_per_sec': -1,   # Unlimited
            'max_compaction_rate_mb_per_sec': 1000,  # 1GB/s compaction limit
            'max_build_index_rate': -1,  # Unlimited
            'ttl_check_interval': -1,  # Disable TTL during bulk load
            'gc_interval': 1800,  # 30 minutes GC interval
            'quota_enabled': False,  # Disable quotas for bulk loading
            'dml_rate_limit': -1,  # Unlimited DML operations
            'dql_rate_limit': 100  # Minimal query rate limit
        }
        
        # System-wide optimizations
        config['system_config'] = {
            'enable_dynamic_field': False,  # Disable for better performance
            'enable_ttl': False,  # Disable TTL during bulk load
            'auto_balance': False,  # Disable auto-balance during bulk load
            'auto_handoff': True,  # Enable auto handoff
            'log_level': 'warn',  # Reduce logging overhead
            'metric_type': 'FLAT',  # Use FLAT during insertion, build index later
            'graceful_time': 0,  # No graceful shutdown delay
            'session_ttl_seconds': 3600,  # 1 hour session timeout
            'heartbeat_interval': 10000,  # 10 second heartbeat
            'high_priority_thread_pool_ratio': 0.8,  # More threads for high priority tasks
            'middle_priority_thread_pool_ratio': 0.5,
            'low_priority_thread_pool_ratio': 0.2
        }
        
        # Calculate expected performance metrics
        vectors_per_gb = (1024 * 1024 * 1024) / (self.EMBEDDING_DIM * self.BYTES_PER_FLOAT)
        
        # Estimate throughput based on node types and hardware
        if self.gpu_count > 0:
            # GPU mode: higher throughput
            expected_throughput_mb_per_sec = config['data_nodes'] * 150  # 150MB/s per data node with GPU
        else:
            # CPU mode: standard throughput
            expected_throughput_mb_per_sec = config['data_nodes'] * 100  # 100MB/s per data node
        
        config['performance_estimate'] = {
            'segment_size_mb': segment_size_mb,
            'vectors_per_segment': int(segment_size_mb * vectors_per_gb),
            'expected_throughput_mb_per_sec': expected_throughput_mb_per_sec,
            'expected_vectors_per_sec': int(expected_throughput_mb_per_sec * vectors_per_gb),
            'time_to_ingest_1m_vectors_minutes': round(1_000_000 / (expected_throughput_mb_per_sec * vectors_per_gb) / 60, 1),
            'time_to_ingest_1b_vectors_hours': round(1_000_000_000 / (expected_throughput_mb_per_sec * vectors_per_gb) / 3600, 1),
            'max_dataset_size_gb': int(usable_ram_gb * 10),  # Can handle 10x RAM size with disk storage
            'optimal_batch_size': min(100000, int(segment_size_mb * 1024 * 1024 / (self.EMBEDDING_DIM * self.BYTES_PER_FLOAT) / 10))
        }
        
        # Summary for logging
        config['summary'] = {
            'mode': 'auto_bulk_optimized',
            'total_nodes': total_nodes,
            'node_distribution': {
                'data_nodes': config['data_nodes'],
                'index_nodes': config['index_nodes'],
                'query_nodes': config['query_nodes']
            },
            'memory_distribution': {
                'data_nodes_total_gb': round(config['memory_allocation']['data_node_memory_mb'] * config['data_nodes'] / 1024, 1),
                'index_nodes_total_gb': round(config['memory_allocation']['index_node_memory_mb'] * config['index_nodes'] / 1024, 1),
                'query_nodes_total_gb': round(config['memory_allocation']['query_node_memory_mb'] * config['query_nodes'] / 1024, 1),
                'per_node_breakdown': {
                    'data_node_mb': config['memory_allocation']['data_node_memory_mb'],
                    'index_node_mb': config['memory_allocation']['index_node_memory_mb'],
                    'query_node_mb': config['memory_allocation']['query_node_memory_mb']
                }
            },
            'segment_size_mb': segment_size_mb,
            'optimization_focus': 'bulk_ingestion_and_indexing',
            'hardware_acceleration': 'GPU' if self.gpu_count > 0 else 'CPU',
            'estimated_performance': f"{expected_throughput_mb_per_sec} MB/s throughput"
        }
        
        return config
    
    def calculate_target_mode(self, target_embeddings: int) -> Dict[str, Any]:
        """
        Target mode: Optimize for specific number of embeddings
        """
        config = {}
        
        # Calculate data size
        embedding_size_bytes = self.EMBEDDING_DIM * self.BYTES_PER_FLOAT
        total_data_gb = (target_embeddings * embedding_size_bytes) / (1024**3)
        required_ram_gb = total_data_gb * self.OVERHEAD_FACTOR
        
        # Check if we have enough resources
        if required_ram_gb > self.total_ram_gb * 0.9:
            print(f"Warning: Target embeddings ({target_embeddings:,}) require ~{required_ram_gb:.1f}GB RAM, "
                  f"but only {self.total_ram_gb * 0.9:.1f}GB is available. Using auto mode scaling.")
            return self.calculate_auto_mode()
        
        # Calculate optimal node counts based on data size
        if self.gpu_count > 0:
            # Scale index nodes based on data size and GPU memory
            embeddings_per_gpu = target_embeddings / self.gpu_count
            min_gpu_memory_gb = min(gpu['memory_total_mb'] for gpu in self.gpus) / 1024
            
            # Estimate index nodes needed per GPU
            index_nodes_per_gpu = max(1, min(8, int(embeddings_per_gpu / 5_000_000)))  # 1 node per 5M embeddings
            config['index_nodes'] = self.gpu_count * index_nodes_per_gpu
            
            config['query_nodes'] = max(self.gpu_count, min(self.gpu_count * 2, 8))
        else:
            # CPU mode scaling
            config['index_nodes'] = max(2, min(16, int(target_embeddings / 10_000_000)))  # 1 node per 10M
            config['query_nodes'] = max(2, min(8, int(target_embeddings / 20_000_000)))
        
        # Data nodes based on ingestion throughput needs
        # Assume we want to ingest in reasonable time (e.g., 1 hour)
        target_throughput_mb_per_sec = (total_data_gb * 1024) / 3600  # GB to MB, 1 hour target
        config['data_nodes'] = max(4, min(32, int(target_throughput_mb_per_sec / 100)))  # 100MB/s per node estimate
        
        # Memory allocation
        total_nodes = config['index_nodes'] + config['data_nodes'] + config['query_nodes']
        config['memory_per_node_mb'] = int((required_ram_gb * 1024) / total_nodes)
        
        # Segment size optimization based on dataset
        if target_embeddings < 10_000_000:
            segment_size_mb = 512  # Smaller segments for smaller datasets
        elif target_embeddings < 50_000_000:
            segment_size_mb = 1024  # 1GB segments
        else:
            segment_size_mb = 2048  # 2GB segments for large datasets
        
        config['segment'] = {
            'max_size_mb': segment_size_mb,
            'seal_proportion': 0.85 if target_embeddings < 50_000_000 else 0.9,
            'insert_buffer_size': min(32 * 1024 * 1024, int(required_ram_gb * 1024 * 1024 * 1024 / 200))
        }
        
        # Parallelism based on data size
        config['parallelism'] = {
            'data_sync_parallel': min(config['data_nodes'] * 4, 256),
            'build_index_parallel': config['index_nodes'],
            'flush_parallel': min(config['data_nodes'], 32),
            'compaction_parallel': min(self.cpu_cores // 2, 16)
        }
        
        # GPU settings if available
        if self.gpu_count > 0:
            min_gpu_memory = min(gpu['memory_total_mb'] for gpu in self.gpus)
            # Scale GPU memory based on dataset size
            gpu_mem_factor = min(0.8, 0.3 + (target_embeddings / 100_000_000) * 0.5)  # 30-80%
            config['gpu'] = {
                'init_mem_size_mb': int(min_gpu_memory * gpu_mem_factor * 0.6),
                'max_mem_size_mb': int(min_gpu_memory * gpu_mem_factor)
            }
        
        # Thread pools - scale with dataset
        thread_factor = min(1.0, target_embeddings / 50_000_000)  # Scale up to 50M
        config['thread_pools'] = {
            'knowhere_thread_pool_ratio': max(4, int(16 * thread_factor)),
            'disk_thread_pool_ratio': max(2, int(8 * thread_factor)),
            'build_parallel_rate': max(0.5, thread_factor)
        }
        
        return config
    
    def get_resource_summary(self, config: Dict[str, Any], target_embeddings: Optional[int] = None) -> Dict[str, Any]:
        """Generate a summary of resource allocation"""
        total_nodes = config['index_nodes'] + config['data_nodes'] + config['query_nodes']
        total_ram_allocated_gb = (total_nodes * config['memory_per_node_mb']) / 1024
        
        # Check if we have the new summary structure from calculate_auto_mode
        if 'summary' in config:
            # Use the enhanced summary from the new auto mode
            summary = config['summary'].copy()
            # Add compatibility fields
            summary['memory'] = {
                'total_allocated_gb': round(total_ram_allocated_gb, 2),
                'per_node_mb': config['memory_per_node_mb'],
                'system_total_gb': self.total_ram_gb,
                'utilization_percent': round((total_ram_allocated_gb / self.total_ram_gb) * 100, 1)
            }
            summary['parallelism'] = config['parallelism']
            summary['segment_config'] = config['segment']
            
            # Add performance estimates if available
            if 'performance_estimate' in config:
                summary['performance_estimate'] = config['performance_estimate']
            
            # Add resource limits if available
            if 'resource_limits' in config:
                summary['resource_limits'] = config['resource_limits']
                
            # Add IO config if available
            if 'io_config' in config:
                summary['io_config'] = config['io_config']
        else:
            # Legacy summary format for backward compatibility
            summary = {
                'mode': 'auto' if target_embeddings is None else f'target ({target_embeddings:,} embeddings)',
                'total_nodes': total_nodes,
                'node_distribution': {
                    'index_nodes': config['index_nodes'],
                    'data_nodes': config['data_nodes'],
                    'query_nodes': config['query_nodes']
                },
                'memory': {
                    'total_allocated_gb': round(total_ram_allocated_gb, 2),
                    'per_node_mb': config['memory_per_node_mb'],
                    'system_total_gb': self.total_ram_gb,
                    'utilization_percent': round((total_ram_allocated_gb / self.total_ram_gb) * 100, 1)
                },
                'parallelism': config['parallelism'],
                'segment_config': config['segment']
            }
        
        if 'gpu' in config:
            summary['gpu'] = config['gpu']
            summary['gpu']['count'] = self.gpu_count
        
        return summary
    
    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate that the configuration is feasible"""
        total_nodes = config['index_nodes'] + config['data_nodes'] + config['query_nodes']
        total_ram_needed_gb = (total_nodes * config['memory_per_node_mb']) / 1024
        
        if total_ram_needed_gb > self.total_ram_gb * 0.95:
            print(f"Warning: Configuration requires {total_ram_needed_gb:.1f}GB RAM, "
                  f"but only {self.total_ram_gb:.1f}GB is available.")
            return False
        
        if config['index_nodes'] > self.gpu_count * 16 and self.gpu_count > 0:
            print(f"Warning: Too many index nodes ({config['index_nodes']}) for {self.gpu_count} GPUs.")
            return False
        
        return True


if __name__ == "__main__":
    # Test with mock resources
    mock_resources = {
        'cpu': {'physical_cores': 32, 'logical_cores': 64},
        'memory': {'total_gb': 128, 'available_gb': 120},
        'gpus': [
            {'index': 0, 'memory_total_mb': 16384},
            {'index': 1, 'memory_total_mb': 16384}
        ]
    }
    
    optimizer = BulkLoadOptimizer(mock_resources)
    
    # Test auto mode
    print("=== Auto Mode Configuration ===")
    auto_config = optimizer.calculate_auto_mode()
    summary = optimizer.get_resource_summary(auto_config)
    print(f"Configuration: {summary}")
    
    # Test target mode
    print("\n=== Target Mode Configuration (50M embeddings) ===")
    target_config = optimizer.calculate_target_mode(50_000_000)
    summary = optimizer.get_resource_summary(target_config, 50_000_000)
    print(f"Configuration: {summary}")