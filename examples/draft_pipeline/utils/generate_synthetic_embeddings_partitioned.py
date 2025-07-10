#!/usr/bin/env python3
"""
Generate synthetic embeddings with partition support for efficient loading.
Creates pre-aggregated embeddings in partitions for better scalability.
"""

import argparse
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import multiprocessing as mp

import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def format_number(num: int) -> str:
    """Format large numbers with commas"""
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return str(num)


def generate_partition_batch(
    partition_id: int,
    start_idx: int,
    end_idx: int,
    embedding_dim: int,
    output_dir: Path,
    experiment_folder: str,
    seed: Optional[int] = None
) -> Tuple[Dict, int]:
    """Generate a complete partition with pre-aggregated embeddings.
    
    Returns:
        Tuple of (partition_info, total_bytes)
    """
    if seed is not None:
        np.random.seed(seed + partition_id)
    
    partition_size = end_idx - start_idx
    partition_dir = output_dir / f"partition_{partition_id:04d}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all embeddings for this partition at once
    embeddings = np.random.randn(partition_size, embedding_dim).astype(np.float32)
    
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    # Generate IDs for this partition
    ids = np.arange(start_idx, end_idx, dtype=np.int64)
    
    # Generate metadata for all embeddings in partition
    metadata_entries = []
    for i in range(partition_size):
        vector_idx = start_idx + i
        doc_id = f"synthetic_{vector_idx // 10:06d}"
        chunk_id = vector_idx % 10
        
        metadata_entry = {
            "id": int(ids[i]),
            "vector_index": i,  # Index within the partition's embedding array
            "content": f"Synthetic content for document {doc_id}, chunk {chunk_id}",
            "source_file": f"data/synthetic/{doc_id}.pdf",
            "source_name": f"{doc_id}.pdf",
            "page_number": chunk_id + 1,
            "chunk_index": chunk_id,
            "document_id": doc_id,
            "collection": "synthetic",
            "is_synthetic": True
        }
        metadata_entries.append(metadata_entry)
    
    # Save partition data
    embeddings_path = partition_dir / "embeddings.npy"
    ids_path = partition_dir / "ids.npy"
    metadata_path = partition_dir / "metadata.json"
    
    np.save(embeddings_path, embeddings, allow_pickle=False)
    np.save(ids_path, ids, allow_pickle=False)
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata_entries, f, indent=2)
    
    # Calculate partition info
    total_bytes = embeddings_path.stat().st_size + ids_path.stat().st_size
    
    partition_info = {
        "partition_id": partition_id,
        "partition_dir": f"partition_{partition_id:04d}",
        "num_embeddings": partition_size,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "embedding_dim": embedding_dim,
        "total_size_bytes": total_bytes,
        "files": {
            "embeddings": "embeddings.npy",
            "ids": "ids.npy",
            "metadata": "metadata.json"
        }
    }
    
    return partition_info, total_bytes


def generate_synthetic_embeddings_partitioned(
    num_vectors: int,
    output_dir: Path,
    embedding_dim: int = 2048,
    num_partitions: int = 10,
    seed: Optional[int] = None,
    num_workers: Optional[int] = None
) -> Dict:
    """Generate synthetic embeddings with partition support.
    
    Args:
        num_vectors: Total number of vectors to generate
        output_dir: Directory to save embeddings
        embedding_dim: Dimension of each embedding vector
        num_partitions: Number of partitions to create
        seed: Random seed for reproducibility
        num_workers: Number of parallel workers (default: CPU count)
    
    Returns:
        Dictionary with generation metrics
    """
    if num_workers is None:
        num_workers = min(mp.cpu_count(), num_partitions)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate vectors per partition
    vectors_per_partition = num_vectors // num_partitions
    remaining_vectors = num_vectors % num_partitions
    
    # Initialize metrics
    metrics = {
        "start_time": datetime.now().isoformat(),
        "num_vectors": num_vectors,
        "embedding_dim": embedding_dim,
        "num_partitions": num_partitions,
        "vectors_per_partition": vectors_per_partition,
        "num_workers": num_workers,
        "seed": seed,
        "total_size_bytes": 0,
        "format_version": "2.0",  # New partitioned format
        "storage_format": "partitioned_aggregated"
    }
    
    # Create partition tasks
    partition_tasks = []
    current_idx = 0
    
    for partition_id in range(num_partitions):
        # Distribute remaining vectors across first partitions
        partition_size = vectors_per_partition
        if partition_id < remaining_vectors:
            partition_size += 1
        
        start_idx = current_idx
        end_idx = current_idx + partition_size
        partition_tasks.append((partition_id, start_idx, end_idx))
        current_idx = end_idx
    
    logger.info(f"Generating {num_vectors:,} vectors in {num_partitions} partitions")
    logger.info(f"Vectors per partition: ~{vectors_per_partition:,}")
    logger.info(f"Using {num_workers} workers")
    
    # Process partitions in parallel
    all_partition_info = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_partition = {}
        for partition_id, start_idx, end_idx in partition_tasks:
            future = executor.submit(
                generate_partition_batch,
                partition_id, start_idx, end_idx,
                embedding_dim, output_dir,
                output_dir.name, seed
            )
            future_to_partition[future] = partition_id
        
        # Process completed tasks
        with tqdm(total=num_partitions, desc="Generating partitions") as pbar:
            for future in as_completed(future_to_partition):
                partition_id = future_to_partition[future]
                try:
                    partition_info, total_bytes = future.result()
                    
                    all_partition_info.append(partition_info)
                    metrics["total_size_bytes"] += total_bytes
                    
                    pbar.update(1)
                    
                except Exception as e:
                    logger.error(f"Error generating partition {partition_id}: {e}")
                    raise
    
    # Sort partition info by partition_id
    all_partition_info.sort(key=lambda x: x['partition_id'])
    
    # Save master metadata
    master_metadata = {
        "format_version": "2.0",
        "storage_format": "partitioned_aggregated",
        "num_vectors": num_vectors,
        "embedding_dim": embedding_dim,
        "num_partitions": num_partitions,
        "partitions": all_partition_info,
        "generation_timestamp": datetime.now().isoformat()
    }
    
    metadata_path = output_dir / "embeddings_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(master_metadata, f, indent=2)
    
    # Update metrics
    metrics["end_time"] = datetime.now().isoformat()
    metrics["duration_seconds"] = (
        datetime.fromisoformat(metrics["end_time"]) - 
        datetime.fromisoformat(metrics["start_time"])
    ).total_seconds()
    
    # Save generation metrics
    metrics_path = output_dir / "generation_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def estimate_requirements(num_vectors: int, embedding_dim: int, num_partitions: int) -> Dict:
    """Estimate memory and storage requirements."""
    bytes_per_vector = embedding_dim * 4  # float32
    total_embedding_bytes = num_vectors * bytes_per_vector
    
    # ID storage
    bytes_per_id = 8  # int64
    total_id_bytes = num_vectors * bytes_per_id
    
    # Metadata estimate (more compact in partitioned format)
    metadata_per_entry = 300  # bytes (JSON)
    metadata_total = num_vectors * metadata_per_entry
    
    # Partition overhead
    partition_overhead = num_partitions * 1024  # Directory structure
    
    return {
        "embeddings_size": format_size(total_embedding_bytes),
        "ids_size": format_size(total_id_bytes),
        "metadata_size": format_size(metadata_total),
        "total_size": format_size(total_embedding_bytes + total_id_bytes + metadata_total + partition_overhead),
        "num_partitions": num_partitions,
        "avg_partition_size": format_size((total_embedding_bytes + total_id_bytes) / num_partitions),
        "memory_per_worker_gb": ((num_vectors / num_partitions) * bytes_per_vector) / (1024**3)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic embeddings with partition support for efficient loading"
    )
    parser.add_argument(
        "num_vectors",
        type=str,
        help="Number of vectors to generate (e.g., 1000, 1M, 5M)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/artemivashchenko/src/infinia-ai-workload-poc/nv-ingest-ddn/examples/draft_pipeline/embeddings",
        help="Base output directory"
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=2048,
        help="Embedding dimension (default: 2048)"
    )
    parser.add_argument(
        "--num-partitions",
        type=int,
        default=10,
        help="Number of partitions to create (default: 10)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: min(CPU count, num_partitions))"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Custom experiment name for output folder"
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Only estimate requirements without generating"
    )
    
    args = parser.parse_args()
    
    # Parse number of vectors
    num_str = args.num_vectors.upper()
    if num_str.endswith('M'):
        num_vectors = int(float(num_str[:-1]) * 1_000_000)
    elif num_str.endswith('K'):
        num_vectors = int(float(num_str[:-1]) * 1_000)
    else:
        num_vectors = int(num_str)
    
    # Validate partition count
    if args.num_partitions > num_vectors:
        logger.warning(f"Number of partitions ({args.num_partitions}) exceeds number of vectors ({num_vectors})")
        args.num_partitions = min(args.num_partitions, num_vectors)
        logger.info(f"Adjusted to {args.num_partitions} partitions")
    
    # Create output directory
    base_output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.experiment_name:
        folder_name = f"{args.experiment_name}_{timestamp}"
    else:
        folder_name = f"synthetic_{format_number(num_vectors)}_p{args.num_partitions}_{timestamp}"
    
    output_dir = base_output_dir / folder_name
    
    # Estimate requirements
    estimates = estimate_requirements(num_vectors, args.embedding_dim, args.num_partitions)
    logger.info(f"\nEstimated requirements for {num_vectors:,} vectors in {args.num_partitions} partitions:")
    logger.info(f"  Embeddings size: {estimates['embeddings_size']}")
    logger.info(f"  IDs size: {estimates['ids_size']}")
    logger.info(f"  Metadata size: {estimates['metadata_size']}")
    logger.info(f"  Total size: {estimates['total_size']}")
    logger.info(f"  Average partition size: {estimates['avg_partition_size']}")
    logger.info(f"  Memory per worker: ~{estimates['memory_per_worker_gb']:.1f} GB")
    
    if args.estimate_only:
        return
    
    logger.info(f"\nGenerating {num_vectors:,} synthetic embeddings")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Embedding dimension: {args.embedding_dim}")
    logger.info(f"Number of partitions: {args.num_partitions}")
    logger.info(f"Workers: {args.num_workers or min(mp.cpu_count(), args.num_partitions)}")
    
    # Generate embeddings
    metrics = generate_synthetic_embeddings_partitioned(
        num_vectors=num_vectors,
        output_dir=output_dir,
        embedding_dim=args.embedding_dim,
        num_partitions=args.num_partitions,
        seed=args.seed,
        num_workers=args.num_workers
    )
    
    # Print summary
    logger.info("\nGeneration complete!")
    logger.info(f"Total size: {format_size(metrics['total_size_bytes'])}")
    logger.info(f"Duration: {metrics['duration_seconds']:.2f} seconds")
    logger.info(f"Vectors/second: {num_vectors / metrics['duration_seconds']:,.0f}")
    logger.info(f"MB/second: {metrics['total_size_bytes'] / metrics['duration_seconds'] / 1024 / 1024:.1f}")
    logger.info(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    main()