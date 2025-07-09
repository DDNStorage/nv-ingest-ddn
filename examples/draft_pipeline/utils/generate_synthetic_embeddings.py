#!/usr/bin/env python3
"""
Generate synthetic embeddings for testing and experiments.
Creates random embeddings with proper metadata structure.
"""

import argparse
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
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


def generate_embedding_batch(
    start_idx: int,
    end_idx: int,
    embedding_dim: int,
    output_dir: Path,
    experiment_folder: str,
    seed: int = None,
    worker_id: int = 0
) -> Tuple[List[Dict], int, int]:
    """Generate a batch of embeddings in a separate process.
    
    Returns:
        Tuple of (metadata_entries, files_created, total_bytes)
    """
    if seed is not None:
        np.random.seed(seed + worker_id)
    
    batch_size = end_idx - start_idx
    metadata_entries = []
    files_created = 0
    total_bytes = 0
    
    # Generate one embedding per file (indexer-compatible format)
    for i in range(batch_size):
        vector_idx = start_idx + i
        
        # Generate single normalized embedding
        embedding = np.random.randn(embedding_dim).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        
        # Create filename for this embedding
        doc_id = f"synthetic_{vector_idx // 10:06d}"
        chunk_id = vector_idx % 10
        filename = f"{doc_id}_chunk_{chunk_id:04d}.npy"
        filepath = output_dir / filename
        
        # Save single embedding (1D array)
        np.save(filepath, embedding, allow_pickle=False)
        
        # Create metadata entry with indexer-compatible format
        metadata_entry = {
            "filename": filename,
            "filepath": f"embeddings/{experiment_folder}/{filename}",
            "embedding_dim": embedding_dim,
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
        
        files_created += 1
        total_bytes += filepath.stat().st_size
    
    return metadata_entries, files_created, total_bytes


def save_metadata_chunk(metadata_entries: List[Dict], output_dir: Path, chunk_id: int):
    """Save a chunk of metadata to avoid memory issues."""
    metadata_path = output_dir / f"metadata_chunk_{chunk_id:04d}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata_entries, f, indent=2)


def merge_metadata_chunks(output_dir: Path):
    """Merge all metadata chunks into a single file."""
    all_metadata = []
    chunk_files = sorted(output_dir.glob("metadata_chunk_*.json"))
    
    for chunk_file in chunk_files:
        with open(chunk_file, 'r') as f:
            all_metadata.extend(json.load(f))
        chunk_file.unlink()
    
    with open(output_dir / "embeddings_metadata.json", 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    return len(all_metadata)


def generate_synthetic_embeddings(
    num_vectors: int,
    output_dir: Path,
    embedding_dim: int = 2048,
    batch_size: int = 100000,
    seed: int = None,
    num_workers: int = None
) -> Dict:
    """Generate synthetic embeddings using parallel processing.
    
    Args:
        num_vectors: Total number of vectors to generate
        output_dir: Directory to save embeddings
        embedding_dim: Dimension of each embedding vector
        batch_size: Number of vectors per worker batch
        vectors_per_file: Number of vectors per numpy file
        seed: Random seed for reproducibility
        num_workers: Number of parallel workers (default: CPU count)
    
    Returns:
        Dictionary with generation metrics
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize metrics
    metrics = {
        "start_time": datetime.now().isoformat(),
        "num_vectors": num_vectors,
        "embedding_dim": embedding_dim,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "seed": seed,
        "files_created": 0,
        "total_size_bytes": 0
    }
    
    # Calculate batches for parallel processing
    batches = []
    for i in range(0, num_vectors, batch_size):
        start_idx = i
        end_idx = min(i + batch_size, num_vectors)
        batches.append((start_idx, end_idx))
    
    logger.info(f"Generating {num_vectors:,} vectors using {num_workers} workers")
    logger.info(f"Total batches: {len(batches)}")
    
    # Process batches in parallel
    all_metadata = []
    metadata_chunk_id = 0
    metadata_chunk_size = 100000
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_batch = {}
        for worker_id, (start_idx, end_idx) in enumerate(batches):
            future = executor.submit(
                generate_embedding_batch,
                start_idx, end_idx,
                embedding_dim, output_dir,
                output_dir.name,
                seed, worker_id
            )
            future_to_batch[future] = (start_idx, end_idx)
        
        # Process completed tasks
        with tqdm(total=num_vectors, desc="Generating embeddings") as pbar:
            for future in as_completed(future_to_batch):
                start_idx, end_idx = future_to_batch[future]
                try:
                    metadata_entries, files_created, total_bytes = future.result()
                    
                    # Update metrics
                    metrics["files_created"] += files_created
                    metrics["total_size_bytes"] += total_bytes
                    
                    # Add metadata to buffer
                    all_metadata.extend(metadata_entries)
                    
                    # Save metadata in chunks
                    if len(all_metadata) >= metadata_chunk_size:
                        save_metadata_chunk(all_metadata[:metadata_chunk_size], 
                                          output_dir, metadata_chunk_id)
                        all_metadata = all_metadata[metadata_chunk_size:]
                        metadata_chunk_id += 1
                    
                    pbar.update(end_idx - start_idx)
                    
                except Exception as e:
                    logger.error(f"Error processing batch {start_idx}-{end_idx}: {e}")
                    raise
    
    # Save remaining metadata
    if all_metadata:
        save_metadata_chunk(all_metadata, output_dir, metadata_chunk_id)
    
    # Merge all metadata chunks
    logger.info("Merging metadata chunks...")
    total_metadata_entries = merge_metadata_chunks(output_dir)
    logger.info(f"Total metadata entries: {total_metadata_entries:,}")
    
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


def estimate_requirements(num_vectors: int, embedding_dim: int) -> Dict:
    """Estimate memory and storage requirements."""
    bytes_per_vector = embedding_dim * 4  # float32
    bytes_per_file = bytes_per_vector + 128  # numpy overhead per file
    total_bytes = num_vectors * bytes_per_file
    
    # Metadata estimate
    metadata_per_entry = 500  # bytes (JSON)
    metadata_total = num_vectors * metadata_per_entry
    
    return {
        "embeddings_size": format_size(total_bytes),
        "metadata_size": format_size(metadata_total),
        "total_size": format_size(total_bytes + metadata_total),
        "num_files": num_vectors,
        "memory_per_worker_gb": (embedding_dim * 100000 * 4) / (1024**3)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Optimized synthetic embedding generator for large-scale datasets"
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
        help="Base output directory (default: /home/artemivashchenko/src/infinia-ai-workload-poc/nv-ingest-ddn/examples/draft_pipeline/embeddings)"
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=2048,
        help="Embedding dimension (default: 2048)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100000,
        help="Batch size per worker (default: 100000)"
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
        help="Number of parallel workers (default: CPU count)"
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
    
    # Create output directory
    base_output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.experiment_name:
        folder_name = f"{args.experiment_name}_{timestamp}"
    else:
        folder_name = f"synthetic_{format_number(num_vectors)}_{timestamp}"
    
    output_dir = base_output_dir / folder_name
    
    # Estimate requirements
    estimates = estimate_requirements(num_vectors, args.embedding_dim)
    logger.info(f"\nEstimated requirements for {num_vectors:,} vectors:")
    logger.info(f"  Embeddings size: {estimates['embeddings_size']}")
    logger.info(f"  Metadata size: {estimates['metadata_size']}")
    logger.info(f"  Total size: {estimates['total_size']}")
    logger.info(f"  Number of files: {estimates['num_files']:,}")
    logger.info(f"  Memory per worker: ~{estimates['memory_per_worker_gb']:.1f} GB")
    
    if args.estimate_only:
        return
    
    logger.info(f"\nGenerating {num_vectors:,} synthetic embeddings")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Embedding dimension: {args.embedding_dim}")
    logger.info(f"Batch size: {args.batch_size:,}")
    logger.info(f"Workers: {args.num_workers or mp.cpu_count()}")
    
    # Generate embeddings
    metrics = generate_synthetic_embeddings(
        num_vectors=num_vectors,
        output_dir=output_dir,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers
    )
    
    # Print summary
    logger.info("\nGeneration complete!")
    logger.info(f"Files created: {metrics['files_created']:,}")
    logger.info(f"Total size: {format_size(metrics['total_size_bytes'])}")
    logger.info(f"Duration: {metrics['duration_seconds']:.2f} seconds")
    logger.info(f"Vectors/second: {num_vectors / metrics['duration_seconds']:,.0f}")
    logger.info(f"MB/second: {metrics['total_size_bytes'] / metrics['duration_seconds'] / 1024 / 1024:.1f}")
    logger.info(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    main()