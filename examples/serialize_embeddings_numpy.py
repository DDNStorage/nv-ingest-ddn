#!/usr/bin/env python3
"""
Example: Serialize embeddings and store as numpy files
This script shows how to generate embeddings and save them as numpy arrays
for efficient storage and later processing.
"""

import numpy as np
import time
import json
import os
from pathlib import Path
from nv_ingest_client.client import Ingestor
from nv_ingest_client.primitives import JobSpec

def get_file_size_mb(filepath):
    size_bytes = os.path.getsize(filepath)
    return round(size_bytes / (1024 * 1024), 6)  # Size in MB


def generate_and_serialize_embeddings(documents, config, output_dir):
    """
    Generate embeddings and serialize them as numpy files.
    
    Args:
        documents: List of document paths
        config: NV-Ingest configuration
        output_dir: Directory to save numpy files
        
    Returns:
        Metadata about saved embeddings
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create ingestor instance
    ingestor = Ingestor(
        host=config["host"],
        port=config["port"]
    )
    
    # Generate embeddings
    results = (
        ingestor
        .files(documents)
        .extract()
        .embed()
        .ingest()
    )
    
    # Process and save embeddings
    embeddings_metadata = []
    embedding_count = 0
    total_file_size = 0
    
    for result1 in results:
        for count, result in enumerate(result1):

            if isinstance(result, dict) and "metadata" in result:
                metadata = result["metadata"]
                
                if "embedding" in metadata:
                    # Convert to numpy array
                    embedding_array = np.array(metadata["embedding"], dtype=np.float32)
                    
                    # Generate unique filename
                    source_name = metadata.get("source_metadata", {}).get("source_name", "unknown")
                    chunk_index = metadata.get("chunk_index", 0)
                    
                    # Clean filename
                    base_name = Path(source_name).stem.replace(" ", "_")
                    # filename = f"{base_name}_chunk_{chunk_index:04d}.npy"
                    filename = f"{base_name}_{count}.npy"
                    filepath = output_path / filename
                    
                    # Save numpy array
                    np.save(filepath, embedding_array)
                    file_size_mb = get_file_size_mb(filepath)
                    total_file_size = total_file_size + file_size_mb
                    print(f"Saved {filename} ({file_size_mb} MB)")
                    
                    if hasattr(embedding_array, 'shape') and len(embedding_array.shape) > 0:
                        embedding_dim = embedding_array.shape[0]
                    else:
                        embedding_dim = embedding_array.shape

                    # Save metadata
                    meta_entry = {
                        "filename": filename,
                        "filepath": str(filepath),
                        "embedding_dim": embedding_dim,
                        "content": metadata.get("content", ""),
                        "source_file": source_name,
                        "page_number": metadata.get("source_metadata", {}).get("page_number"),
                        "chunk_index": chunk_index,
                        "chunk_count": metadata.get("chunk_count", 1),
                        "collection": metadata.get("source_metadata", {}).get("collection", "default")
                    }
                    
                    embeddings_metadata.append(meta_entry)
                    embedding_count += 1


    # Save metadata as JSON
    metadata_file = output_path / "embeddings_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(embeddings_metadata, f, indent=2)
    
    print(f"Saved {embedding_count} embeddings to {output_dir}")
    print(f"Metadata saved to {metadata_file}")
    print(f"Total size of embeddings: {round(total_file_size, 6)} MB")
    
    return embeddings_metadata


def load_embeddings_from_disk(output_dir, indices=None):
    """
    Load embeddings from numpy files.
    
    Args:
        output_dir: Directory containing numpy files
        indices: Optional list of indices to load (None = load all)
        
    Returns:
        Tuple of (embeddings_array, metadata_list)
    """
    output_path = Path(output_dir)
    
    # Load metadata
    metadata_file = output_path / "embeddings_metadata.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Filter by indices if provided
    if indices is not None:
        metadata = [metadata[i] for i in indices if i < len(metadata)]
    
    # Load embeddings
    embeddings = []
    total_loaded_size_mb = 0.0
    total_file_loaded = 0
    for meta in metadata:
        filepath = meta["filepath"]
        embedding = np.load(filepath)
        embeddings.append(embedding)
        # Accumulate file size
        total_loaded_size_mb += os.path.getsize(filepath) / (1024 * 1024)
        total_file_loaded += 1

    # Stack into single array if multiple embeddings
    if embeddings:

        # Filter only valid (2048,) embeddings
        valid_embeddings = [
            emb for emb in embeddings
            if isinstance(emb, np.ndarray) and emb.ndim == 1 and emb.shape[0] == 2048
        ]

        if not valid_embeddings:
            raise ValueError("No valid embeddings found to load.")

        embeddings_array = np.vstack(valid_embeddings)


        # embeddings_array = np.vstack(embeddings)
    else:
        embeddings_array = np.array([])
    print(f"Total loaded .npy size: {round(total_loaded_size_mb, 6)} MB")
    print(f"Total No of files loaded: {total_file_loaded}")
    
    return embeddings_array, metadata


def batch_serialize_embeddings(documents, config, output_dir, batch_size=1000):
    """
    Serialize embeddings in batches for very large datasets.
    
    Args:
        documents: List of document paths
        config: NV-Ingest configuration
        output_dir: Directory to save numpy files
        batch_size: Number of embeddings per batch file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create ingestor instance
    ingestor = Ingestor(
        host=config["host"],
        port=config["port"]
    )
    
    # Generate embeddings
    results = (
        ingestor
        .files(documents)
        .extract()
        .embed()
        .ingest()
    )
    
    # Collect embeddings and metadata
    all_embeddings = []
    all_metadata = []
    for result1 in results:
        for result in result1:
            if isinstance(result, dict) and "metadata" in result:
                metadata = result["metadata"]
                
                if "embedding" in metadata:
                    embedding = np.array(metadata["embedding"], dtype=np.float32)
                    all_embeddings.append(embedding)
                    
                    meta_entry = {
                        "content": metadata.get("content", ""),
                        "source_file": metadata.get("source_metadata", {}).get("source_name", ""),
                        "page_number": metadata.get("source_metadata", {}).get("page_number"),
                        "chunk_index": metadata.get("chunk_index", 0)
                    }
                    all_metadata.append(meta_entry)
    
    # Save in batches
    num_embeddings = len(all_embeddings)
    num_batches = (num_embeddings + batch_size - 1) // batch_size
    
    batch_info = []
    
    for batch_idx in range(num_batches):
        print("No of Batches:", num_batches)
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_embeddings)
        
        # Get batch data
        batch_embeddings = all_embeddings[start_idx:end_idx]
        batch_metadata = all_metadata[start_idx:end_idx]
        
        # Stack embeddings
        try:
            # Filter out malformed embeddings
            filtered = [
                emb for emb in batch_embeddings
                if isinstance(emb, np.ndarray) and emb.ndim == 1 and emb.shape[0] == 2048
            ]
            if not filtered:
                print(f"⚠️ Skipping batch {batch_idx}: no valid embeddings.")
                continue
            batch_array = np.vstack(filtered)
        except Exception as e:
            print(f"❌ Error stacking embeddings in batch {batch_idx}: {e}")
            continue
        
        # Save batch
        batch_file = output_path / f"embeddings_batch_{int(time.time()) + batch_idx}.npy"
        np.save(batch_file, batch_array)
        
        # Save batch metadata
        batch_meta_file = output_path / f"metadata_batch_{int(time.time()) + batch_idx}.json"
        with open(batch_meta_file, 'w') as f:
            json.dump(batch_metadata, f, indent=2)
        
        batch_info.append({
            "batch_index": batch_idx,
            "embeddings_file": str(batch_file),
            "metadata_file": str(batch_meta_file),
            "num_embeddings": len(batch_embeddings),
            "shape": batch_array.shape
        })
    
    # Save batch info
    batch_info_file = output_path / "batch_info.json"
    with open(batch_info_file, 'w') as f:
        json.dump(batch_info, f, indent=2)
    
    print(f"Saved {num_embeddings} embeddings in {num_batches} batches")
    print(f"Batch info saved to {batch_info_file}")
    
    return batch_info


if __name__ == "__main__":
    # Configuration
    config = {
        "host": "localhost",
        "port": 8082
    }
    
    # Example documents
    documents = [
        "/home/artemivashchenko/src/infinia-ai-workload-poc/multimodal_test.pdf",
        "/home/artemivashchenko/src/infinia-ai-workload-poc/nv-ingest-ddn/data/functional_validation.pdf"
    ]
    
    # Example 1: Individual numpy files per embedding
    print("=== Example 1: Individual files ===")
    output_dir = "./embeddings_individual"
    metadata = generate_and_serialize_embeddings(documents, config, output_dir)
    
    # Load embeddings back
    embeddings, loaded_metadata = load_embeddings_from_disk(output_dir)
    print(f"Loaded embeddings shape: {embeddings.shape}")
    
    # Example 2: Batch serialization for large datasets
    print("\n=== Example 2: Batch serialization ===")
    batch_output_dir = "./embeddings_batched"
    batch_info = batch_serialize_embeddings(
        documents, 
        config, 
        batch_output_dir, 
        batch_size=1000
    )
    
    # Example 3: Convert to other formats
    print("\n=== Example 3: Export formats ===")
    
    # Save as HDF5 (requires h5py)
    try:
        import h5py
        
        h5_file = Path(output_dir) / "embeddings.h5"
        with h5py.File(h5_file, 'w') as f:
            f.create_dataset('embeddings', data=embeddings)
            f.create_dataset('metadata', data=json.dumps(loaded_metadata))
        print(f"Saved as HDF5: {h5_file}")
        print(f"HDF5 file size: {get_file_size_mb(h5_file)} MB")
    except ImportError:
        print("h5py not installed, skipping HDF5 export")
    
    # Save as compressed numpy
    npz_file = Path(output_dir) / "embeddings.npz"
    np.savez_compressed(
        npz_file,
        embeddings=embeddings,
        metadata=json.dumps(loaded_metadata)
    )
    print(f"Saved as compressed NPZ: {npz_file}")
    print(f"NPZ file size: {get_file_size_mb(npz_file)} MB")