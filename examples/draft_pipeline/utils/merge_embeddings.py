#!/usr/bin/env python3
"""
Merge embeddings from multiple folders into a single experiment folder.
Supports deduplication and flexible folder selection.
"""

import argparse
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

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


def load_metadata(folder_path: Path) -> List[Dict]:
    """Load metadata from a folder"""
    metadata_path = folder_path / "embeddings_metadata.json"
    if not metadata_path.exists():
        logger.warning(f"No metadata found in {folder_path}")
        return []
    
    with open(metadata_path, 'r') as f:
        return json.load(f)


def get_embedding_folders(base_dir: Path, specific_folders: List[str] = None) -> List[Path]:
    """Get list of embedding folders to merge.
    
    Args:
        base_dir: Base embeddings directory
        specific_folders: List of specific folder names to merge (None for all)
    
    Returns:
        List of folder paths to merge
    """
    if specific_folders:
        # Use specific folders
        folders = []
        for folder_name in specific_folders:
            folder_path = base_dir / folder_name
            if folder_path.exists() and folder_path.is_dir():
                folders.append(folder_path)
            else:
                logger.warning(f"Folder not found: {folder_path}")
        return folders
    else:
        # Get all folders with embeddings_metadata.json
        folders = []
        for folder_path in base_dir.iterdir():
            if folder_path.is_dir() and (folder_path / "embeddings_metadata.json").exists():
                folders.append(folder_path)
        return sorted(folders)


def compute_content_hash(metadata: Dict) -> str:
    """Compute a hash for deduplication based on content"""
    # Use source file and chunk index for deduplication
    source = metadata.get('source_file', metadata.get('source_name', ''))
    chunk_idx = metadata.get('chunk_index', 0)
    # Include content for synthetic data
    if metadata.get('is_synthetic', False):
        content = metadata.get('content', '')
        return f"{source}:{chunk_idx}:{hash(content)}"
    return f"{source}:{chunk_idx}"


def merge_embeddings(
    folders: List[Path],
    output_dir: Path,
    deduplicate: bool = True,
    copy_files: bool = False
) -> Dict:
    """Merge embeddings from multiple folders.
    
    Args:
        folders: List of folders to merge
        output_dir: Output directory for merged data
        deduplicate: Whether to deduplicate based on content
        copy_files: Whether to copy files (True) or create symlinks (False)
    
    Returns:
        Dictionary with merge metrics
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize metrics
    metrics = {
        "start_time": datetime.now().isoformat(),
        "input_folders": [str(f) for f in folders],
        "num_input_folders": len(folders),
        "total_input_files": 0,
        "total_output_files": 0,
        "duplicates_found": 0,
        "total_size_bytes": 0,
        "deduplicate": deduplicate,
        "copy_files": copy_files
    }
    
    # Merged metadata
    merged_metadata = []
    seen_hashes = set() if deduplicate else None
    
    # Process each folder
    for folder_idx, folder in enumerate(folders):
        logger.info(f"Processing folder {folder_idx + 1}/{len(folders)}: {folder.name}")
        
        # Load metadata
        metadata_list = load_metadata(folder)
        if not metadata_list:
            continue
        
        # Process each embedding file
        for metadata in tqdm(metadata_list, desc=f"Processing {folder.name}"):
            metrics["total_input_files"] += 1
            
            # Check for duplicates
            if deduplicate:
                content_hash = compute_content_hash(metadata)
                if content_hash in seen_hashes:
                    metrics["duplicates_found"] += 1
                    continue
                seen_hashes.add(content_hash)
            
            # Source and destination paths
            source_file = folder / metadata['filename']
            if not source_file.exists():
                logger.warning(f"File not found: {source_file}")
                continue
            
            # Create new filename to avoid conflicts
            new_filename = f"{folder.name}_{metadata['filename']}"
            dest_file = output_dir / new_filename
            
            # Copy or link file
            if copy_files:
                shutil.copy2(source_file, dest_file)
            else:
                # Create relative symlink
                try:
                    rel_path = os.path.relpath(source_file, output_dir)
                    dest_file.symlink_to(rel_path)
                except Exception as e:
                    logger.error(f"Failed to create symlink: {e}")
                    # Fall back to copying
                    shutil.copy2(source_file, dest_file)
            
            # Update metadata
            new_metadata = metadata.copy()
            new_metadata['filename'] = new_filename
            new_metadata['filepath'] = f"embeddings/{output_dir.name}/{new_filename}"
            new_metadata['original_folder'] = folder.name
            new_metadata['merge_timestamp'] = datetime.now().isoformat()
            
            merged_metadata.append(new_metadata)
            
            # Update metrics
            metrics["total_output_files"] += 1
            metrics["total_size_bytes"] += source_file.stat().st_size
    
    # Save merged metadata
    metadata_path = output_dir / "embeddings_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(merged_metadata, f, indent=2)
    
    # Create folder info file
    folder_info = {
        "merge_type": "custom" if len(folders) < 10 else "all",
        "source_folders": [f.name for f in folders],
        "num_folders": len(folders),
        "created_at": datetime.now().isoformat()
    }
    
    folder_info_path = output_dir / "folder_info.json"
    with open(folder_info_path, 'w') as f:
        json.dump(folder_info, f, indent=2)
    
    # Update metrics
    metrics["end_time"] = datetime.now().isoformat()
    metrics["duration_seconds"] = (
        datetime.fromisoformat(metrics["end_time"]) - 
        datetime.fromisoformat(metrics["start_time"])
    ).total_seconds()
    
    # Save metrics
    metrics_path = output_dir / "merge_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def verify_merge(output_dir: Path) -> bool:
    """Verify the merge was successful"""
    metadata_path = output_dir / "embeddings_metadata.json"
    if not metadata_path.exists():
        return False
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Check that all referenced files exist
    missing_files = []
    for entry in metadata:
        file_path = output_dir / entry['filename']
        if not file_path.exists():
            missing_files.append(entry['filename'])
    
    if missing_files:
        logger.error(f"Missing {len(missing_files)} files after merge")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Merge embeddings from multiple folders"
    )
    parser.add_argument(
        "--folders",
        nargs='+',
        type=str,
        default=None,
        help="Specific folders to merge (default: merge all folders)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/home/artemivashchenko/src/infinia-ai-workload-poc/nv-ingest-ddn/examples/draft_pipeline/embeddings",
        help="Base input directory containing embedding folders (default: /home/artemivashchenko/src/infinia-ai-workload-poc/nv-ingest-ddn/examples/draft_pipeline/embeddings)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/artemivashchenko/src/infinia-ai-workload-poc/nv-ingest-ddn/examples/draft_pipeline/embeddings",
        help="Base output directory (default: /home/artemivashchenko/src/infinia-ai-workload-poc/nv-ingest-ddn/examples/draft_pipeline/embeddings)"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Custom experiment name for output folder"
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Disable deduplication"
    )
    parser.add_argument(
        "--copy-files",
        action="store_true",
        help="Copy files instead of creating symlinks"
    )
    
    args = parser.parse_args()
    
    # Get folders to merge
    base_input_dir = Path(args.input_dir)
    if not base_input_dir.exists():
        logger.error(f"Input directory not found: {base_input_dir}")
        return
    
    folders = get_embedding_folders(base_input_dir, args.folders)
    if not folders:
        logger.error("No valid embedding folders found to merge")
        return
    
    # Create output directory
    base_output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.experiment_name:
        folder_name = f"{args.experiment_name}_{timestamp}"
    else:
        merge_type = "custom" if args.folders else "all"
        folder_name = f"experiment_{merge_type}_merged_{timestamp}"
    
    output_dir = base_output_dir / folder_name
    
    logger.info(f"Merging {len(folders)} folders:")
    for folder in folders:
        logger.info(f"  - {folder.name}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Deduplication: {not args.no_deduplicate}")
    logger.info(f"Copy files: {args.copy_files}")
    
    # Merge embeddings
    metrics = merge_embeddings(
        folders=folders,
        output_dir=output_dir,
        deduplicate=not args.no_deduplicate,
        copy_files=args.copy_files
    )
    
    # Verify merge
    if verify_merge(output_dir):
        logger.info("\nMerge completed successfully!")
    else:
        logger.error("\nMerge verification failed!")
        return
    
    # Print summary
    logger.info(f"\nMerge summary:")
    logger.info(f"Input files: {metrics['total_input_files']:,}")
    logger.info(f"Output files: {metrics['total_output_files']:,}")
    if not args.no_deduplicate:
        logger.info(f"Duplicates removed: {metrics['duplicates_found']:,}")
    logger.info(f"Total size: {format_size(metrics['total_size_bytes'])}")
    logger.info(f"Duration: {metrics['duration_seconds']:.2f} seconds")
    logger.info(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    main()