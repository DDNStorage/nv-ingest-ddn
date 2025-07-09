#!/usr/bin/env python3
"""
Deduplication script for embeddings metadata
Removes duplicate entries and ensures metadata matches actual .npy files
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import argparse


def deduplicate_metadata(embeddings_dir: str):
    """Remove duplicate entries from embeddings metadata"""
    embeddings_path = Path(embeddings_dir)
    metadata_file = embeddings_path / "embeddings_metadata.json"
    
    if not metadata_file.exists():
        print(f"Error: Metadata file not found: {metadata_file}")
        return 1
    
    # Load current metadata
    print(f"Loading metadata from: {metadata_file}")
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print(f"Original metadata entries: {len(metadata)}")
    
    # Count actual .npy files
    npy_files = list(embeddings_path.glob("*.npy"))
    print(f"Actual .npy files found: {len(npy_files)}")
    
    # Create set of existing files for validation
    existing_files = {f.name for f in npy_files}
    
    # Deduplicate by filename, keeping first occurrence
    seen_filenames = set()
    unique_metadata = []
    duplicate_count = 0
    missing_file_count = 0
    
    for entry in metadata:
        filename = entry.get('filename', '')
        
        # Skip if we've already seen this filename
        if filename in seen_filenames:
            duplicate_count += 1
            continue
        
        # Check if the file actually exists
        if filename not in existing_files:
            missing_file_count += 1
            print(f"  Warning: Referenced file not found: {filename}")
            continue
        
        seen_filenames.add(filename)
        unique_metadata.append(entry)
    
    print(f"\nDeduplication results:")
    print(f"  Unique entries: {len(unique_metadata)}")
    print(f"  Duplicates removed: {duplicate_count}")
    print(f"  Missing files skipped: {missing_file_count}")
    
    # Verify all .npy files have metadata
    metadata_filenames = {entry['filename'] for entry in unique_metadata}
    orphaned_files = existing_files - metadata_filenames
    
    if orphaned_files:
        print(f"\nWarning: {len(orphaned_files)} .npy files without metadata:")
        for f in sorted(list(orphaned_files))[:10]:
            print(f"  {f}")
        if len(orphaned_files) > 10:
            print(f"  ... and {len(orphaned_files) - 10} more")
    
    # Save cleaned metadata
    if duplicate_count > 0 or missing_file_count > 0:
        # Backup original
        backup_file = metadata_file.with_suffix('.json.backup')
        print(f"\nBacking up original metadata to: {backup_file}")
        with open(backup_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save cleaned metadata
        print(f"Saving cleaned metadata to: {metadata_file}")
        with open(metadata_file, 'w') as f:
            json.dump(unique_metadata, f, indent=2)
        
        print("\nMetadata cleaned successfully!")
    else:
        print("\nNo duplicates found. Metadata is already clean.")
    
    # Analyze documents
    print("\n=== Document Analysis ===")
    doc_chunks = defaultdict(int)
    for entry in unique_metadata:
        source = entry.get('source_file', 'unknown')
        doc_chunks[source] += 1
    
    # Show top 10 documents by chunk count
    sorted_docs = sorted(doc_chunks.items(), key=lambda x: x[1], reverse=True)
    print(f"Total documents: {len(doc_chunks)}")
    print("\nTop 10 documents by chunk count:")
    for doc, count in sorted_docs[:10]:
        doc_name = Path(doc).name if doc != 'unknown' else 'unknown'
        print(f"  {doc_name}: {count} chunks")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Remove duplicate entries from embeddings metadata"
    )
    parser.add_argument(
        'embeddings_dir',
        help='Directory containing embeddings and metadata'
    )
    
    args = parser.parse_args()
    
    return deduplicate_metadata(args.embeddings_dir)


if __name__ == "__main__":
    sys.exit(main())