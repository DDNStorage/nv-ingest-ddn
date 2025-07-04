"""
Segment Aggregator for NVIDIA-style Bulk Indexing
Aggregates multiple small embedding files into larger segments
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SegmentAggregator:
    """Aggregate small embedding files into larger segments for efficient bulk loading"""
    
    def __init__(self, temp_dir: str = "/tmp/milvus_segments"):
        """
        Initialize segment aggregator
        
        Args:
            temp_dir: Directory for temporary segment files
        """
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized segment aggregator with temp dir: {self.temp_dir}")
    
    def aggregate_segment(self, segment_metadata: List[Dict], segment_id: str, 
                         enable_search: bool = False) -> Tuple[Dict[str, str], Dict]:
        """
        Aggregate multiple embeddings into single NPY files
        
        Args:
            segment_metadata: List of metadata dicts for embeddings in this segment
            segment_id: Unique identifier for this segment
            enable_search: If True, save additional metadata fields for search capability
            
        Returns:
            Tuple of (file_paths_dict, segment_info)
        """
        segment_dir = self.temp_dir / segment_id
        segment_dir.mkdir(exist_ok=True)
        
        all_ids = []
        all_embeddings = []
        all_texts = []
        all_sources = []
        all_chunk_indices = []
        all_page_numbers = []
        
        # Load and aggregate embeddings
        for idx, meta in enumerate(tqdm(segment_metadata, desc=f"Aggregating {segment_id}")):
            try:
                # Load embedding
                embedding = np.load(meta['filepath'])
                
                # Generate unique integer ID based on segment and index
                # This ensures unique IDs across all segments
                segment_num = int(segment_id.split('_')[-1])
                unique_id = segment_num * 1_000_000 + idx  # Allows up to 1M embeddings per segment
                
                all_ids.append(unique_id)
                all_embeddings.append(embedding)
                
                # Extract metadata
                all_texts.append(meta.get('content', '')[:65535])  # Truncate for Milvus VARCHAR limit
                all_sources.append(meta.get('source_name', meta.get('source_file', 'unknown')))
                all_chunk_indices.append(int(meta.get('chunk_index', idx)))
                
                # Handle None values for page_number
                page_num = meta.get('page_number')
                all_page_numbers.append(int(page_num) if page_num is not None else -1)
                
            except Exception as e:
                logger.error(f"Error loading embedding {meta.get('filepath', 'unknown')}: {e}")
                continue
        
        if not all_ids:
            raise ValueError(f"No valid embeddings found in segment {segment_id}")
        
        # Convert to numpy arrays with correct dtypes
        ids_array = np.array(all_ids, dtype=np.int64)  # Use int64 for IDs
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        
        # Save aggregated files
        id_path = segment_dir / "id.npy"
        vector_path = segment_dir / "vector.npy"  # Milvus expects 'vector' not 'embedding'
        metadata_path = segment_dir / "metadata.json"
        
        np.save(id_path, ids_array)
        np.save(vector_path, embeddings_array)
        
        # Save additional fields only if search is enabled
        if enable_search:
            # Convert lists to numpy arrays and save
            text_path = segment_dir / "text.npy"
            source_path = segment_dir / "source.npy"
            chunk_index_path = segment_dir / "chunk_index.npy"
            page_number_path = segment_dir / "page_number.npy"
            
            # Save as object arrays for string fields
            np.save(text_path, np.array(all_texts, dtype=object))
            np.save(source_path, np.array(all_sources, dtype=object))
            np.save(chunk_index_path, np.array(all_chunk_indices, dtype=np.int64))
            np.save(page_number_path, np.array(all_page_numbers, dtype=np.int64))
        
        # Save additional metadata for reference
        segment_info = {
            'segment_id': segment_id,
            'num_embeddings': len(all_ids),
            'id_range': [int(ids_array.min()), int(ids_array.max())],
            'embedding_dim': embeddings_array.shape[1],
            'segment_size_bytes': embeddings_array.nbytes + ids_array.nbytes,
            'segment_size_gb': (embeddings_array.nbytes + ids_array.nbytes) / (1024**3),
            'timestamp': time.time(),
            'enable_search': enable_search
        }
        
        # Include metadata in segment info only if search is enabled
        if enable_search:
            segment_info.update({
                'texts': all_texts,
                'sources': all_sources,
                'chunk_indices': all_chunk_indices,
                'page_numbers': all_page_numbers
            })
        
        with open(metadata_path, 'w') as f:
            json.dump(segment_info, f)
        
        logger.info(
            f"Aggregated {len(all_ids)} embeddings into segment {segment_id} "
            f"(size: {segment_info['segment_size_gb']:.2f} GB)"
        )
        
        # Return all file paths based on whether search is enabled
        if enable_search:
            file_paths = {
                'id': str(id_path),
                'vector': str(vector_path),
                'text': str(text_path),
                'source': str(source_path),
                'chunk_index': str(chunk_index_path),
                'page_number': str(page_number_path)
            }
        else:
            file_paths = {
                'id': str(id_path),
                'vector': str(vector_path)
            }
        
        return file_paths, segment_info
    
    def group_into_segments(self, metadata: List[Dict], max_rows: int = 240_000) -> List[List[Dict]]:
        """
        Group embeddings into segments of approximately max_rows size
        
        Args:
            metadata: List of embedding metadata dicts
            max_rows: Maximum embeddings per segment
            
        Returns:
            List of segments, where each segment is a list of metadata dicts
        """
        segments = []
        current_segment = []
        current_size = 0
        
        for meta in metadata:
            current_segment.append(meta)
            current_size += 1
            
            if current_size >= max_rows:
                segments.append(current_segment)
                current_segment = []
                current_size = 0
        
        # Add remaining embeddings
        if current_segment:
            segments.append(current_segment)
        
        logger.info(f"Grouped {len(metadata)} embeddings into {len(segments)} segments")
        for i, segment in enumerate(segments):
            logger.info(f"  Segment {i}: {len(segment)} embeddings")
        
        return segments
    
    def cleanup_temp_files(self, segment_id: str):
        """Clean up temporary files for a segment after successful upload"""
        segment_dir = self.temp_dir / segment_id
        if segment_dir.exists():
            import shutil
            shutil.rmtree(segment_dir)
            logger.debug(f"Cleaned up temporary files for segment {segment_id}")