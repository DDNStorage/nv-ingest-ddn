"""
Optimized Embedding Generator using NV-Ingest
Key improvements:
- Better batch processing
- Proper error handling and retry logic
- Memory-efficient processing
- Progress tracking with detailed metrics
- Checkpoint support for resume capability
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import numpy as np
from tqdm import tqdm

from nv_ingest_client.client import Ingestor
from nv_ingest_client.primitives import JobSpec

from .utils import ensure_directory, format_time, format_size

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Optimized embedding generator using NV-Ingest"""
    
    def __init__(
        self,
        output_base_dir: str = "./embeddings",
        experiment_name: str = None,
        nv_ingest_host: str = "localhost",
        nv_ingest_port: int = 7670,
        batch_size: int = 10,
        enable_checkpoints: bool = True
    ):
        self.output_base_dir = Path(output_base_dir)
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.nv_ingest_host = nv_ingest_host
        self.nv_ingest_port = nv_ingest_port
        self.batch_size = batch_size
        self.enable_checkpoints = enable_checkpoints
        
        # Check for existing experiment directory with checkpoint
        existing_dir = self._find_resumable_experiment()
        
        if existing_dir:
            self.output_dir = existing_dir
            logger.info(f"Resuming existing experiment: {self.output_dir}")
        else:
            # Create new output directory
            self.output_dir = ensure_directory(
                self.output_base_dir / f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            logger.info(f"Creating new experiment: {self.output_dir}")
        
        # Initialize checkpoint data
        self.checkpoint_file = self.output_dir / "checkpoint.json"
        self.metadata_file = self.output_dir / "embeddings_metadata.json"
        self.processed_files = set()
        self.failed_files = set()
        self.existing_metadata = []
        
        # Load existing metadata if resuming
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    self.existing_metadata = json.load(f)
                logger.info(f"Loaded {len(self.existing_metadata)} existing metadata entries")
            except Exception as e:
                logger.warning(f"Failed to load existing metadata: {e}")
                self.existing_metadata = []
        
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Checkpoints: {'enabled' if enable_checkpoints else 'disabled'}")
    
    def _find_resumable_experiment(self) -> Optional[Path]:
        """Find existing experiment directory with checkpoint for the given experiment name"""
        if not self.enable_checkpoints:
            return None
            
        # Look for directories matching the experiment name pattern
        pattern = f"{self.experiment_name}_*"
        matching_dirs = []
        
        try:
            for path in self.output_base_dir.glob(pattern):
                if path.is_dir():
                    checkpoint_path = path / "checkpoint.json"
                    if checkpoint_path.exists():
                        # Verify it's a valid checkpoint
                        try:
                            with open(checkpoint_path, 'r') as f:
                                checkpoint_data = json.load(f)
                                if "processed" in checkpoint_data:
                                    matching_dirs.append(path)
                                    logger.info(f"Found resumable experiment: {path.name}")
                        except Exception as e:
                            logger.debug(f"Invalid checkpoint in {path}: {e}")
            
            if matching_dirs:
                # Sort by directory name (which includes timestamp) and get the most recent
                most_recent = sorted(matching_dirs)[-1]
                logger.info(f"Selected most recent resumable experiment: {most_recent.name}")
                return most_recent
                
        except Exception as e:
            logger.error(f"Error searching for resumable experiments: {e}")
            
        return None
    
    def generate_embeddings(self, pdf_files: List[Path]) -> Dict:
        """Generate embeddings for PDF files with optimized processing"""
        start_time = time.time()
        
        # Load checkpoint if exists
        remaining_files = self._load_checkpoint(pdf_files) if self.enable_checkpoints else pdf_files
        
        metrics = {
            "start_time": start_time,
            "experiment_name": self.experiment_name,
            "output_dir": str(self.output_dir),
            "total_documents": len(pdf_files),
            "total_embeddings": 0,
            "failed_documents": 0,
            "document_metrics": [],
            "batch_size": self.batch_size,
            "resumed_from_checkpoint": len(pdf_files) != len(remaining_files)
        }
        
        if metrics["resumed_from_checkpoint"]:
            logger.info(f"Resuming from checkpoint: {len(pdf_files) - len(remaining_files)} files already processed")
        
        try:
            # Create ingestor
            logger.info("Creating NV-Ingest client...")
            ingestor = Ingestor(
                host=self.nv_ingest_host,
                port=self.nv_ingest_port
            )
            
            # Process files in optimized batches
            total_embeddings = len(self.existing_metadata)  # Start with existing count
            embeddings_metadata = list(self.existing_metadata)  # Start with existing metadata
            
            # Initialize chunk counters that persist across batches
            doc_chunk_counters = defaultdict(int)
            
            # Load existing chunk counts from metadata to handle resume correctly
            for meta in self.existing_metadata:
                source_file = meta.get('source_file', '')
                chunk_idx = meta.get('chunk_index', -1)
                if source_file and chunk_idx >= 0:
                    # Update counter to be at least one more than the highest existing chunk
                    doc_chunk_counters[source_file] = max(doc_chunk_counters[source_file], chunk_idx + 1)
            
            # Calculate optimal batch size based on dataset
            optimal_batch_size = self._calculate_optimal_batch_size(len(remaining_files))
            if optimal_batch_size != self.batch_size:
                logger.info(f"Adjusted batch size from {self.batch_size} to {optimal_batch_size}")
                self.batch_size = optimal_batch_size
            
            # Process in batches
            total_batches = (len(remaining_files) + self.batch_size - 1) // self.batch_size
            
            for batch_idx in range(0, len(remaining_files), self.batch_size):
                batch = remaining_files[batch_idx:batch_idx + self.batch_size]
                batch_num = batch_idx // self.batch_size + 1
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} PDFs)")
                
                # Process batch with retry logic
                batch_results = self._process_batch_with_retry(
                    ingestor, batch, batch_num, embeddings_metadata, doc_chunk_counters
                )
                
                total_embeddings += batch_results['embeddings_count']
                
                # Save metadata incrementally after each batch
                self._save_metadata(embeddings_metadata)
                
                # Update checkpoint after each successful batch
                if self.enable_checkpoints:
                    self._save_checkpoint()
                
                # Log batch performance
                logger.info(
                    f"Batch {batch_num} complete: "
                    f"{batch_results['embeddings_count']} embeddings in "
                    f"{batch_results['processing_time']:.2f}s "
                    f"({batch_results['embeddings_per_second']:.1f} emb/s)"
                )
            
            # Save all metadata
            metadata_file = self.output_dir / "embeddings_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(embeddings_metadata, f, indent=2)
            
            # Update metrics
            metrics["total_embeddings"] = total_embeddings
            metrics["failed_documents"] = len(self.failed_files)
            metrics["end_time"] = time.time()
            metrics["total_time"] = metrics["end_time"] - metrics["start_time"]
            
            # Count total embeddings from final metadata
            metrics["total_embeddings"] = len(embeddings_metadata)
            
            # Add document-level metrics
            for pdf_path in pdf_files:
                pdf_str = str(pdf_path)
                if pdf_str in self.processed_files:
                    # Get actual embedding count from metadata
                    doc_embeddings = sum(1 for m in embeddings_metadata 
                                       if m.get('source_file') == pdf_str)
                    status = "success" if doc_embeddings > 0 else "no_embeddings"
                elif pdf_str in self.failed_files:
                    doc_embeddings = 0
                    status = "failed"
                else:
                    doc_embeddings = 0
                    status = "skipped"  # Already processed in previous run
                
                metrics["document_metrics"].append({
                    "document": pdf_str,
                    "embeddings_generated": doc_embeddings,
                    "status": status
                })
            
            # Calculate performance metrics
            if metrics["total_time"] > 0:
                metrics["overall_embeddings_per_second"] = total_embeddings / metrics["total_time"]
                metrics["documents_per_minute"] = (len(self.processed_files) / metrics["total_time"]) * 60
            
            # Save metrics
            metrics_file = self.output_dir / "generation_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"Saved metadata to: {metadata_file}")
            logger.info(f"Saved metrics to: {metrics_file}")
            
            # Clean up checkpoint file on successful completion
            if self.enable_checkpoints and len(self.failed_files) == 0:
                self.checkpoint_file.unlink(missing_ok=True)
                logger.info("Removed checkpoint file after successful completion")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    def _calculate_optimal_batch_size(self, num_files: int) -> int:
        """Calculate optimal batch size based on dataset size"""
        if num_files <= 50:
            return min(10, num_files)
        elif num_files <= 500:
            return 20
        elif num_files <= 5000:
            return 50
        else:
            return 100
    
    def _process_batch_with_retry(
        self, 
        ingestor, 
        batch: List[Path], 
        batch_num: int,
        embeddings_metadata: List[Dict],
        doc_chunk_counters: Dict[str, int],
        max_retries: int = 3
    ) -> Dict:
        """Process a batch with retry logic and correct result parsing"""
        batch_start_time = time.time()
        embeddings_count = 0
        
        for attempt in range(max_retries):
            try:
                # Convert paths to strings
                batch_paths_str = [str(pdf_path) for pdf_path in batch]
                
                logger.debug(f"Processing batch {batch_num} with {len(batch_paths_str)} files")
                
                # Process batch
                results = (
                    ingestor
                    .files(batch_paths_str)
                    .extract(
                        extract_text=True,
                        extract_tables=False,
                        extract_charts=False,
                        extract_images=False,
                        text_depth="page"
                    )
                    .embed()
                    .ingest()
                )
                
                # Process results with correct structure
                doc_embeddings_count = defaultdict(int)
                
                # Use tqdm for batch progress
                with tqdm(desc=f"Batch {batch_num} embeddings", leave=False) as pbar:
                    # Results is a list of lists, where each inner list contains result dicts
                    logger.debug(f"Results structure: type={type(results)}, len={len(results) if hasattr(results, '__len__') else 'N/A'}")
                    for idx, result_group in enumerate(results):
                        logger.debug(f"Result group {idx}: type={type(result_group)}, len={len(result_group) if hasattr(result_group, '__len__') else 'N/A'}")
                        if isinstance(result_group, list):
                            for result_dict in result_group:
                                if isinstance(result_dict, dict) and 'metadata' in result_dict:
                                    embedding_data = self._extract_embedding_from_nv_result(result_dict)
                                    if embedding_data:
                                        source_file = embedding_data['source_file']
                                        
                                        # Generate filename
                                        chunk_index = doc_chunk_counters[source_file]
                                        doc_name = Path(source_file).stem
                                        filename = f"{doc_name}_chunk_{chunk_index:04d}.npy"
                                        filepath = self.output_dir / filename
                                        
                                        # Check if file already exists
                                        if filepath.exists():
                                            logger.warning(f"File already exists, skipping: {filename}")
                                            continue
                                        
                                        # Save embedding (already validated and normalized in extraction methods)
                                        embedding_array = embedding_data['embedding']
                                        np.save(filepath, embedding_array)
                                        
                                        # Embedding dimension is always 2048 after validation
                                        embedding_dim = 2048
                                        
                                        # Create metadata entry
                                        meta_entry = {
                                            "filename": filename,
                                            "filepath": str(filepath),
                                            "embedding_dim": embedding_dim,
                                            "content": embedding_data['content'],
                                            "source_file": source_file,
                                            "source_name": Path(source_file).name,
                                            "page_number": embedding_data.get('page_number'),
                                            "chunk_index": chunk_index,
                                            "chunk_count": embedding_data.get('chunk_count', 1),
                                            "collection": embedding_data.get('collection', 'default'),
                                            "document_type": embedding_data.get('document_type', 'text')
                                        }
                                        
                                        embeddings_metadata.append(meta_entry)
                                        doc_embeddings_count[source_file] += 1
                                        doc_chunk_counters[source_file] += 1
                                        embeddings_count += 1
                                        pbar.update(1)
                
                # Mark files as processed only if they have valid embeddings
                for pdf_path in batch:
                    pdf_str = str(pdf_path)
                    embedding_count = doc_embeddings_count.get(pdf_str, 0)
                    
                    if embedding_count > 0:
                        # Only mark as processed if we got valid embeddings
                        self.processed_files.add(pdf_str)
                        logger.info(f"  ✓ {pdf_path.name}: {embedding_count} embeddings")
                    else:
                        # Don't mark as processed - will be retried on next run
                        logger.warning(f"  ⚠ {pdf_path.name}: No valid embeddings generated (skipping)")
                        # Optionally add to failed files if you want to track them
                        # self.failed_files.add(pdf_str)
                
                # Success - break retry loop
                break
                
            except Exception as e:
                logger.error(f"Batch {batch_num} attempt {attempt + 1} failed: {e}")
                import traceback
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                
                if attempt == max_retries - 1:
                    # Final attempt failed - mark files as failed
                    for pdf_path in batch:
                        self.failed_files.add(str(pdf_path))
                    embeddings_count = 0
                else:
                    # Wait before retry
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        batch_time = time.time() - batch_start_time
        
        return {
            'embeddings_count': embeddings_count,
            'processing_time': batch_time,
            'embeddings_per_second': embeddings_count / batch_time if batch_time > 0 else 0
        }
    def _extract_embedding_from_nv_result(self, result_dict: Dict) -> Optional[Dict]:
        """Extract embedding data from NV-Ingest result format"""
        try:
            
            if 'metadata' not in result_dict:
                return None
            
            metadata = result_dict['metadata']
            
            # Check for embedding
            if 'embedding' not in metadata:
                return None
            
            # Extract embedding as numpy array
            embedding_raw = metadata['embedding']
            logger.debug(f"Raw embedding type: {type(embedding_raw)}")
            
            # Check if it's a list and log its properties
            if isinstance(embedding_raw, list):
                logger.debug(f"Embedding list length: {len(embedding_raw)}")
                if len(embedding_raw) > 0:
                    logger.debug(f"First element type: {type(embedding_raw[0])}")
                    if len(embedding_raw) < 10:
                        logger.debug(f"Full embedding (small): {embedding_raw}")
                    else:
                        logger.debug(f"First 5 elements: {embedding_raw[:5]}")
            
            embedding = np.array(embedding_raw, dtype=np.float32)
            logger.debug(f"Numpy embedding shape: {embedding.shape}, size: {embedding.size}")
            
            # Validate embedding dimension
            expected_dim = 2048
            if embedding.size != expected_dim:
                logger.warning(f"Invalid embedding dimension: expected {expected_dim}, got {embedding.size}")
                return None
            
            # Ensure it's a 1D array
            if embedding.ndim != 1:
                embedding = embedding.flatten()
                logger.debug(f"Flattened embedding from {embedding.ndim}D to 1D")
            
            # Extract content
            content = metadata.get('content', '')
            
            # Extract source file - need to find it from content or other metadata
            # Since your example doesn't show source_metadata, we need to infer it
            source_file = None
            
            # Check various possible locations for source info
            if 'source_metadata' in metadata:
                source_metadata = metadata['source_metadata']
                source_file = (
                    source_metadata.get('source_id') or 
                    source_metadata.get('source_name') or
                    source_metadata.get('source_file') or
                    source_metadata.get('filename')
                )
            
            # If no source_metadata, try other fields
            if not source_file:
                source_file = (
                    metadata.get('source_file') or
                    metadata.get('source_id') or
                    metadata.get('filename') or
                    metadata.get('source') or
                    result_dict.get('source_file') or
                    result_dict.get('source_id')
                )
            
            # If still no source file, try to extract from content_url or generate one
            if not source_file:
                content_url = metadata.get('content_url', '')
                if content_url:
                    source_file = Path(content_url).name
                else:
                    # Generate a unique identifier based on content hash
                    import hashlib
                    content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
                    source_file = f"document_{content_hash}.pdf"
                    logger.warning(f"No source file found, generated: {source_file}")
            
            # Extract page number if available
            page_number = None
            if 'source_metadata' in metadata:
                page_number = metadata['source_metadata'].get('page_number')
            
            return {
                "embedding": embedding,
                "content": content[:65535],  # Truncate for Milvus VARCHAR limit
                "source_file": source_file,
                "page_number": page_number,
                "chunk_count": metadata.get('chunk_count', 1),
                "collection": metadata.get('collection', 'default'),
                "document_type": result_dict.get('document_type', 'text')
            }
            
        except Exception as e:
            logger.debug(f"Failed to extract embedding from result: {e}")
            return None
    
    def _extract_embedding_data_safe(self, result) -> Optional[Dict]:
        """Safely extract embedding and metadata from various result formats"""
        try:
            # Handle None or empty results
            if result is None:
                return None
            
            # If result is a string, skip it
            if isinstance(result, str):
                return None
            
            # Try to extract from dict-like objects
            if hasattr(result, 'get'):
                # Standard format: result with metadata
                metadata = result.get("metadata", {})
                
                # Check for embedding in metadata
                if metadata and "embedding" in metadata:
                    return self._process_metadata(metadata)
                
                # Check for embedding at top level
                if "embedding" in result:
                    return self._process_embedding_result(result)
                
                # Check for nested structures
                if "data" in result and isinstance(result["data"], dict):
                    if "metadata" in result["data"]:
                        metadata = result["data"]["metadata"]
                        if "embedding" in metadata:
                            return self._process_metadata(metadata)
            
            # Handle list/tuple results (shouldn't happen at this level, but just in case)
            if isinstance(result, (list, tuple)) and len(result) > 0:
                # Try to extract from first element
                return self._extract_embedding_data_safe(result[0])
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to extract embedding data: {e}, Result type: {type(result)}")
            return None
    def _process_metadata(self, metadata: Dict) -> Optional[Dict]:
        """Process metadata dict to extract embedding data"""
        try:
            # Extract embedding
            embedding_raw = metadata.get("embedding")
            if embedding_raw is None:
                logger.warning("No embedding found in metadata")
                return None
            
            # Convert to numpy array and ensure proper shape
            embedding = np.array(embedding_raw, dtype=np.float32)
            if embedding.size == 0:
                logger.warning("Empty embedding array")
                return None
            
            # Validate embedding dimension
            expected_dim = 2048
            if embedding.size != expected_dim:
                logger.warning(f"Invalid embedding dimension in _process_metadata: expected {expected_dim}, got {embedding.size}")
                return None
            
            # Ensure it's a 1D array
            if embedding.ndim != 1:
                embedding = embedding.flatten()
            
            # Extract source information
            source_metadata = metadata.get("source_metadata", {})
            source_file = (
                source_metadata.get("source_id") or 
                source_metadata.get("source_file") or
                metadata.get("source_file") or
                metadata.get("source_id", "")
            )
            
            if not source_file:
                logger.warning("No source file found in metadata")
                return None
            
            # Extract content
            content = (
                metadata.get("content") or 
                metadata.get("text") or
                source_metadata.get("content") or
                source_metadata.get("text", "")
            )
            
            return {
                "embedding": embedding,
                "content": content[:65535],  # Truncate for Milvus VARCHAR limit
                "source_file": source_file,
                "page_number": source_metadata.get("page_number"),
                "chunk_count": metadata.get("chunk_count"),
                "collection": source_metadata.get("collection_id")
            }
            
        except Exception as e:
            logger.debug(f"Failed to process metadata: {e}")
            return None
        
    def _process_embedding_result(self, result: Dict) -> Optional[Dict]:
        """Process result with embedding at top level"""
        try:
            embedding = np.array(result["embedding"], dtype=np.float32)
            
            # Try to find source information
            source_file = (
                result.get("source_file") or
                result.get("source_id") or
                result.get("filename", "")
            )
            
            if not source_file:
                logger.warning("No source file found in result")
                return None
            
            content = result.get("content") or result.get("text", "")
            
            return {
                "embedding": embedding,
                "content": content[:65535],
                "source_file": source_file,
                "page_number": result.get("page_number"),
                "chunk_count": result.get("chunk_count"),
                "collection": result.get("collection", "default")
            }
            
        except Exception as e:
            logger.debug(f"Failed to process embedding result: {e}")
            return None
        
    
    
    
    def _load_checkpoint(self, all_files: List[Path]) -> List[Path]:
        """Load checkpoint and return remaining files to process"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                    self.processed_files = set(checkpoint_data.get("processed", []))
                    self.failed_files = set(checkpoint_data.get("failed", []))
                
                logger.info(f"Loaded checkpoint: {len(self.processed_files)} processed, {len(self.failed_files)} failed")
                
                # Return only files that haven't been processed
                remaining = []
                for file_path in all_files:
                    if str(file_path) not in self.processed_files:
                        remaining.append(file_path)
                
                return remaining
                
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
                return all_files
        
        return all_files
    
    def _save_metadata(self, metadata: List[Dict]):
        """Save metadata incrementally"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def _save_checkpoint(self):
        """Save checkpoint data"""
        if self.enable_checkpoints:
            try:
                with open(self.checkpoint_file, 'w') as f:
                    json.dump({
                        "processed": list(self.processed_files),
                        "failed": list(self.failed_files),
                        "timestamp": datetime.now().isoformat()
                    }, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")