"""
Embedding Generator using NV-Ingest Library Mode

Library mode is designed for small-scale workloads (< 100 PDFs) and provides
a self-contained setup that doesn't require external service deployment.
For larger workloads, use Docker Compose or Kubernetes deployment.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

# NV-Ingest library mode imports
try:
    from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import run_pipeline
    from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import PipelineCreationSchema
    from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient
    from nv_ingest_client.client import Ingestor, NvIngestClient
    from nv_ingest_client.util.process_json_files import ingest_json_results_to_blob
    NV_INGEST_LIBRARY_MODE = True
except ImportError:
    # Fallback to simple client if library mode not available
    from nv_ingest_client.client import Ingestor
    NV_INGEST_LIBRARY_MODE = False

from .utils import ensure_directory, format_time, format_size

logger = logging.getLogger(__name__)


class OptimizedEmbeddingGenerator:
    """Generate embeddings using NV-Ingest in library mode for small-scale workloads"""
    
    def __init__(
        self,
        output_base_dir: str = "./embeddings",
        experiment_name: str = None,
        nv_ingest_host: str = "localhost",
        nv_ingest_port: int = 7670,
        use_library_mode: bool = True,
        batch_size: int = 10,
        max_retries: int = 3,
        checkpoint_interval: int = 100
    ):
        self.output_base_dir = Path(output_base_dir)
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.nv_ingest_host = nv_ingest_host
        self.nv_ingest_port = nv_ingest_port
        self.use_library_mode = use_library_mode and NV_INGEST_LIBRARY_MODE
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.checkpoint_interval = checkpoint_interval
        
        # Initialize pipeline if using library mode
        self.pipeline_started = False
        self.client = None
        
        # Create output directory
        self.output_dir = ensure_directory(
            self.output_base_dir / f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Using library mode: {self.use_library_mode}")
    
    def _start_pipeline(self):
        """Start the NV-Ingest pipeline in library mode"""
        if not self.use_library_mode or self.pipeline_started:
            return
            
        logger.info("Starting NV-Ingest pipeline in library mode...")
        
        # Create pipeline configuration
        config = PipelineCreationSchema()
        
        # Start the pipeline subprocess with optimizations
        run_pipeline(
            config, 
            block=False,  # Non-blocking mode
            disable_dynamic_scaling=True,  # Better for controlled workloads
            run_in_subprocess=True  # Run in separate process for stability
        )
        
        # Create optimized client
        self.client = NvIngestClient(
            message_client_allocator=SimpleClient,
            message_client_port=7671,
            message_client_hostname="localhost"
        )
        
        self.pipeline_started = True
        logger.info("Pipeline started successfully in library mode")
        
        # Give pipeline time to initialize
        time.sleep(2)
    
    def generate_embeddings(self, pdf_files: List[Path]) -> Dict:
        """Generate embeddings for PDF files with optimized processing"""
        start_time = time.time()
        metrics = {
            "start_time": start_time,
            "experiment_name": self.experiment_name,
            "output_dir": str(self.output_dir),
            "total_documents": len(pdf_files),
            "total_embeddings": 0,
            "failed_documents": 0,
            "document_metrics": [],
            "library_mode": self.use_library_mode
        }
        
        try:
            if self.use_library_mode:
                # Start pipeline if not already started
                self._start_pipeline()
                return self._generate_embeddings_library_mode(pdf_files, metrics)
            else:
                # Fallback to standard mode
                return self._generate_embeddings_standard_mode(pdf_files, metrics)
                
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    def _generate_embeddings_library_mode(self, pdf_files: List[Path], metrics: Dict) -> Dict:
        """Generate embeddings using library mode for small workloads"""
        if len(pdf_files) > 100:
            logger.warning(f"Library mode is designed for < 100 PDFs. You have {len(pdf_files)} files.")
            logger.warning("Consider using Docker Compose or Kubernetes deployment for better performance.")
        
        logger.info(f"Processing {len(pdf_files)} files in library mode with batch size {self.batch_size}")
        
        total_embeddings = 0
        all_metadata = []
        
        # Load checkpoint if exists
        checkpoint_file = self.output_dir / "checkpoint.json"
        processed_files = set()
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                processed_files = set(checkpoint.get("processed_files", []))
                logger.info(f"Resuming from checkpoint: {len(processed_files)} files already processed")
        
        # Process files in batches for better performance
        for batch_start in range(0, len(pdf_files), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(pdf_files))
            batch_files = pdf_files[batch_start:batch_end]
            batch_num = batch_start // self.batch_size + 1
            total_batches = (len(pdf_files) + self.batch_size - 1) // self.batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_files)} files)")
            batch_start_time = time.time()
            
            try:
                # Create ingestor for this batch
                ingestor = Ingestor(
                    client=self.client
                )
                
                # Process batch
                results = (
                    ingestor
                    .files([str(f) for f in batch_files])
                    .extract(
                        extract_text=True,
                        extract_tables=False,  # Can be enabled if needed
                        extract_charts=False,   # Can be enabled if needed
                        extract_images=False,   # Can be enabled if needed
                        text_depth="page"
                    )
                    .embed()
                    .ingest()
                )
                
                # Process results
                batch_embeddings = 0
                for result_batch in results:
                    for result in result_batch:
                        if isinstance(result, dict) and "metadata" in result:
                            metadata = result["metadata"]
                            
                            if "embedding" in metadata:
                                # Determine source file
                                source_path = None
                                source_name = metadata.get("source_metadata", {}).get("source_id", "")
                                
                                for f in batch_files:
                                    if f.name in source_name or str(f) in source_name:
                                        source_path = f
                                        break
                                
                                if not source_path:
                                    # Try to extract from source_id
                                    for f in batch_files:
                                        if f.stem in source_name:
                                            source_path = f
                                            break
                                
                                if source_path:
                                    # Convert to numpy array
                                    embedding_array = np.array(metadata["embedding"], dtype=np.float32)
                                    
                                    # Generate filename
                                    chunk_index = metadata.get("chunk_index", batch_embeddings)
                                    filename = f"{source_path.stem}_chunk_{chunk_index:04d}.npy"
                                    filepath = self.output_dir / filename
                                    
                                    # Save embedding
                                    np.save(filepath, embedding_array)
                                    
                                    # Save metadata
                                    meta_entry = {
                                        "filename": filename,
                                        "filepath": str(filepath),
                                        "embedding_dim": embedding_array.shape[0],
                                        "content": metadata.get("content", ""),
                                        "source_file": str(source_path),
                                        "source_name": source_path.name,
                                        "page_number": metadata.get("source_metadata", {}).get("page_number"),
                                        "chunk_index": chunk_index,
                                        "chunk_count": metadata.get("chunk_count", 1),
                                        "collection": metadata.get("source_metadata", {}).get("collection", "default")
                                    }
                                    
                                    all_metadata.append(meta_entry)
                                    batch_embeddings += 1
                                    total_embeddings += 1
                
                batch_time = time.time() - batch_start_time
                
                # Record batch metrics
                for f in batch_files:
                    doc_metric = {
                        "document": str(f),
                        "batch_number": batch_num,
                        "processing_time": batch_time / len(batch_files),  # Average per doc
                        "status": "success"
                    }
                    metrics["document_metrics"].append(doc_metric)
                
                logger.info(
                    f"  ✓ Batch {batch_num}: Generated {batch_embeddings} embeddings in {format_time(batch_time)}"
                )
                
            except Exception as e:
                logger.error(f"  ✗ Failed to process batch {batch_num}: {e}")
                metrics["failed_documents"] += len(batch_files)
                for f in batch_files:
                    metrics["document_metrics"].append({
                        "document": str(f),
                        "batch_number": batch_num,
                        "processing_time": 0,
                        "status": "failed",
                        "error": str(e)
                    })
        
        # Save all metadata
        metadata_file = self.output_dir / "embeddings_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(all_metadata, f, indent=2)
        
        # Update metrics
        metrics["total_embeddings"] = total_embeddings
        metrics["end_time"] = time.time()
        metrics["total_time"] = metrics["end_time"] - metrics["start_time"]
        
        # Save metrics
        metrics_file = self.output_dir / "generation_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved metadata to: {metadata_file}")
        logger.info(f"Saved metrics to: {metrics_file}")
        
        return metrics
    
    def _generate_embeddings_standard_mode(self, pdf_files: List[Path], metrics: Dict) -> Dict:
        """Fallback to standard mode (same as original implementation)"""
        logger.info("Using standard mode for embedding generation")
        
        # Create ingestor
        ingestor = Ingestor(
            host=self.nv_ingest_host,
            port=self.nv_ingest_port
        )
        
        # Process each PDF (same as original implementation)
        total_embeddings = 0
        all_metadata = []
        
        for pdf_idx, pdf_path in enumerate(pdf_files):
            doc_start = time.time()
            logger.info(f"Processing {pdf_idx+1}/{len(pdf_files)}: {pdf_path.name}")
            
            try:
                # Generate embeddings
                results = (
                    ingestor
                    .files([str(pdf_path)])
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
                
                # Save embeddings (same as original)
                doc_embeddings = 0
                
                for result_batch in results:
                    for result in result_batch:
                        if isinstance(result, dict) and "metadata" in result:
                            metadata = result["metadata"]
                            
                            if "embedding" in metadata:
                                # Convert to numpy array
                                embedding_array = np.array(metadata["embedding"], dtype=np.float32)
                                
                                # Generate filename
                                chunk_index = metadata.get("chunk_index", doc_embeddings)
                                filename = f"{pdf_path.stem}_chunk_{chunk_index:04d}.npy"
                                filepath = self.output_dir / filename
                                
                                # Save embedding
                                np.save(filepath, embedding_array)
                                
                                # Save metadata
                                meta_entry = {
                                    "filename": filename,
                                    "filepath": str(filepath),
                                    "embedding_dim": embedding_array.shape[0],
                                    "content": metadata.get("content", ""),
                                    "source_file": str(pdf_path),
                                    "source_name": pdf_path.name,
                                    "page_number": metadata.get("source_metadata", {}).get("page_number"),
                                    "chunk_index": chunk_index,
                                    "chunk_count": metadata.get("chunk_count", 1),
                                    "collection": metadata.get("source_metadata", {}).get("collection", "default")
                                }
                                
                                all_metadata.append(meta_entry)
                                doc_embeddings += 1
                                total_embeddings += 1
                
                doc_time = time.time() - doc_start
                
                # Record document metrics
                doc_metric = {
                    "document": str(pdf_path),
                    "embeddings_generated": doc_embeddings,
                    "processing_time": doc_time,
                    "status": "success"
                }
                metrics["document_metrics"].append(doc_metric)
                
                logger.info(
                    f"  ✓ Generated {doc_embeddings} embeddings in {format_time(doc_time)}"
                )
                
            except Exception as e:
                logger.error(f"  ✗ Failed to process {pdf_path.name}: {e}")
                metrics["failed_documents"] += 1
                metrics["document_metrics"].append({
                    "document": str(pdf_path),
                    "embeddings_generated": 0,
                    "processing_time": time.time() - doc_start,
                    "status": "failed",
                    "error": str(e)
                })
        
        # Save all metadata
        metadata_file = self.output_dir / "embeddings_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(all_metadata, f, indent=2)
        
        # Update metrics
        metrics["total_embeddings"] = total_embeddings
        metrics["end_time"] = time.time()
        metrics["total_time"] = metrics["end_time"] - metrics["start_time"]
        
        # Save metrics
        metrics_file = self.output_dir / "generation_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved metadata to: {metadata_file}")
        logger.info(f"Saved metrics to: {metrics_file}")
        
        return metrics
    
    def close(self):
        """Clean up resources"""
        if self.pipeline_started:
            logger.info("Closing NV-Ingest pipeline...")
            # Pipeline cleanup would go here if needed
            self.pipeline_started = False