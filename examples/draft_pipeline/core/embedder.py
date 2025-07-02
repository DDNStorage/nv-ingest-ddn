"""
Embedding Generator using NV-Ingest
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import numpy as np
from nv_ingest_client.client import Ingestor
from nv_ingest_client.primitives import JobSpec

from .utils import ensure_directory, format_time, format_size

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings using NV-Ingest"""
    
    def __init__(
        self,
        output_base_dir: str = "./embeddings",
        experiment_name: str = None,
        nv_ingest_host: str = "localhost",
        nv_ingest_port: int = 7670
    ):
        self.output_base_dir = Path(output_base_dir)
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.nv_ingest_host = nv_ingest_host
        self.nv_ingest_port = nv_ingest_port
        
        # Create output directory
        self.output_dir = ensure_directory(
            self.output_base_dir / f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        logger.info(f"Output directory: {self.output_dir}")
    
    def generate_embeddings(self, pdf_files: List[Path]) -> Dict:
        """Generate embeddings for PDF files"""
        start_time = time.time()
        metrics = {
            "start_time": start_time,
            "experiment_name": self.experiment_name,
            "output_dir": str(self.output_dir),
            "total_documents": len(pdf_files),
            "total_embeddings": 0,
            "failed_documents": 0,
            "document_metrics": []
        }
        
        try:
            # Create ingestor
            logger.info("Creating NV-Ingest client...")
            ingestor = Ingestor(
                host=self.nv_ingest_host,
                port=self.nv_ingest_port
            )
            
            # Process each PDF
            total_embeddings = 0
            
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
                    
                    # Save embeddings
                    doc_embeddings = 0
                    embeddings_metadata = []
                    
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
                                    
                                    embeddings_metadata.append(meta_entry)
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
                json.dump(embeddings_metadata, f, indent=2)
            
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
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise