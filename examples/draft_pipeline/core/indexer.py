"""
Embedding Indexer for Milvus
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from pymilvus import Collection, DataType, MilvusClient, connections, utility
from tqdm import tqdm

from .utils import format_time, format_size

logger = logging.getLogger(__name__)


class EmbeddingIndexer:
    """Index embeddings into Milvus with detailed status tracking"""
    
    def __init__(
        self,
        collection_name: str,
        milvus_host: str = "localhost",
        milvus_port: int = 19530,
        embedding_dim: int = 2048,
        batch_size: int = 1000,
        recreate_collection: bool = False
    ):
        self.collection_name = collection_name
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.recreate_collection = recreate_collection
        
        self.client = None
        self.collection = None
        
        self.metrics = {
            "start_time": None,
            "end_time": None,
            "loading_time": 0,
            "ingestion_time": 0,
            "index_creation_time": 0,
            "total_embeddings": 0,
            "processed_embeddings": 0,
            "failed_embeddings": 0,
            "batches_processed": 0,
            "batches_failed": 0,
            "collection_name": collection_name
        }
    
    def connect(self):
        """Connect to Milvus"""
        try:
            connections.connect("default", host=self.milvus_host, port=self.milvus_port)
            self.client = MilvusClient(uri=f"http://{self.milvus_host}:{self.milvus_port}")
            logger.info(f"Connected to Milvus at {self.milvus_host}:{self.milvus_port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def setup_collection(self):
        """Create or setup collection"""
        try:
            exists = utility.has_collection(self.collection_name)
            
            if exists and self.recreate_collection:
                logger.info(f"Dropping existing collection: {self.collection_name}")
                utility.drop_collection(self.collection_name)
                exists = False
            
            if not exists:
                # Create schema
                schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
                
                schema.add_field(
                    field_name="id",
                    datatype=DataType.INT64,
                    is_primary=True,
                    auto_id=False,
                )
                schema.add_field(
                    field_name="vector",
                    datatype=DataType.FLOAT_VECTOR,
                    dim=self.embedding_dim
                )
                schema.add_field(
                    field_name="text",
                    datatype=DataType.VARCHAR,
                    max_length=65535
                )
                schema.add_field(
                    field_name="source",
                    datatype=DataType.VARCHAR,
                    max_length=1000
                )
                schema.add_field(
                    field_name="chunk_index",
                    datatype=DataType.INT64
                )
                schema.add_field(
                    field_name="page_number",
                    datatype=DataType.INT64
                )
                
                self.collection = Collection(self.collection_name, schema)
                logger.info(f"Created collection: {self.collection_name}")
            else:
                self.collection = Collection(self.collection_name)
                logger.info(f"Using existing collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to setup collection: {e}")
            raise
    
    def load_embeddings(self, embeddings_dir: str) -> Tuple[List[np.ndarray], List[Dict]]:
        """Load embeddings from directory"""
        start_time = time.time()
        embeddings_path = Path(embeddings_dir)
        
        # Load metadata
        metadata_file = embeddings_path / "embeddings_metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Found {len(metadata)} embeddings to load")
        
        # Load embeddings with progress bar
        embeddings = []
        valid_metadata = []
        
        with tqdm(total=len(metadata), desc="Loading embeddings") as pbar:
            for meta in metadata:
                filepath = meta.get("filepath", "")
                if filepath and os.path.exists(filepath):
                    try:
                        emb = np.load(filepath)
                        # Validate embedding dimension
                        if emb.shape[0] == self.embedding_dim:
                            embeddings.append(emb)
                            valid_metadata.append(meta)
                        else:
                            logger.warning(f"Skipping embedding with wrong dimension: {emb.shape}")
                    except Exception as e:
                        logger.error(f"Error loading {filepath}: {e}")
                else:
                    logger.warning(f"File not found: {filepath}")
                pbar.update(1)
        
        self.metrics["loading_time"] = time.time() - start_time
        self.metrics["total_embeddings"] = len(embeddings)
        
        logger.info(
            f"Loaded {len(embeddings)} valid embeddings in {format_time(self.metrics['loading_time'])}"
        )
        
        return embeddings, valid_metadata
    
    def ingest_embeddings(self, embeddings: List[np.ndarray], metadata: List[Dict]):
        """Ingest embeddings with detailed progress tracking"""
        start_time = time.time()
        total_embeddings = len(embeddings)
        
        logger.info(f"Starting ingestion of {total_embeddings} embeddings")
        logger.info(f"Batch size: {self.batch_size}")
        
        with tqdm(total=total_embeddings, desc="Ingesting embeddings") as pbar:
            for batch_start in range(0, total_embeddings, self.batch_size):
                batch_end = min(batch_start + self.batch_size, total_embeddings)
                batch_num = batch_start // self.batch_size + 1
                total_batches = (total_embeddings + self.batch_size - 1) // self.batch_size
                
                # Prepare batch data - using lists for each field
                batch_ids = []
                batch_embeddings = []
                batch_texts = []
                batch_sources = []
                batch_chunk_indices = []
                batch_page_numbers = []
                
                for i in range(batch_start, batch_end):
                    # Generate unique ID
                    unique_id = int(time.time() * 1000000) + i
                    
                    # Get metadata
                    meta = metadata[i]
                    
                    batch_ids.append(unique_id)
                    batch_embeddings.append(embeddings[i].tolist())
                    batch_texts.append(meta.get("content", "") or "")
                    batch_sources.append(meta.get("source_name", meta.get("source_file", "unknown")) or "unknown")
                    batch_chunk_indices.append(int(meta.get("chunk_index", i)))
                    # Handle None values for page_number
                    page_num = meta.get("page_number")
                    batch_page_numbers.append(int(page_num) if page_num is not None else -1)
                
                # Prepare data for insertion
                batch_data = [
                    batch_ids,
                    batch_embeddings,
                    batch_texts,
                    batch_sources,
                    batch_chunk_indices,
                    batch_page_numbers
                ]
                
                # Insert batch with status tracking
                try:
                    logger.debug(f"Inserting batch {batch_num}/{total_batches}")
                    
                    # Use collection.insert instead of client.insert
                    result = self.collection.insert(batch_data)
                    
                    if result:
                        insert_count = len(result.primary_keys)
                        self.metrics["processed_embeddings"] += insert_count
                        self.metrics["batches_processed"] += 1
                        pbar.update(insert_count)
                        
                        # Update progress description
                        pbar.set_description(
                            f"Ingesting embeddings [Batch {batch_num}/{total_batches}]"
                        )
                    else:
                        self.metrics["batches_failed"] += 1
                        logger.error(f"Failed to insert batch {batch_num}")
                        
                except Exception as e:
                    self.metrics["batches_failed"] += 1
                    logger.error(f"Error inserting batch {batch_num}: {e}")
        
        # Flush to ensure persistence
        logger.info("Flushing data to persistent storage...")
        self.collection.flush()
        
        self.metrics["ingestion_time"] = time.time() - start_time
        self.metrics["failed_embeddings"] = total_embeddings - self.metrics["processed_embeddings"]
        
        logger.info(
            f"Ingestion completed: {self.metrics['processed_embeddings']}/{total_embeddings} "
            f"embeddings in {format_time(self.metrics['ingestion_time'])}"
        )
    
    def create_index(self):
        """Create index with detailed status tracking"""
        start_time = time.time()
        
        try:
            logger.info("Creating HNSW index...")
            logger.info("Index parameters: M=16, efConstruction=200")
            
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                metric_type="L2",
                index_type="HNSW",
                index_name="embedding_index",
                params={"M": 16, "efConstruction": 200}
            )
            
            # Create index (this is synchronous)
            self.client.create_index(
                collection_name=self.collection_name,
                index_params=index_params,
                sync=False  # Start async, we'll monitor progress
            )
            
            # Monitor index building progress
            self._monitor_index_progress()
            
            self.metrics["index_creation_time"] = time.time() - start_time
            logger.info(f"Index created in {format_time(self.metrics['index_creation_time'])}")
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise
    
    def _monitor_index_progress(self):
        """Monitor index building progress with status updates"""
        index_name = "embedding_index"
        poll_interval = 2  # seconds
        start_time = time.time()
        
        # Status mapping from the Milvus code
        state_map = {2: "Started", 6: "Finished", 1: "Failed"}
        
        logger.info("Monitoring index build progress...")
        
        while True:
            try:
                # Get index building status
                indexes = self.client.list_indexes(collection_name=self.collection_name)
                
                if index_name in indexes:
                    index_info = self.client.describe_index(
                        collection_name=self.collection_name,
                        index_name=index_name
                    )
                    
                    # Get state
                    state = index_info.get("state", "Unknown")
                    if isinstance(state, int):
                        state_str = state_map.get(state, f"Unknown-{state}")
                    elif isinstance(state, str):
                        state_str = state
                    else:
                        state_str = str(state)
                        logger.debug(f"Unexpected state type: {type(state)}, value: {state}")
                    
                    # Get progress info
                    indexed_rows = index_info.get("indexed_rows", 0)
                    pending_rows = index_info.get("pending_index_rows", 0)
                    total_rows = indexed_rows + pending_rows
                    
                    if total_rows > 0:
                        progress = (indexed_rows / total_rows) * 100
                    else:
                        progress = 0
                    
                    # Log status
                    logger.info(
                        f"Index build status: {state_str} | "
                        f"Progress: {progress:.1f}% | "
                        f"Indexed: {indexed_rows:,} | "
                        f"Pending: {pending_rows:,}"
                    )
                    
                    # Check if completed - must be Finished AND no pending rows
                    # Handle both string and int state values
                    is_finished = (
                        state_str == "Finished" or 
                        state == "Finished" or 
                        (isinstance(state, int) and state == 6)
                    )
                    
                    if is_finished and pending_rows == 0:
                        elapsed_total = time.time() - start_time
                        logger.info(f"✓ Index build completed successfully in {format_time(elapsed_total)}!")
                        break
                    
                    # Check if failed
                    if state_str == "Failed" or (isinstance(state, int) and state == 1):
                        raise Exception("Index build failed!")
                    
                    # Log elapsed time periodically
                    elapsed = time.time() - start_time
                    if int(elapsed) % 60 == 0:  # Every minute
                        logger.info(f"Index building in progress for {format_time(elapsed)}...")
                    
                else:
                    logger.warning(f"Index '{index_name}' not found in collection")
                
                time.sleep(poll_interval)
                
            except Exception as e:
                if "describe_index" in str(e):
                    # Index might not be created yet
                    logger.info("Waiting for index creation to start...")
                    time.sleep(poll_interval)
                else:
                    raise
    
    def index_embeddings(self, embeddings_dir: str) -> Dict:
        """Run the complete indexing pipeline"""
        self.metrics["start_time"] = time.time()
        
        try:
            # Connect and setup
            self.connect()
            self.setup_collection()
            
            # Load embeddings
            embeddings, metadata = self.load_embeddings(embeddings_dir)
            
            # Calculate total data size
            total_size = sum(emb.nbytes for emb in embeddings)
            logger.info(f"Total data size: {format_size(total_size)}")
            
            # Ingest data
            self.ingest_embeddings(embeddings, metadata)
            
            # Create index
            self.create_index()
            
            # Load collection for queries
            logger.info("Loading collection for queries...")
            self.collection.load()
            
            # Calculate final metrics
            self.metrics["end_time"] = time.time()
            total_time = self.metrics["end_time"] - self.metrics["start_time"]
            
            # Calculate throughput
            if total_time > 0:
                self.metrics["total_time"] = total_time
                self.metrics["embeddings_per_second"] = self.metrics["processed_embeddings"] / total_time
                self.metrics["mb_per_second"] = (total_size / 1024 / 1024) / total_time
                self.metrics["success_rate"] = (
                    self.metrics["processed_embeddings"] / self.metrics["total_embeddings"] * 100
                )
            
            # Collection stats
            self.metrics["collection_entities"] = self.collection.num_entities
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            raise
        finally:
            if self.collection:
                self.collection.release()
            connections.disconnect("default")