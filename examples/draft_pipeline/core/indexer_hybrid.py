"""
Hybrid Embedding Indexer for Milvus
Supports both stream insert and bulk insert based on dataset size
"""

import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from pymilvus import Collection, DataType, MilvusClient, connections, utility
from tqdm import tqdm

from .storage_clients import StorageClientFactory
from .segment_aggregator import SegmentAggregator
from .upload_manager import ParallelUploadManager
from .utils import format_time, format_size

logger = logging.getLogger(__name__)


class HybridEmbeddingIndexer:
    """Index embeddings into Milvus using optimized bulk indexing with segment aggregation"""
    
    def __init__(
        self,
        collection_name: str,
        milvus_host: str = "localhost",
        milvus_port: int = 19530,
        embedding_dim: int = 2048,
        batch_size: int = 5000,
        recreate_collection: bool = False,
        index_type: str = "cpu",  # "cpu" or "gpu"
        storage_mode: str = "gcs",  # "gcs" or "infinia"
        max_segment_rows: int = 240_000,  # ~1GB segments
        num_upload_workers: int = 8,
        enable_search: bool = False  # Include metadata fields for search capability
    ):
        self.collection_name = collection_name
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.recreate_collection = recreate_collection
        self.index_type = index_type.lower()
        self.storage_mode = storage_mode
        self.max_segment_rows = int(os.environ.get('MAX_SEGMENT_ROWS', max_segment_rows))
        self.num_upload_workers = int(os.environ.get('NUM_UPLOAD_WORKERS', num_upload_workers))
        self.enable_search = enable_search
        
        # Initialize storage client
        self.storage_client = None
        self._init_storage_client()
        
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
            "collection_name": collection_name,
            "index_type": index_type,
            "storage_mode": storage_mode,
            "aggregation_time": 0,
            "upload_time": 0,
            "num_segments": 0
        }
    
    def _init_storage_client(self):
        """Initialize storage client based on mode"""
        try:
            if self.storage_mode == "gcs":
                config = {
                    'bucket_name': os.environ.get('MY_BUCKET_NAME', 'infinia-multimodal-milvus'),
                    'project_id': os.environ.get('GCP_PROJECT', 'infinia-solutions-436513')
                }
            else:  # infinia
                config = {}
                
            self.storage_client = StorageClientFactory.create_client(self.storage_mode, config)
            logger.info(f"Initialized storage client for {self.storage_mode} mode")
        except Exception as e:
            logger.error(f"Failed to initialize storage client: {e}")
            raise
    
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
                
                # Always add required fields
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
                
                # Add metadata fields only if search is enabled
                if self.enable_search:
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
                    logger.info("Creating collection with search capabilities (including metadata fields)")
                else:
                    logger.info("Creating minimal collection for performance testing (id + vector only)")
                
                self.collection = Collection(self.collection_name, schema)
                logger.info(f"Created collection: {self.collection_name}")
            else:
                self.collection = Collection(self.collection_name)
                logger.info(f"Using existing collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to setup collection: {e}")
            raise
    
    def load_embeddings(self, embeddings_dir: str) -> Tuple[List[np.ndarray], List[Dict]]:
        """Load embeddings from directory with validation"""
        start_time = time.time()
        embeddings_path = Path(embeddings_dir)
        
        # Load metadata
        metadata_file = embeddings_path / "embeddings_metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Count actual .npy files for validation
        npy_files = list(embeddings_path.glob("*.npy"))
        
        logger.info(f"Found {len(metadata)} embeddings in metadata")
        logger.info(f"Found {len(npy_files)} .npy files in directory")
        
        # Validate consistency
        if len(metadata) != len(npy_files):
            logger.warning(f"⚠️  File count mismatch: {len(metadata)} metadata entries vs {len(npy_files)} .npy files")
            
            # Check for orphaned files
            metadata_files = {m.get("filename") for m in metadata if m.get("filename")}
            actual_files = {f.name for f in npy_files}
            
            orphaned = actual_files - metadata_files
            missing = metadata_files - actual_files
            
            if orphaned:
                logger.warning(f"Found {len(orphaned)} orphaned .npy files without metadata")
                logger.debug(f"Orphaned files: {sorted(list(orphaned))[:10]}...")  # Show first 10
                
            if missing:
                logger.warning(f"Found {len(missing)} metadata entries without corresponding files")
                logger.debug(f"Missing files: {sorted(list(missing))[:10]}...")
        
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
    
    def load_embeddings_partitioned(self, embeddings_dir: str) -> Tuple[Optional[List[np.ndarray]], Optional[List[Dict]], List[Dict]]:
        """Load embeddings from partitioned directory format with parallel loading.
        
        Returns:
            Tuple of (embeddings, metadata, partition_info)
            For partitioned format, embeddings and metadata are None as data is pre-aggregated
        """
        start_time = time.time()
        embeddings_path = Path(embeddings_dir)
        
        # Load master metadata
        metadata_file = embeddings_path / "embeddings_metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            master_metadata = json.load(f)
        
        # Check format version
        format_version = master_metadata.get("format_version", "1.0")
        storage_format = master_metadata.get("storage_format", "individual_files")
        
        if format_version != "2.0" or storage_format != "partitioned_aggregated":
            logger.info("Not a partitioned format, falling back to standard loading")
            embeddings, metadata = self.load_embeddings(embeddings_dir)
            return embeddings, metadata, []
        
        logger.info(f"Loading partitioned embeddings (format {format_version})")
        logger.info(f"Total vectors: {master_metadata['num_vectors']:,}")
        logger.info(f"Number of partitions: {master_metadata['num_partitions']}")
        
        # Validate partition directories
        partition_info = master_metadata.get("partitions", [])
        valid_partitions = []
        
        for p_info in partition_info:
            partition_dir = embeddings_path / p_info["partition_dir"]
            if partition_dir.exists():
                # Check required files
                embeddings_file = partition_dir / "embeddings.npy"
                ids_file = partition_dir / "ids.npy"
                metadata_file = partition_dir / "metadata.json"
                
                if all(f.exists() for f in [embeddings_file, ids_file, metadata_file]):
                    valid_partitions.append(p_info)
                else:
                    logger.warning(f"Partition {p_info['partition_id']} missing required files")
            else:
                logger.warning(f"Partition directory not found: {partition_dir}")
        
        self.metrics["loading_time"] = time.time() - start_time
        self.metrics["total_embeddings"] = master_metadata['num_vectors']
        self.metrics["num_partitions"] = len(valid_partitions)
        
        logger.info(f"Found {len(valid_partitions)} valid partitions")
        logger.info(f"Partition validation completed in {format_time(self.metrics['loading_time'])}")
        
        # Return None for embeddings/metadata as they're pre-partitioned
        # Return partition info for direct processing
        return None, None, valid_partitions
    
    def aggregate_and_upload_segments(self, embeddings: List[np.ndarray], metadata: List[Dict]) -> List[Dict]:
        """Aggregate embeddings into segments and upload to storage"""
        # Step 1: Group embeddings into segments
        aggregator = SegmentAggregator()
        segments = aggregator.group_into_segments(metadata, self.max_segment_rows)
        self.metrics["num_segments"] = len(segments)
        
        # Step 2: Aggregate each segment
        logger.info(f"Aggregating {len(segments)} segments...")
        aggregate_start = time.time()
        segments_to_upload = []
        segment_infos = {}  # Store segment info for later use
        
        for i, segment_metadata in enumerate(segments):
            segment_id = f"segment_{i:04d}"
            try:
                file_paths, segment_info = aggregator.aggregate_segment(
                    segment_metadata, segment_id, enable_search=self.enable_search
                )
                segments_to_upload.append((segment_id, file_paths))
                segment_infos[segment_id] = segment_info  # Store segment info
            except Exception as e:
                logger.error(f"Failed to aggregate segment {segment_id}: {e}")
                continue
        
        self.metrics["aggregation_time"] = time.time() - aggregate_start
        logger.info(f"Aggregated {len(segments_to_upload)} segments in {format_time(self.metrics['aggregation_time'])}")
        
        # Step 3: Upload segments in parallel
        logger.info(f"Uploading segments to {self.storage_mode} storage...")
        upload_start = time.time()
        
        upload_manager = ParallelUploadManager(self.storage_client, self.num_upload_workers)
        upload_results = upload_manager.upload_all_segments(segments_to_upload, self.collection_name)
        
        # Add segment info to upload results
        for result in upload_results:
            if result['status'] == 'success' and result['segment_id'] in segment_infos:
                result['num_embeddings'] = segment_infos[result['segment_id']]['num_embeddings']
        
        self.metrics["upload_time"] = time.time() - upload_start
        logger.info(f"Uploaded segments in {format_time(self.metrics['upload_time'])}")
        
        # Clean up temporary files
        for segment_id, _ in segments_to_upload:
            aggregator.cleanup_temp_files(segment_id)
        
        return upload_results
    
    def upload_partitions_direct(self, embeddings_dir: str, partition_info: List[Dict]) -> List[Dict]:
        """Upload pre-partitioned data directly to storage without aggregation.
        
        Args:
            embeddings_dir: Base directory containing partitions
            partition_info: List of partition information dictionaries
            
        Returns:
            List of upload results
        """
        embeddings_path = Path(embeddings_dir)
        logger.info(f"Uploading {len(partition_info)} pre-partitioned segments to {self.storage_mode} storage...")
        
        upload_start = time.time()
        
        # Prepare upload tasks for pre-partitioned data
        segments_to_upload = []
        
        for p_info in partition_info:
            partition_dir = embeddings_path / p_info["partition_dir"]
            segment_id = f"partition_{p_info['partition_id']:04d}"
            
            # Map partition files to Milvus expected format
            file_mapping = {
                "id": str(partition_dir / "ids.npy"),
                "vector": str(partition_dir / "embeddings.npy")  # Milvus expects 'vector' not 'embeddings'
            }
            
            # Add metadata fields if search is enabled
            if self.enable_search:
                metadata_file = partition_dir / "metadata.json"
                if metadata_file.exists():
                    # We'll need to convert metadata to column format
                    # For now, mark it for processing
                    file_mapping["_metadata"] = str(metadata_file)
            
            segments_to_upload.append((segment_id, file_mapping))
        
        # Upload partitions in parallel
        upload_manager = ParallelUploadManager(self.storage_client, self.num_upload_workers)
        upload_results = []
        
        with tqdm(total=len(segments_to_upload), desc="Uploading partitions") as pbar:
            upload_batch_results = upload_manager.upload_all_segments(segments_to_upload, self.collection_name)
            
            for result in upload_batch_results:
                # Add partition info to results
                partition_id = int(result['segment_id'].split('_')[-1])
                matching_partition = next((p for p in partition_info if p['partition_id'] == partition_id), None)
                
                if matching_partition:
                    result['num_embeddings'] = matching_partition['num_embeddings']
                
                upload_results.append(result)
                pbar.update(1)
        
        self.metrics["upload_time"] = time.time() - upload_start
        self.metrics["num_segments"] = len(upload_results)
        
        logger.info(f"Uploaded {len(upload_results)} partitions in {format_time(self.metrics['upload_time'])}")
        
        return upload_results
    
    def calculate_partition_groupings(self, partition_info: List[Dict], target_segment_mb: int = 1024) -> List[List[Dict]]:
        """Calculate optimal groupings of partitions to achieve target segment size.
        
        Args:
            partition_info: List of partition information dictionaries
            target_segment_mb: Target segment size in MB (default: 1024 MB = 1GB)
            
        Returns:
            List of partition groups, where each group forms one segment
        """
        # Convert target to bytes for comparison
        target_segment_bytes = target_segment_mb * 1024 * 1024
        min_segment_bytes = int(target_segment_bytes * 0.5)  # Allow 50% minimum
        max_segment_bytes = int(target_segment_bytes * 1.5)  # Allow 150% maximum
        
        logger.info(f"Calculating partition groupings for {len(partition_info)} partitions")
        logger.info(f"Target segment size: {target_segment_mb} MB ({format_size(target_segment_bytes)})")
        logger.info(f"Acceptable range: {format_size(min_segment_bytes)} - {format_size(max_segment_bytes)}")
        
        # Sort partitions by size for better grouping
        sorted_partitions = sorted(partition_info, key=lambda p: p.get('total_size_bytes', 0))
        
        groups = []
        current_group = []
        current_size = 0
        
        for partition in sorted_partitions:
            partition_size = partition.get('total_size_bytes', 0)
            
            # If single partition is already large enough, make it its own segment
            if partition_size >= min_segment_bytes:
                # Finish current group if it exists
                if current_group:
                    groups.append(current_group)
                    logger.debug(f"Created segment group with {len(current_group)} partitions, size: {format_size(current_size)}")
                
                # Add large partition as its own group
                groups.append([partition])
                logger.debug(f"Created single-partition segment, size: {format_size(partition_size)}")
                
                # Reset current group
                current_group = []
                current_size = 0
            
            # If adding this partition keeps us under max, add it
            elif current_size + partition_size <= max_segment_bytes:
                current_group.append(partition)
                current_size += partition_size
                
                # If we've reached optimal size, finish this group
                if current_size >= min_segment_bytes:
                    groups.append(current_group)
                    logger.debug(f"Created segment group with {len(current_group)} partitions, size: {format_size(current_size)}")
                    current_group = []
                    current_size = 0
            
            # Otherwise, finish current group and start new one
            else:
                if current_group:
                    groups.append(current_group)
                    logger.debug(f"Created segment group with {len(current_group)} partitions, size: {format_size(current_size)}")
                
                current_group = [partition]
                current_size = partition_size
        
        # Don't forget the last group
        if current_group:
            groups.append(current_group)
            logger.debug(f"Created final segment group with {len(current_group)} partitions, size: {format_size(current_size)}")
        
        # Log summary
        logger.info(f"Created {len(groups)} segment groups from {len(partition_info)} partitions")
        for i, group in enumerate(groups):
            group_size = sum(p.get('total_size_bytes', 0) for p in group)
            logger.info(f"  Segment {i}: {len(group)} partitions, {format_size(group_size)}")
        
        return groups
    
    def load_and_merge_partitions(self, embeddings_dir: str, partition_group: List[Dict], 
                                  segment_id: str) -> Dict[str, str]:
        """Load and merge multiple partitions into a single segment.
        
        Args:
            embeddings_dir: Base directory containing partitions
            partition_group: List of partition info dicts to merge
            segment_id: Unique identifier for the output segment
            
        Returns:
            Dictionary mapping field names to temporary file paths
        """
        embeddings_path = Path(embeddings_dir)
        segment_dir = Path(self.temp_dir) / segment_id
        segment_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Merging {len(partition_group)} partitions into segment {segment_id}")
        
        # Collect all data from partitions
        all_ids = []
        all_embeddings = []
        total_vectors = 0
        
        for partition in partition_group:
            partition_dir = embeddings_path / partition["partition_dir"]
            
            # Load partition data
            ids_path = partition_dir / "ids.npy"
            embeddings_path_file = partition_dir / "embeddings.npy"
            
            if ids_path.exists() and embeddings_path_file.exists():
                ids = np.load(ids_path)
                embeddings = np.load(embeddings_path_file)
                
                all_ids.append(ids)
                all_embeddings.append(embeddings)
                total_vectors += len(ids)
                
                logger.debug(f"Loaded partition {partition['partition_id']}: {len(ids)} vectors")
            else:
                logger.warning(f"Skipping partition {partition['partition_id']}: missing files")
        
        if not all_ids:
            raise ValueError(f"No valid partitions found for segment {segment_id}")
        
        # Concatenate all arrays
        merged_ids = np.concatenate(all_ids)
        merged_embeddings = np.concatenate(all_embeddings)
        
        # Save merged data
        id_path = segment_dir / "id.npy"
        vector_path = segment_dir / "vector.npy"
        
        np.save(id_path, merged_ids)
        np.save(vector_path, merged_embeddings)
        
        # Calculate segment size
        segment_size_bytes = id_path.stat().st_size + vector_path.stat().st_size
        
        logger.info(f"Created segment {segment_id}: {total_vectors} vectors, {format_size(segment_size_bytes)}")
        
        # Return file paths
        file_paths = {
            "id": str(id_path),
            "vector": str(vector_path)
        }
        
        return file_paths
    
    def aggregate_partitions_to_segments(self, embeddings_dir: str, partition_info: List[Dict]) -> List[Dict]:
        """Aggregate partitions into optimal-sized segments for bulk insert.
        
        Args:
            embeddings_dir: Base directory containing partitions
            partition_info: List of partition information dictionaries
            
        Returns:
            List of upload results for aggregated segments
        """
        # Set up temporary directory for aggregated segments
        self.temp_dir = Path("/tmp/milvus_segments_aggregated")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate optimal partition groupings
        partition_groups = self.calculate_partition_groupings(partition_info)
        
        # Process each group
        upload_results = []
        segments_to_upload = []
        
        logger.info(f"Aggregating {len(partition_info)} partitions into {len(partition_groups)} segments")
        
        with tqdm(total=len(partition_groups), desc="Aggregating segments") as pbar:
            for group_idx, partition_group in enumerate(partition_groups):
                segment_id = f"segment_{group_idx:04d}"
                
                try:
                    # Load and merge partitions for this segment
                    file_paths = self.load_and_merge_partitions(
                        embeddings_dir, partition_group, segment_id
                    )
                    
                    segments_to_upload.append((segment_id, file_paths))
                    
                except Exception as e:
                    logger.error(f"Failed to aggregate segment {segment_id}: {e}")
                    continue
                
                pbar.update(1)
        
        # Upload aggregated segments
        logger.info(f"Uploading {len(segments_to_upload)} aggregated segments to {self.storage_mode} storage")
        upload_start = time.time()
        
        upload_manager = ParallelUploadManager(self.storage_client, self.num_upload_workers)
        upload_results = upload_manager.upload_all_segments(segments_to_upload, self.collection_name)
        
        # Add segment info to results
        for i, result in enumerate(upload_results):
            if result['status'] == 'success' and i < len(partition_groups):
                # Calculate total embeddings in this segment
                group = partition_groups[i]
                num_embeddings = sum(p['num_embeddings'] for p in group)
                result['num_embeddings'] = num_embeddings
        
        self.metrics["upload_time"] = time.time() - upload_start
        self.metrics["num_segments"] = len(upload_results)
        
        logger.info(f"Uploaded {len(upload_results)} segments in {format_time(self.metrics['upload_time'])}")
        
        # Clean up temporary files
        for segment_id, _ in segments_to_upload:
            segment_dir = self.temp_dir / segment_id
            if segment_dir.exists():
                import shutil
                shutil.rmtree(segment_dir)
        
        return upload_results
    
    def bulk_insert_segments(self, upload_results: List[Dict]):
        """Perform bulk insert using uploaded segment files"""
        start_time = time.time()
        job_ids = []
        
        logger.info(f"Starting bulk insert for {len(upload_results)} segments")
        
        for result in upload_results:
            if result['status'] != 'success':
                continue
                
            try:
                # Perform bulk insert with the remote paths
                job_id = utility.do_bulk_insert(
                    collection_name=self.collection_name,
                    files=result['remote_paths']
                )
                job_ids.append(job_id)
                logger.info(f"Started bulk insert job {job_id} for segment {result['segment_id']}")
                
            except Exception as e:
                logger.error(f"Failed to start bulk insert for segment {result['segment_id']}: {e}")
        
        # Monitor all bulk insert jobs
        if job_ids:
            self._monitor_bulk_insert_jobs(job_ids)
        
        self.metrics["ingestion_time"] = time.time() - start_time
        logger.info(f"Bulk insert completed in {format_time(self.metrics['ingestion_time'])}")
        
        # Count successfully processed embeddings
        for result in upload_results:
            if result['status'] == 'success':
                # Use actual embedding count from segment info
                self.metrics["processed_embeddings"] += result.get('num_embeddings', 0)
    
    def _monitor_bulk_insert_jobs(self, job_ids: List[str]):
        """Monitor multiple bulk insert jobs"""
        logger.info(f"Monitoring {len(job_ids)} bulk insert jobs")
        
        completed_jobs = set()
        failed_jobs = set()
        
        with tqdm(total=len(job_ids), desc="Bulk insert progress") as pbar:
            while len(completed_jobs) + len(failed_jobs) < len(job_ids):
                for job_id in job_ids:
                    if job_id in completed_jobs or job_id in failed_jobs:
                        continue
                        
                    try:
                        state = utility.get_bulk_insert_state(task_id=job_id)
                        
                        if state.state_name == "Completed":
                            completed_jobs.add(job_id)
                            pbar.update(1)
                            row_count = getattr(state, 'row_count', 0)
                            logger.info(f"Job {job_id} completed successfully (rows: {row_count:,})")
                        elif state.state_name == "Failed":
                            failed_jobs.add(job_id)
                            pbar.update(1)
                            failed_reason = getattr(state, 'failed_reason', 'Unknown')
                            logger.error(f"Job {job_id} failed: {failed_reason}")
                            
                    except Exception as e:
                        logger.debug(f"Error checking job {job_id}: {e}")
                
                time.sleep(2)  # Poll interval
        
        logger.info(f"Bulk insert complete: {len(completed_jobs)} succeeded, {len(failed_jobs)} failed")
        self.metrics["batches_processed"] = len(completed_jobs)
        self.metrics["batches_failed"] = len(failed_jobs)
    
    def create_index(self):
        """Create index with CPU (HNSW) or GPU (GPU_CAGRA) based on configuration"""
        start_time = time.time()
        
        try:
            if self.index_type == "gpu":
                # Adaptive GPU_CAGRA parameters based on dataset size
                num_embeddings = self.metrics["total_embeddings"]
                
                if num_embeddings < 10000:
                    # Small datasets
                    intermediate_graph_degree = 32
                    graph_degree = 16
                elif num_embeddings < 100000:
                    # Medium datasets
                    intermediate_graph_degree = 64
                    graph_degree = 32
                else:
                    # Large datasets
                    intermediate_graph_degree = 128
                    graph_degree = 64
                
                logger.info(f"Creating GPU_CAGRA index for {num_embeddings} embeddings...")
                logger.info(f"Adaptive index parameters: intermediate_graph_degree={intermediate_graph_degree}, graph_degree={graph_degree}")
                
                index_params = self.client.prepare_index_params()
                index_params.add_index(
                    field_name="vector",
                    metric_type="L2",
                    index_type="GPU_CAGRA",
                    index_name="embedding_index",
                    params={
                        "intermediate_graph_degree": intermediate_graph_degree,
                        "graph_degree": graph_degree,
                        "build_algo": "NN_DESCENT"
                    }
                )
            else:
                # Optimize HNSW parameters based on dataset size
                if self.metrics["total_embeddings"] > 100000:
                    m_value = 32
                    ef_construction = 512
                elif self.metrics["total_embeddings"] > 10000:
                    m_value = 24
                    ef_construction = 400
                else:
                    m_value = 16
                    ef_construction = 200
                
                logger.info("Creating HNSW index...")
                logger.info(f"Index parameters: M={m_value}, efConstruction={ef_construction}")
                
                index_params = self.client.prepare_index_params()
                index_params.add_index(
                    field_name="vector",
                    metric_type="L2",
                    index_type="HNSW",
                    index_name="embedding_index",
                    params={"M": m_value, "efConstruction": ef_construction}
                )
            
            # Create index
            self.client.create_index(
                collection_name=self.collection_name,
                index_params=index_params,
                sync=False
            )
            
            # Monitor index building progress
            self._monitor_index_progress()
            
            self.metrics["index_creation_time"] = time.time() - start_time
            logger.info(f"Index created in {format_time(self.metrics['index_creation_time'])}")
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise
    
    def _monitor_index_progress(self):
        """Monitor index building progress"""
        index_name = "embedding_index"
        poll_interval = 2
        start_time = time.time()
        state_map = {2: "Started", 6: "Finished", 1: "Failed"}
        
        logger.info("Monitoring index build progress...")
        
        while True:
            try:
                indexes = self.client.list_indexes(collection_name=self.collection_name)
                
                if index_name in indexes:
                    index_info = self.client.describe_index(
                        collection_name=self.collection_name,
                        index_name=index_name
                    )
                    
                    state = index_info.get("state", "Unknown")
                    if isinstance(state, int):
                        state_str = state_map.get(state, f"Unknown-{state}")
                    else:
                        state_str = str(state)
                    
                    indexed_rows = index_info.get("indexed_rows", 0)
                    pending_rows = index_info.get("pending_index_rows", 0)
                    total_rows = indexed_rows + pending_rows
                    
                    progress = (indexed_rows / total_rows * 100) if total_rows > 0 else 0
                    
                    logger.info(
                        f"Index build status: {state_str} | "
                        f"Progress: {progress:.1f}% | "
                        f"Indexed: {indexed_rows:,} | "
                        f"Pending: {pending_rows:,}"
                    )
                    
                    is_finished = (
                        state_str == "Finished" or 
                        state == "Finished" or 
                        (isinstance(state, int) and state == 6)
                    )
                    
                    if is_finished and pending_rows == 0:
                        elapsed_total = time.time() - start_time
                        logger.info(f"✓ Index build completed successfully in {format_time(elapsed_total)}!")
                        logger.info(f"Total rows indexed: {indexed_rows:,}")
                        logger.info(f"Average indexing rate: {indexed_rows/elapsed_total:.0f} rows/sec")
                        break
                    
                    if state_str == "Failed" or (isinstance(state, int) and state == 1):
                        raise Exception("Index build failed!")
                    
                time.sleep(poll_interval)
                
            except Exception as e:
                if "describe_index" in str(e):
                    logger.info("Waiting for index creation to start...")
                    time.sleep(poll_interval)
                else:
                    raise
    
    def index_embeddings(self, embeddings_dir: str) -> Dict:
        """Run the complete indexing pipeline with optimized bulk approach"""
        self.metrics["start_time"] = time.time()
        
        try:
            # Connect and setup
            self.connect()
            self.setup_collection()
            
            # Try partitioned loading first
            embeddings, metadata, partition_info = self.load_embeddings_partitioned(embeddings_dir)
            
            if partition_info:
                # Partitioned format - use aggregation for optimal segment sizes
                logger.info("Using partitioned data pipeline with smart aggregation")
                
                # Calculate total data size from partition info
                total_size = sum(p['total_size_bytes'] for p in partition_info)
                logger.info(f"Total data size: {format_size(total_size)}")
                
                # Step 1: Aggregate partitions into optimal segments and upload
                self.metrics["aggregation_start"] = time.time()
                upload_results = self.aggregate_partitions_to_segments(embeddings_dir, partition_info)
                self.metrics["aggregation_time"] = time.time() - self.metrics["aggregation_start"]
                
                # Step 2: Perform bulk insert
                self.bulk_insert_segments(upload_results)
            else:
                # Legacy format - use original pipeline
                logger.info("Using legacy individual file pipeline")
                
                # Calculate total data size
                total_size = sum(emb.nbytes for emb in embeddings)
                logger.info(f"Total data size: {format_size(total_size)}")
                
                # Step 1: Aggregate and upload segments
                upload_results = self.aggregate_and_upload_segments(embeddings, metadata)
                
                # Step 2: Perform bulk insert
                self.bulk_insert_segments(upload_results)
            
            # Step 3: Create index AFTER data ingestion (following the optimized approach)
            self.create_index()
            
            # Step 4: Load collection for queries
            #logger.info("Loading collection for queries...")
            #self.collection.load()
            
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
            
            # Log summary
            logger.info("="*60)
            logger.info("INDEXING COMPLETE")
            logger.info("="*60)
            logger.info(f"Total embeddings: {self.metrics['total_embeddings']:,}")
            logger.info(f"Processed embeddings: {self.metrics['processed_embeddings']:,}")
            logger.info(f"Number of segments: {self.metrics['num_segments']}")
            logger.info(f"Aggregation time: {format_time(self.metrics['aggregation_time'])}")
            logger.info(f"Upload time: {format_time(self.metrics['upload_time'])}")
            logger.info(f"Ingestion time: {format_time(self.metrics['ingestion_time'])}")
            logger.info(f"Index creation time: {format_time(self.metrics['index_creation_time'])}")
            logger.info(f"Total time: {format_time(total_time)}")
            logger.info(f"Throughput: {self.metrics['embeddings_per_second']:.0f} embeddings/second")
            logger.info("="*60)
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            raise
        finally:
            if self.collection:
                self.collection.release()
            connections.disconnect("default")