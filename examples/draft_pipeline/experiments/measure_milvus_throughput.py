#!/usr/bin/env python3
"""
Milvus Bulk Insert Throughput Measurement

This script measures the throughput of Milvus bulk insert operations in terms of:
- Embeddings per second
- Megabytes per second

Measurements are taken only during the bulk insert phase (not including upload).
Supports both Infinia and GCS storage backends.

Usage:
    python measure_milvus_throughput.py --datasets 1M,2M,5M,10M,20M --storage-mode infinia
    python measure_milvus_throughput.py --datasets 1M,2M,5M,10M,20M --storage-mode gcs
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import threading
import queue
import argparse
import logging
import shutil
from dataclasses import dataclass, asdict, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Milvus imports
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

# Configure matplotlib
plt.switch_backend('Agg')
sns.set_style("whitegrid")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def format_time(seconds: float) -> str:
    """Format time in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


@dataclass
class ThroughputSample:
    """Single throughput measurement during bulk insert"""
    timestamp: float  # Relative to bulk insert start
    completed_embeddings: int
    completed_bytes: int
    active_jobs: int
    completed_jobs: int
    embeddings_per_second: float
    megabytes_per_second: float


@dataclass
class BulkInsertThroughputMetrics:
    """Throughput metrics for bulk insert operation"""
    start_time: float
    end_time: float
    num_jobs: int
    num_segments: int
    total_embeddings: int
    total_bytes: int
    throughput_samples: List[ThroughputSample] = field(default_factory=list)
    job_completion_times: List[float] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def avg_embeddings_per_second(self) -> float:
        return self.total_embeddings / self.duration if self.duration > 0 else 0
    
    @property
    def avg_megabytes_per_second(self) -> float:
        return (self.total_bytes / (1024 * 1024)) / self.duration if self.duration > 0 else 0
    
    @property
    def peak_embeddings_per_second(self) -> float:
        if not self.throughput_samples:
            return 0
        return max(s.embeddings_per_second for s in self.throughput_samples)
    
    @property
    def peak_megabytes_per_second(self) -> float:
        if not self.throughput_samples:
            return 0
        return max(s.megabytes_per_second for s in self.throughput_samples)
    
    @property
    def time_to_first_completion(self) -> Optional[float]:
        if self.job_completion_times:
            return min(self.job_completion_times)
        return None


class ThroughputMonitor:
    """Monitors throughput during bulk insert operations"""
    
    def __init__(self, interval: float = 1.0):
        """
        Initialize throughput monitor
        
        Args:
            interval: Sampling interval in seconds
        """
        self.interval = interval
        self.monitoring = False
        self.samples = []
        self.monitor_thread = None
        self.start_time = None
        self.completed_embeddings = 0
        self.completed_bytes = 0
        self.active_jobs = 0
        self.completed_jobs = 0
        self._lock = threading.Lock()
        self._last_sample_time = None
        self._last_embeddings = 0
        self._last_bytes = 0
        
    def start(self):
        """Start throughput monitoring"""
        self.monitoring = True
        self.samples = []
        self.start_time = time.time()
        self._last_sample_time = self.start_time
        self._last_embeddings = 0
        self._last_bytes = 0
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("Throughput monitoring started")
        
    def stop(self) -> List[ThroughputSample]:
        """Stop throughput monitoring and return samples"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info(f"Throughput monitoring stopped. Collected {len(self.samples)} samples")
        return self.samples
    
    def update_progress(self, completed_embeddings: int, completed_bytes: int, 
                       active_jobs: int, completed_jobs: int):
        """Update current progress metrics"""
        with self._lock:
            self.completed_embeddings = completed_embeddings
            self.completed_bytes = completed_bytes
            self.active_jobs = active_jobs
            self.completed_jobs = completed_jobs
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            current_time = time.time()
            elapsed = current_time - self.start_time
            
            with self._lock:
                # Calculate instantaneous throughput
                time_delta = current_time - self._last_sample_time
                if time_delta > 0:
                    embeddings_delta = self.completed_embeddings - self._last_embeddings
                    bytes_delta = self.completed_bytes - self._last_bytes
                    
                    embeddings_per_second = embeddings_delta / time_delta
                    megabytes_per_second = (bytes_delta / (1024 * 1024)) / time_delta
                else:
                    embeddings_per_second = 0
                    megabytes_per_second = 0
                
                # Create sample
                sample = ThroughputSample(
                    timestamp=elapsed,
                    completed_embeddings=self.completed_embeddings,
                    completed_bytes=self.completed_bytes,
                    active_jobs=self.active_jobs,
                    completed_jobs=self.completed_jobs,
                    embeddings_per_second=embeddings_per_second,
                    megabytes_per_second=megabytes_per_second
                )
                
                self.samples.append(sample)
                
                # Update last values
                self._last_sample_time = current_time
                self._last_embeddings = self.completed_embeddings
                self._last_bytes = self.completed_bytes
            
            time.sleep(self.interval)


class StorageClient:
    """Storage client supporting both Infinia and GCS"""
    
    def __init__(self, mode: str = "infinia"):
        self.mode = mode
        
        if mode == "infinia":
            # Set Infinia credentials
            os.environ["MY_STORAGE_ENDPOINT"] = "https://10.168.15.234:8111"
            os.environ["MY_ACCESS_KEY_ID"] = "D8BKP21LB091U1YFUX9Z"
            os.environ["MY_SECRET_ACCESS_KEY"] = "81VqVHbeDz5tITocnPBeM1KltMCr1YaMzv3XbBTc"
            os.environ["MY_BUCKET_NAME"] = "milvus-db"
            
            import boto3
            from boto3.s3.transfer import TransferConfig
            from botocore.config import Config as BotoConfig
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            self.bucket_name = "milvus-db"
            self.client = boto3.client(
                's3',
                endpoint_url="https://10.168.15.234:8111",
                aws_access_key_id="D8BKP21LB091U1YFUX9Z",
                aws_secret_access_key="81VqVHbeDz5tITocnPBeM1KltMCr1YaMzv3XbBTc",
                config=BotoConfig(
                    signature_version='s3v4',
                    max_pool_connections=60
                ),
                verify=False
            )
            
            self.transfer_config = TransferConfig(
                multipart_threshold=100 * 1024 * 1024,
                multipart_chunksize=64 * 1024 * 1024,
                max_concurrency=10,
                use_threads=True
            )
            
            logger.info("Initialized Infinia storage client")
        
        elif mode == "gcs":
            # GCS configuration
            import google.auth
            from google.cloud import storage as gcs
            from google.auth.transport.requests import AuthorizedSession
            
            self.bucket_name = os.environ.get('MY_BUCKET_NAME', 'infinia-multimodal-milvus')
            project_id = os.environ.get('GCP_PROJECT', 'infinia-solutions-436513')
            
            try:
                if os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
                    logger.info(f"Using service account from: {os.environ['GOOGLE_APPLICATION_CREDENTIALS']}")
                    self.gcs_client = gcs.Client.from_service_account_json(
                        os.environ['GOOGLE_APPLICATION_CREDENTIALS'],
                        project=project_id
                    )
                else:
                    credentials, project = google.auth.default(
                        scopes=['https://www.googleapis.com/auth/cloud-platform',
                                'https://www.googleapis.com/auth/devstorage.full_control']
                    )
                    self.gcs_client = gcs.Client(
                        project=project_id or project,
                        credentials=credentials
                    )
                    logger.info(f"Using default credentials for project: {project_id or project}")
                
                if hasattr(self.gcs_client._http, '_pool_manager'):
                    self.gcs_client._http._pool_manager.connection_pool_kw['maxsize'] = 60
                    self.gcs_client._http._pool_manager.clear()
                
                self.bucket = self.gcs_client.bucket(self.bucket_name)
                
                # Test access
                try:
                    test_blob = self.bucket.blob('_test_access_check')
                    test_blob.upload_from_string('test')
                    test_blob.delete()
                    logger.info(f"✓ GCS bucket write access verified for: {self.bucket_name}")
                except Exception as e:
                    logger.error(f"❌ Bucket write access test failed: {e}")
                    if "403" in str(e):
                        logger.error("Try: gcloud auth application-default login --scopes=https://www.googleapis.com/auth/cloud-platform")
                    raise
                
                logger.info(f"Initialized GCS storage client for bucket: {self.bucket_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize GCS client: {e}")
                raise
        
        else:
            raise ValueError(f"Unsupported storage mode: {mode}")
    
    def upload_file(self, local_path: str, remote_path: str) -> str:
        """Upload file with multipart support for large files"""
        file_size = os.path.getsize(local_path)
        
        if self.mode == "infinia":
            if file_size > 100 * 1024 * 1024:
                logger.debug(f"Using multipart upload for {format_size(file_size)} file")
                self.client.upload_file(
                    local_path, 
                    self.bucket_name, 
                    remote_path,
                    Config=self.transfer_config
                )
            else:
                self.client.upload_file(local_path, self.bucket_name, remote_path)
                
            return f"s3://{self.bucket_name}/{remote_path}"
            
        elif self.mode == "gcs":
            blob = self.bucket.blob(remote_path)
            
            max_retries = 3
            retry_delay = 1.0
            
            for attempt in range(max_retries):
                try:
                    if file_size > 100 * 1024 * 1024:
                        logger.debug(f"Using resumable upload for {format_size(file_size)} file")
                        blob.chunk_size = 32 * 1024 * 1024
                        blob.upload_from_filename(local_path)
                    else:
                        blob.upload_from_filename(local_path)
                    
                    break
                    
                except Exception as e:
                    if "Connection pool is full" in str(e) and attempt < max_retries - 1:
                        logger.warning(f"Connection pool full, retrying in {retry_delay}s")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        raise
                
            return f"gs://{self.bucket_name}/{remote_path}"


class MilvusThroughputTester:
    """Test Milvus bulk insert throughput"""
    
    def __init__(self, collection_name: str, storage_mode: str = "infinia", 
                 embedding_dim: int = 2048):
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.storage_mode = storage_mode
        
        # Initialize connections
        self.milvus_host = "localhost"
        self.milvus_port = 19530
        
        # Storage client
        self.storage_client = StorageClient(storage_mode)
        
        # Throughput monitor
        self.throughput_monitor = ThroughputMonitor(interval=1.0)
        
        # Metrics
        self.metrics = {}
        
        # Size calculations
        self.bytes_per_embedding = (self.embedding_dim * 4) + 8  # float32 vector + int64 id
        
    def connect(self):
        """Connect to Milvus"""
        connections.connect(
            alias="default",
            host=self.milvus_host,
            port=self.milvus_port
        )
        logger.info(f"Connected to Milvus at {self.milvus_host}:{self.milvus_port}")
    
    def create_collection(self, recreate: bool = True):
        """Create minimal collection for bulk insert testing"""
        if recreate and utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            logger.info(f"Dropped existing collection: {self.collection_name}")
        
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="Collection for throughput testing"
        )
        
        collection = Collection(
            name=self.collection_name,
            schema=schema
        )
        
        logger.info(f"Created collection: {self.collection_name}")
        return collection
    
    def prepare_segments_fast(self, embeddings_dir: Path) -> List[Dict]:
        """Ultra-fast segment preparation"""
        metadata_path = embeddings_dir / "embeddings_metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
            
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        logger.info(f"Found {len(metadata['partitions'])} partitions to use as segments")
        
        segments = []
        for partition in metadata['partitions']:
            partition_dir = embeddings_dir / partition['partition_dir']
            
            id_file = partition_dir / "ids.npy"
            vector_file = partition_dir / "embeddings.npy"
            
            segments.append({
                'segment_id': partition['partition_dir'],
                'local_files': {
                    'id': str(id_file),
                    'vector': str(vector_file)
                },
                'num_embeddings': partition['num_embeddings'],
                'size_bytes': partition['num_embeddings'] * self.bytes_per_embedding
            })
            
        return segments
    
    def upload_segments_parallel(self, segments: List[Dict]) -> List[Dict]:
        """Upload segments in parallel"""
        logger.info(f"Starting parallel upload of {len(segments)} segments to {self.storage_mode}")
        
        def upload_single_segment(segment: Dict) -> Dict:
            try:
                segment_id = segment['segment_id']
                remote_paths = []
                
                for field_name, local_path in segment['local_files'].items():
                    if 'ids.npy' in local_path:
                        milvus_field_name = 'id'
                    elif 'embeddings.npy' in local_path:
                        milvus_field_name = 'vector'
                    else:
                        milvus_field_name = field_name
                    
                    remote_path = f"throughput_test/{self.storage_mode}/{self.collection_name}/{segment_id}/{milvus_field_name}.npy"
                    url = self.storage_client.upload_file(local_path, remote_path)
                    remote_paths.append(remote_path)
                
                return {
                    'segment_id': segment_id,
                    'remote_paths': remote_paths,
                    'num_embeddings': segment['num_embeddings'],
                    'size_bytes': segment['size_bytes'],
                    'status': 'success'
                }
            except Exception as e:
                logger.error(f"Failed to upload segment {segment['segment_id']}: {e}")
                return {
                    'segment_id': segment['segment_id'],
                    'status': 'failed',
                    'error': str(e)
                }
        
        max_workers = 8 if self.storage_mode == "gcs" else 16
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for segment in segments:
                future = executor.submit(upload_single_segment, segment)
                futures.append(future)
            
            results = []
            with tqdm(total=len(futures), desc="Uploading segments") as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Upload error: {e}")
                        pbar.update(1)
        
        successful = sum(1 for r in results if r.get('status') == 'success')
        logger.info(f"Upload complete: {successful}/{len(segments)} segments uploaded successfully")
        
        return results
    
    def bulk_insert_with_monitoring(self, upload_results: List[Dict]) -> BulkInsertThroughputMetrics:
        """Perform bulk insert with throughput monitoring"""
        # Filter successful uploads
        successful_uploads = [r for r in upload_results if r.get('status') == 'success']
        
        if not successful_uploads:
            raise ValueError("No successful uploads to insert")
        
        # Start throughput monitoring
        self.throughput_monitor.start()
        bulk_start = time.time()
        
        # Submit bulk insert jobs
        job_ids = []
        job_info = {}  # job_id -> segment info
        
        logger.info(f"Starting bulk insert for {len(successful_uploads)} segments")
        
        for result in successful_uploads:
            try:
                job_id = utility.do_bulk_insert(
                    collection_name=self.collection_name,
                    files=result['remote_paths']
                )
                job_ids.append(job_id)
                job_info[job_id] = {
                    'segment_id': result['segment_id'],
                    'num_embeddings': result['num_embeddings'],
                    'size_bytes': result['size_bytes'],
                    'submit_time': time.time() - bulk_start
                }
                logger.info(f"Started bulk insert job {job_id} for segment {result['segment_id']}")
            except Exception as e:
                logger.error(f"Failed to start bulk insert for segment {result['segment_id']}: {e}")
        
        # Monitor jobs until complete
        self._monitor_jobs_with_throughput(job_ids, job_info, bulk_start)
        
        # Stop throughput monitoring
        throughput_samples = self.throughput_monitor.stop()
        
        # Calculate totals
        total_embeddings = sum(info['num_embeddings'] for info in job_info.values())
        total_bytes = sum(info['size_bytes'] for info in job_info.values())
        
        # Extract job completion times
        job_completion_times = [info.get('completion_time', float('inf')) 
                              for info in job_info.values() 
                              if 'completion_time' in info]
        
        # Create metrics
        metrics = BulkInsertThroughputMetrics(
            start_time=bulk_start,
            end_time=time.time(),
            num_jobs=len(job_ids),
            num_segments=len(successful_uploads),
            total_embeddings=total_embeddings,
            total_bytes=total_bytes,
            throughput_samples=throughput_samples,
            job_completion_times=job_completion_times
        )
        
        logger.info(f"\nBulk insert completed:")
        logger.info(f"  Duration: {metrics.duration:.1f}s")
        logger.info(f"  Average throughput: {metrics.avg_embeddings_per_second:.0f} embeddings/s")
        logger.info(f"  Average throughput: {metrics.avg_megabytes_per_second:.1f} MB/s")
        logger.info(f"  Peak throughput: {metrics.peak_embeddings_per_second:.0f} embeddings/s")
        logger.info(f"  Peak throughput: {metrics.peak_megabytes_per_second:.1f} MB/s")
        logger.info(f"  Time to first completion: {metrics.time_to_first_completion:.1f}s")
        
        return metrics
    
    def _monitor_jobs_with_throughput(self, job_ids: List[str], job_info: Dict, start_time: float):
        """Monitor bulk insert jobs and update throughput metrics"""
        logger.info(f"Monitoring {len(job_ids)} bulk insert jobs")
        
        completed_jobs = set()
        failed_jobs = set()
        completed_embeddings = 0
        completed_bytes = 0
        
        with tqdm(total=len(job_ids), desc="Bulk insert progress") as pbar:
            while len(completed_jobs) + len(failed_jobs) < len(job_ids):
                active_count = len(job_ids) - len(completed_jobs) - len(failed_jobs)
                
                for job_id in job_ids:
                    if job_id in completed_jobs or job_id in failed_jobs:
                        continue
                    
                    try:
                        state = utility.get_bulk_insert_state(job_id)
                        
                        if state.state_name == "Completed":
                            completed_jobs.add(job_id)
                            info = job_info[job_id]
                            info['completion_time'] = time.time() - start_time
                            
                            completed_embeddings += info['num_embeddings']
                            completed_bytes += info['size_bytes']
                            
                            pbar.update(1)
                            logger.debug(f"Job {job_id} completed successfully")
                            
                        elif state.state_name in ["Failed", "FailedAndCleaned"]:
                            failed_jobs.add(job_id)
                            pbar.update(1)
                            logger.error(f"Job {job_id} failed: {state.failed_reason}")
                            
                    except Exception as e:
                        logger.error(f"Error checking job {job_id}: {e}")
                
                # Update throughput monitor
                self.throughput_monitor.update_progress(
                    completed_embeddings=completed_embeddings,
                    completed_bytes=completed_bytes,
                    active_jobs=active_count,
                    completed_jobs=len(completed_jobs)
                )
                
                # Avoid busy waiting
                if active_count > 0:
                    time.sleep(0.5)
        
        logger.info(f"Bulk insert complete: {len(completed_jobs)} succeeded, {len(failed_jobs)} failed")
    
    def run_test(self, embeddings_dir: Path) -> Dict:
        """Run complete throughput test"""
        try:
            # Connect to Milvus
            self.connect()
            
            # Create collection
            self.create_collection(recreate=True)
            
            # Prepare segments
            logger.info("Preparing segments from partitions...")
            segments = self.prepare_segments_fast(embeddings_dir)
            
            # Upload segments
            logger.info("Uploading segments to storage...")
            upload_start = time.time()
            upload_results = self.upload_segments_parallel(segments)
            upload_time = time.time() - upload_start
            
            # Bulk insert with throughput monitoring
            bulk_metrics = self.bulk_insert_with_monitoring(upload_results)
            
            # Collect final metrics
            collection = Collection(self.collection_name)
            
            return {
                'embeddings_dir': str(embeddings_dir),
                'storage_mode': self.storage_mode,
                'num_embeddings': bulk_metrics.total_embeddings,
                'total_size_mb': bulk_metrics.total_bytes / (1024 * 1024),
                'num_segments': bulk_metrics.num_segments,
                'upload_time': upload_time,
                'bulk_insert_metrics': bulk_metrics,
                'collection_entities': collection.num_entities,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Test failed: {e}")
            raise


class MilvusThroughputAnalyzer:
    """Analyzer for throughput experiments"""
    
    def __init__(self, output_dir: str = "./milvus_throughput_results"):
        self.output_dir = Path(output_dir)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{self.run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.run_dir / "raw_data").mkdir(exist_ok=True)
        (self.run_dir / "visualizations").mkdir(exist_ok=True)
        
        self.results = []
        
    def analyze_dataset(self, num_embeddings: int, collection_prefix: str, 
                       storage_mode: str = "infinia") -> Dict:
        """Analyze throughput for a specific dataset size"""
        logger.info(f"\n{'='*80}")
        logger.info(f"Analyzing {num_embeddings:,} embeddings with {storage_mode} storage")
        logger.info(f"{'='*80}")
        
        # Generate embeddings
        embeddings_dir = self._generate_or_use_cached_embeddings(num_embeddings)
        
        # Create tester
        collection_name = f"{collection_prefix}_{num_embeddings}_{storage_mode}"
        tester = MilvusThroughputTester(
            collection_name=collection_name,
            storage_mode=storage_mode
        )
        
        # Run test
        try:
            result = tester.run_test(embeddings_dir)
            result['num_embeddings_requested'] = num_embeddings
            
            # Save raw data
            self._save_raw_data(num_embeddings, storage_mode, result)
            
            logger.info(f"Analysis complete for {num_embeddings:,} embeddings on {storage_mode}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing dataset: {e}")
            raise
    
    def _find_optimal_embedding_combination(self, target_embeddings: int, cache_base: Path) -> Tuple[List[Path], int]:
        """
        Find optimal combination of existing embeddings to minimize generation.
        
        Returns:
            Tuple of (list of existing embedding directories, remaining embeddings to generate)
        """
        # Find all existing embedding directories
        existing_dirs = []
        for dir_path in cache_base.glob("embeddings_*"):
            if dir_path.is_dir() and (dir_path / "embeddings_metadata.json").exists():
                try:
                    # Extract number from directory name
                    num_str = dir_path.name.replace("embeddings_", "")
                    num = int(num_str)
                    existing_dirs.append((num, dir_path))
                except ValueError:
                    continue
        
        if not existing_dirs:
            return [], target_embeddings
        
        # Sort by size descending for greedy approach
        existing_dirs.sort(key=lambda x: x[0], reverse=True)
        
        # Greedy algorithm to find best combination
        selected = []
        current_total = 0
        
        for num, dir_path in existing_dirs:
            if current_total + num <= target_embeddings:
                selected.append(dir_path)
                current_total += num
                logger.info(f"Selected existing embeddings: {num:,} from {dir_path.name}")
        
        remaining = target_embeddings - current_total
        logger.info(f"Total from existing: {current_total:,}, Need to generate: {remaining:,}")
        
        return selected, remaining

    def _merge_existing_embeddings(self, existing_dirs: List[Path], additional_embeddings: int, 
                                 target_dir: Path, total_embeddings: int) -> Path:
        """
        Merge existing embeddings and generate additional ones if needed.
        
        Args:
            existing_dirs: List of directories containing existing embeddings
            additional_embeddings: Number of additional embeddings to generate
            target_dir: Target directory for merged embeddings
            total_embeddings: Total number of embeddings needed
        
        Returns:
            Path to the merged embeddings directory
        """
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate total partitions needed
        embeddings_per_partition = 100_000
        total_partitions = max(10, (total_embeddings + embeddings_per_partition - 1) // embeddings_per_partition)
        
        if total_partitions > 200:
            total_partitions = 200
            embeddings_per_partition = total_embeddings // total_partitions
        
        metadata = {
            "format_version": "2.0",
            "storage_format": "partitioned_aggregated",
            "num_vectors": total_embeddings,
            "embedding_dim": 2048,
            "num_partitions": 0,
            "partitions": []
        }
        
        partition_idx = 0
        current_embedding_idx = 0
        
        # First, copy partitions from existing directories
        logger.info("Merging existing embeddings...")
        for source_dir in existing_dirs:
            # Load source metadata
            with open(source_dir / "embeddings_metadata.json") as f:
                source_metadata = json.load(f)
            
            # Copy each partition
            for source_partition in source_metadata["partitions"]:
                source_partition_dir = source_dir / source_partition["partition_dir"]
                target_partition_dir = target_dir / f"partition_{partition_idx:04d}"
                
                # Create target partition directory
                target_partition_dir.mkdir(exist_ok=True)
                
                # Copy files using hard links for efficiency (same filesystem)
                try:
                    os.link(source_partition_dir / "ids.npy", target_partition_dir / "ids.npy")
                    os.link(source_partition_dir / "embeddings.npy", target_partition_dir / "embeddings.npy")
                    logger.debug(f"Hard linked partition {partition_idx} from {source_partition_dir}")
                except OSError:
                    # Fall back to copying if hard link fails
                    shutil.copy2(source_partition_dir / "ids.npy", target_partition_dir / "ids.npy")
                    shutil.copy2(source_partition_dir / "embeddings.npy", target_partition_dir / "embeddings.npy")
                    logger.debug(f"Copied partition {partition_idx} from {source_partition_dir}")
                
                # Update metadata
                partition_info = {
                    "partition_id": partition_idx,
                    "partition_dir": f"partition_{partition_idx:04d}",
                    "num_embeddings": source_partition["num_embeddings"],
                    "start_idx": current_embedding_idx,
                    "end_idx": current_embedding_idx + source_partition["num_embeddings"],
                    "embedding_dim": 2048,
                    "total_size_bytes": source_partition["total_size_bytes"]
                }
                metadata["partitions"].append(partition_info)
                
                current_embedding_idx += source_partition["num_embeddings"]
                partition_idx += 1
        
        # Generate additional embeddings if needed
        if additional_embeddings > 0:
            logger.info(f"Generating {additional_embeddings:,} additional embeddings...")
            
            # Calculate how many embeddings per new partition
            remaining_partitions = total_partitions - partition_idx
            if remaining_partitions > 0:
                base_embeddings = additional_embeddings // remaining_partitions
                extra_embeddings = additional_embeddings % remaining_partitions
            else:
                # Need to create new partitions
                new_partitions_needed = (additional_embeddings + embeddings_per_partition - 1) // embeddings_per_partition
                remaining_partitions = new_partitions_needed
                base_embeddings = additional_embeddings // remaining_partitions
                extra_embeddings = additional_embeddings % remaining_partitions
            
            generated_count = 0
            for i in range(remaining_partitions):
                partition_size = base_embeddings
                if i < extra_embeddings:
                    partition_size += 1
                
                if partition_size == 0:
                    break
                
                partition_dir = target_dir / f"partition_{partition_idx:04d}"
                partition_dir.mkdir(exist_ok=True)
                
                # Generate data
                ids = np.arange(current_embedding_idx, current_embedding_idx + partition_size, dtype=np.int64)
                embeddings = np.random.randn(partition_size, 2048).astype(np.float32)
                
                # Normalize
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / norms
                
                # Save
                np.save(partition_dir / "ids.npy", ids)
                np.save(partition_dir / "embeddings.npy", embeddings)
                
                # Update metadata
                partition_info = {
                    "partition_id": partition_idx,
                    "partition_dir": f"partition_{partition_idx:04d}",
                    "num_embeddings": partition_size,
                    "start_idx": current_embedding_idx,
                    "end_idx": current_embedding_idx + partition_size,
                    "embedding_dim": 2048,
                    "total_size_bytes": embeddings.nbytes + ids.nbytes
                }
                metadata["partitions"].append(partition_info)
                
                current_embedding_idx += partition_size
                partition_idx += 1
                generated_count += partition_size
                
                if (partition_idx % 10) == 0:
                    logger.info(f"Generated {generated_count:,}/{additional_embeddings:,} additional embeddings...")
        
        # Update final metadata
        metadata["num_partitions"] = len(metadata["partitions"])
        
        # Save metadata
        with open(target_dir / "embeddings_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Merged embeddings saved to: {target_dir}")
        logger.info(f"Total partitions: {metadata['num_partitions']}, Total embeddings: {current_embedding_idx:,}")
        
        return target_dir

    def _generate_or_use_cached_embeddings(self, num_embeddings: int) -> Path:
        """Generate synthetic embeddings or use cached ones with smart combination"""
        cache_base = Path("/tmp/milvus_embeddings_cache")
        cache_dir = cache_base / f"embeddings_{num_embeddings}"
        
        # First check for exact match
        if cache_dir.exists() and (cache_dir / "embeddings_metadata.json").exists():
            logger.info(f"Using cached embeddings from: {cache_dir}")
            return cache_dir
        
        # Try to find optimal combination of existing embeddings
        logger.info(f"No exact match for {num_embeddings:,} embeddings, checking for combinations...")
        existing_dirs, remaining = self._find_optimal_embedding_combination(num_embeddings, cache_base)
        
        if existing_dirs:
            logger.info(f"Found combination: {len(existing_dirs)} existing directories + {remaining:,} new embeddings")
            # Use merge approach
            return self._merge_existing_embeddings(existing_dirs, remaining, cache_dir, num_embeddings)
        
        # No existing embeddings found, generate all from scratch
        logger.info(f"No existing embeddings found, generating {num_embeddings:,} new synthetic embeddings...")
        
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate partitions
        embeddings_per_partition = 100_000
        num_partitions = max(10, (num_embeddings + embeddings_per_partition - 1) // embeddings_per_partition)
        
        if num_partitions > 200:
            num_partitions = 200
            embeddings_per_partition = num_embeddings // num_partitions
        
        remaining = num_embeddings % num_partitions
        
        logger.info(f"Creating {num_partitions} partitions with ~{embeddings_per_partition:,} embeddings each")
        
        metadata = {
            "format_version": "2.0",
            "storage_format": "partitioned_aggregated",
            "num_vectors": num_embeddings,
            "embedding_dim": 2048,
            "num_partitions": num_partitions,
            "partitions": []
        }
        
        current_idx = 0
        for i in range(num_partitions):
            partition_size = embeddings_per_partition
            if i < remaining:
                partition_size += 1
            
            partition_dir = cache_dir / f"partition_{i:04d}"
            partition_dir.mkdir(exist_ok=True)
            
            # Generate data
            ids = np.arange(current_idx, current_idx + partition_size, dtype=np.int64)
            embeddings = np.random.randn(partition_size, 2048).astype(np.float32)
            
            # Normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
            
            # Save
            np.save(partition_dir / "ids.npy", ids)
            np.save(partition_dir / "embeddings.npy", embeddings)
            
            # Partition info
            partition_info = {
                "partition_id": i,
                "partition_dir": f"partition_{i:04d}",
                "num_embeddings": partition_size,
                "start_idx": current_idx,
                "end_idx": current_idx + partition_size,
                "embedding_dim": 2048,
                "total_size_bytes": embeddings.nbytes + ids.nbytes
            }
            metadata["partitions"].append(partition_info)
            
            current_idx += partition_size
            
            if (i + 1) % 10 == 0:
                logger.info(f"Generated {i + 1}/{num_partitions} partitions...")
        
        # Save metadata
        with open(cache_dir / "embeddings_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Generated and cached embeddings at: {cache_dir}")
        return cache_dir
    
    def _save_raw_data(self, num_embeddings: int, storage_mode: str, result: Dict):
        """Save raw result data"""
        filename = self.run_dir / "raw_data" / f"result_{num_embeddings}_{storage_mode}.json"
        
        # Convert dataclasses to dicts
        result_copy = result.copy()
        if 'bulk_insert_metrics' in result_copy:
            metrics = result_copy['bulk_insert_metrics']
            result_copy['bulk_insert_metrics'] = {
                'duration': metrics.duration,
                'avg_embeddings_per_second': metrics.avg_embeddings_per_second,
                'avg_megabytes_per_second': metrics.avg_megabytes_per_second,
                'peak_embeddings_per_second': metrics.peak_embeddings_per_second,
                'peak_megabytes_per_second': metrics.peak_megabytes_per_second,
                'time_to_first_completion': metrics.time_to_first_completion,
                'num_jobs': metrics.num_jobs,
                'num_segments': metrics.num_segments,
                'total_embeddings': metrics.total_embeddings,
                'total_bytes': metrics.total_bytes,
                'throughput_samples': [asdict(s) for s in metrics.throughput_samples]
            }
        
        with open(filename, 'w') as f:
            json.dump(result_copy, f, indent=2)
        
        logger.info(f"Saved raw data to: {filename}")
    
    def generate_visualizations(self, results: List[Dict]):
        """Generate throughput visualizations"""
        if not results:
            logger.warning("No results to visualize")
            return
        
        # Create summary plot
        self._create_summary_plot(results)
        
        # Create per-dataset throughput timeline plots
        for result in results:
            self._create_throughput_timeline_plot(result)
        
        # Create storage comparison if both modes tested
        self._create_storage_comparison(results)
        
        logger.info(f"Visualizations saved to: {self.run_dir / 'visualizations'}")
    
    def _create_summary_plot(self, results: List[Dict]):
        """Create summary plot of throughput metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Milvus Bulk Insert Throughput Analysis', fontsize=16)
        
        # Group by storage mode
        infinia_results = [r for r in results if r['storage_mode'] == 'infinia']
        gcs_results = [r for r in results if r['storage_mode'] == 'gcs']
        
        # Plot 1: Average Embeddings/sec vs Dataset Size
        ax = axes[0, 0]
        if infinia_results:
            sizes = [r['num_embeddings_requested'] / 1_000_000 for r in infinia_results]
            throughputs = []
            for r in infinia_results:
                metrics = r['bulk_insert_metrics']
                if isinstance(metrics, dict):
                    throughputs.append(metrics['avg_embeddings_per_second'])
                else:
                    throughputs.append(metrics.avg_embeddings_per_second)
            ax.plot(sizes, throughputs, 'o-', label='Infinia', linewidth=2, markersize=8)
        
        if gcs_results:
            sizes = [r['num_embeddings_requested'] / 1_000_000 for r in gcs_results]
            throughputs = []
            for r in gcs_results:
                metrics = r['bulk_insert_metrics']
                if isinstance(metrics, dict):
                    throughputs.append(metrics['avg_embeddings_per_second'])
                else:
                    throughputs.append(metrics.avg_embeddings_per_second)
            ax.plot(sizes, throughputs, 's-', label='GCS', linewidth=2, markersize=8)
        
        ax.set_xlabel('Dataset Size (M embeddings)')
        ax.set_ylabel('Embeddings/second')
        ax.set_title('Average Throughput (Embeddings/sec)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Average MB/sec vs Dataset Size
        ax = axes[0, 1]
        if infinia_results:
            sizes = [r['num_embeddings_requested'] / 1_000_000 for r in infinia_results]
            throughputs = []
            for r in infinia_results:
                metrics = r['bulk_insert_metrics']
                if isinstance(metrics, dict):
                    throughputs.append(metrics['avg_megabytes_per_second'])
                else:
                    throughputs.append(metrics.avg_megabytes_per_second)
            ax.plot(sizes, throughputs, 'o-', label='Infinia', linewidth=2, markersize=8)
        
        if gcs_results:
            sizes = [r['num_embeddings_requested'] / 1_000_000 for r in gcs_results]
            throughputs = []
            for r in gcs_results:
                metrics = r['bulk_insert_metrics']
                if isinstance(metrics, dict):
                    throughputs.append(metrics['avg_megabytes_per_second'])
                else:
                    throughputs.append(metrics.avg_megabytes_per_second)
            ax.plot(sizes, throughputs, 's-', label='GCS', linewidth=2, markersize=8)
        
        ax.set_xlabel('Dataset Size (M embeddings)')
        ax.set_ylabel('MB/second')
        ax.set_title('Average Throughput (MB/sec)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Time to First Completion
        ax = axes[1, 0]
        if infinia_results:
            sizes = [r['num_embeddings_requested'] / 1_000_000 for r in infinia_results]
            times = []
            for r in infinia_results:
                metrics = r['bulk_insert_metrics']
                if isinstance(metrics, dict):
                    times.append(metrics['time_to_first_completion'])
                else:
                    times.append(metrics.time_to_first_completion)
            ax.plot(sizes, times, 'o-', label='Infinia', linewidth=2, markersize=8)
        
        if gcs_results:
            sizes = [r['num_embeddings_requested'] / 1_000_000 for r in gcs_results]
            times = []
            for r in gcs_results:
                metrics = r['bulk_insert_metrics']
                if isinstance(metrics, dict):
                    times.append(metrics['time_to_first_completion'])
                else:
                    times.append(metrics.time_to_first_completion)
            ax.plot(sizes, times, 's-', label='GCS', linewidth=2, markersize=8)
        
        ax.axhline(y=120, color='red', linestyle='--', alpha=0.5, label='120s threshold')
        ax.set_xlabel('Dataset Size (M embeddings)')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Time to First Job Completion')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Peak vs Average Throughput
        ax = axes[1, 1]
        for result in results:
            metrics = result['bulk_insert_metrics']
            size = result['num_embeddings_requested'] / 1_000_000
            if isinstance(metrics, dict):
                avg_throughput = metrics['avg_embeddings_per_second'] / 1000  # K/s
                peak_throughput = metrics['peak_embeddings_per_second'] / 1000  # K/s
            else:
                avg_throughput = metrics.avg_embeddings_per_second / 1000  # K/s
                peak_throughput = metrics.peak_embeddings_per_second / 1000  # K/s
            
            color = 'blue' if result['storage_mode'] == 'infinia' else 'green'
            marker = 'o' if result['storage_mode'] == 'infinia' else 's'
            
            ax.scatter(avg_throughput, peak_throughput, s=100, c=color, marker=marker, alpha=0.7)
            ax.annotate(f"{size:.0f}M", (avg_throughput, peak_throughput), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Add diagonal line
        max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Peak = Avg')
        
        ax.set_xlabel('Average Throughput (K embeddings/sec)')
        ax.set_ylabel('Peak Throughput (K embeddings/sec)')
        ax.set_title('Peak vs Average Throughput')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.run_dir / 'visualizations' / 'throughput_summary.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_throughput_timeline_plot(self, result: Dict):
        """Create timeline plot of throughput for a specific dataset"""
        metrics = result['bulk_insert_metrics']
        # Handle both dict and object access
        if isinstance(metrics, dict):
            samples = metrics['throughput_samples']
        else:
            samples = metrics.throughput_samples
        
        if not samples:
            return
        
        num_embeddings = result['num_embeddings_requested']
        storage_mode = result['storage_mode']
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f'Throughput Timeline - {num_embeddings / 1_000_000:.1f}M Embeddings ({storage_mode})', 
                    fontsize=14)
        
        # Extract timeline data - handle both dict and object access
        if samples and isinstance(samples[0], dict):
            timestamps = [s['timestamp'] for s in samples]
            embeddings_per_sec = [s['embeddings_per_second'] for s in samples]
            mb_per_sec = [s['megabytes_per_second'] for s in samples]
            active_jobs = [s['active_jobs'] for s in samples]
            completed_jobs = [s['completed_jobs'] for s in samples]
        else:
            timestamps = [s.timestamp for s in samples]
            embeddings_per_sec = [s.embeddings_per_second for s in samples]
            mb_per_sec = [s.megabytes_per_second for s in samples]
            active_jobs = [s.active_jobs for s in samples]
            completed_jobs = [s.completed_jobs for s in samples]
        
        # Plot 1: Embeddings per second
        ax1.plot(timestamps, embeddings_per_sec, linewidth=2, color='blue')
        ax1.fill_between(timestamps, embeddings_per_sec, alpha=0.3, color='blue')
        # Handle both dict and object access
        if isinstance(metrics, dict):
            avg_emb = metrics['avg_embeddings_per_second']
        else:
            avg_emb = metrics.avg_embeddings_per_second
        ax1.axhline(y=avg_emb, color='red', linestyle='--', 
                   label=f'Average: {avg_emb:.0f}')
        ax1.set_ylabel('Embeddings/second')
        ax1.set_title('Throughput (Embeddings/sec)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: MB per second
        ax2.plot(timestamps, mb_per_sec, linewidth=2, color='green')
        ax2.fill_between(timestamps, mb_per_sec, alpha=0.3, color='green')
        # Handle both dict and object access
        if isinstance(metrics, dict):
            avg_mb = metrics['avg_megabytes_per_second']
        else:
            avg_mb = metrics.avg_megabytes_per_second
        ax2.axhline(y=avg_mb, color='red', linestyle='--',
                   label=f'Average: {avg_mb:.1f} MB/s')
        ax2.set_ylabel('MB/second')
        ax2.set_title('Throughput (MB/sec)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Active and completed jobs
        ax3.plot(timestamps, active_jobs, linewidth=2, color='orange', label='Active Jobs')
        ax3.plot(timestamps, completed_jobs, linewidth=2, color='darkgreen', label='Completed Jobs')
        ax3.fill_between(timestamps, active_jobs, alpha=0.3, color='orange')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Number of Jobs')
        ax3.set_title('Job Progress')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'throughput_timeline_{num_embeddings}_{storage_mode}.png'
        plt.savefig(self.run_dir / 'visualizations' / filename, 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_storage_comparison(self, results: List[Dict]):
        """Create comparison between storage backends"""
        # Group by dataset size
        sizes = sorted(set(r['num_embeddings_requested'] for r in results))
        
        if len(set(r['storage_mode'] for r in results)) < 2:
            return  # Need both storage modes for comparison
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data
        infinia_throughputs = []
        gcs_throughputs = []
        
        for size in sizes:
            infinia_result = next((r for r in results 
                                 if r['num_embeddings_requested'] == size 
                                 and r['storage_mode'] == 'infinia'), None)
            gcs_result = next((r for r in results 
                             if r['num_embeddings_requested'] == size 
                             and r['storage_mode'] == 'gcs'), None)
            
            if infinia_result:
                metrics = infinia_result['bulk_insert_metrics']
                if isinstance(metrics, dict):
                    infinia_throughputs.append(metrics['avg_megabytes_per_second'])
                else:
                    infinia_throughputs.append(metrics.avg_megabytes_per_second)
            else:
                infinia_throughputs.append(0)
                
            if gcs_result:
                metrics = gcs_result['bulk_insert_metrics']
                if isinstance(metrics, dict):
                    gcs_throughputs.append(metrics['avg_megabytes_per_second'])
                else:
                    gcs_throughputs.append(metrics.avg_megabytes_per_second)
            else:
                gcs_throughputs.append(0)
        
        # Create grouped bar chart
        x = np.arange(len(sizes))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, infinia_throughputs, width, label='Infinia', alpha=0.8)
        bars2 = ax.bar(x + width/2, gcs_throughputs, width, label='GCS', alpha=0.8)
        
        ax.set_xlabel('Dataset Size')
        ax.set_ylabel('Throughput (MB/second)')
        ax.set_title('Storage Backend Throughput Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s/1_000_000:.0f}M" for s in sizes])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.run_dir / 'visualizations' / 'storage_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()


def parse_dataset_size(size_str: str) -> int:
    """Parse dataset size string (e.g., '1M', '500K') to integer"""
    size_str = size_str.strip().upper()
    if size_str.endswith('M'):
        return int(float(size_str[:-1]) * 1_000_000)
    elif size_str.endswith('K'):
        return int(float(size_str[:-1]) * 1_000)
    else:
        return int(size_str)


def main():
    parser = argparse.ArgumentParser(
        description="Measure Milvus bulk insert throughput"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="1M,2M,5M,10M,20M",
        help="Comma-separated list of dataset sizes (e.g., 1M,2M,5M,10M,20M)"
    )
    parser.add_argument(
        "--collection-prefix",
        type=str,
        default="throughput_test",
        help="Prefix for collection names"
    )
    parser.add_argument(
        "--storage-mode",
        type=str,
        choices=["infinia", "gcs", "both"],
        default="infinia",
        help="Storage backend to use (or 'both' to test both)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./milvus_throughput_results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Parse dataset sizes
    dataset_sizes = [parse_dataset_size(s) for s in args.datasets.split(',')]
    
    # Determine storage modes to test
    if args.storage_mode == "both":
        storage_modes = ["infinia", "gcs"]
    else:
        storage_modes = [args.storage_mode]
    
    logger.info(f"Starting Milvus throughput analysis")
    logger.info(f"Dataset sizes: {[f'{s:,}' for s in dataset_sizes]}")
    logger.info(f"Storage modes: {storage_modes}")
    
    # Create analyzer
    analyzer = MilvusThroughputAnalyzer(args.output_dir)
    
    # Analyze each combination
    results = []
    for storage_mode in storage_modes:
        for num_embeddings in dataset_sizes:
            try:
                result = analyzer.analyze_dataset(
                    num_embeddings=num_embeddings,
                    collection_prefix=args.collection_prefix,
                    storage_mode=storage_mode
                )
                results.append(result)
                analyzer.results.append(result)
            except Exception as e:
                logger.error(f"Failed to analyze {num_embeddings} embeddings on {storage_mode}: {e}")
                continue
    
    # Generate visualizations
    if results:
        logger.info("Generating visualizations...")
        analyzer.generate_visualizations(results)
        
        # Save summary
        summary_path = analyzer.run_dir / "summary.json"
        with open(summary_path, 'w') as f:
            summary = {
                "run_id": analyzer.run_id,
                "datasets_analyzed": len(results),
                "dataset_sizes": dataset_sizes,
                "storage_modes": storage_modes,
                "collection_prefix": args.collection_prefix,
                "timestamp": datetime.now().isoformat()
            }
            json.dump(summary, f, indent=2)
        
        logger.info(f"Analysis complete! Results saved to: {analyzer.run_dir}")
        
        # Print summary table
        logger.info("\n" + "="*100)
        logger.info("THROUGHPUT SUMMARY")
        logger.info("="*100)
        logger.info(f"{'Dataset':<10} {'Storage':<10} {'Avg (emb/s)':<15} {'Avg (MB/s)':<12} "
                   f"{'Peak (emb/s)':<15} {'Peak (MB/s)':<12} {'Duration':<10}")
        logger.info("-"*100)
        
        for result in results:
            metrics = result['bulk_insert_metrics']
            size_str = f"{result['num_embeddings_requested'] / 1_000_000:.0f}M"
            storage = result['storage_mode']
            
            # Handle both dict and object access
            if isinstance(metrics, dict):
                avg_emb = f"{metrics['avg_embeddings_per_second']:.0f}"
                avg_mb = f"{metrics['avg_megabytes_per_second']:.1f}"
                peak_emb = f"{metrics['peak_embeddings_per_second']:.0f}"
                peak_mb = f"{metrics['peak_megabytes_per_second']:.1f}"
                duration = f"{metrics['duration']:.1f}s"
            else:
                avg_emb = f"{metrics.avg_embeddings_per_second:.0f}"
                avg_mb = f"{metrics.avg_megabytes_per_second:.1f}"
                peak_emb = f"{metrics.peak_embeddings_per_second:.0f}"
                peak_mb = f"{metrics.peak_megabytes_per_second:.1f}"
                duration = f"{metrics.duration:.1f}s"
            
            logger.info(f"{size_str:<10} {storage:<10} {avg_emb:<15} {avg_mb:<12} "
                       f"{peak_emb:<15} {peak_mb:<12} {duration:<10}")
        
        logger.info("="*100)
    else:
        logger.error("No successful results to analyze")


if __name__ == "__main__":
    main()