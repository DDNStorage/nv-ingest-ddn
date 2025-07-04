"""
Parallel Upload Manager for Milvus Bulk Indexing
Manages concurrent uploads of segment files to storage
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple

from tqdm import tqdm

from .storage_clients import StorageClient

logger = logging.getLogger(__name__)


class ParallelUploadManager:
    """Manage parallel uploads of segments to storage"""
    
    def __init__(self, storage_client: StorageClient, num_workers: int = 8):
        """
        Initialize upload manager
        
        Args:
            storage_client: Storage client instance
            num_workers: Number of parallel upload workers
        """
        self.storage_client = storage_client
        self.num_workers = num_workers
        logger.info(f"Initialized parallel upload manager with {num_workers} workers")
    
    def upload_segment(self, segment_id: str, file_paths: Dict[str, str], 
                      collection_name: str) -> Dict:
        """
        Upload a single segment to storage
        
        Args:
            segment_id: Unique segment identifier
            file_paths: Dict mapping field names to local file paths
            collection_name: Name of the Milvus collection
            
        Returns:
            Dict with upload results
        """
        try:
            start_time = time.time()
            
            # Upload all files and track remote paths
            remote_paths = []
            urls = {}
            
            for field_name, local_path in file_paths.items():
                remote_path = f"milvus_segments/{collection_name}/{segment_id}/{field_name}.npy"
                url = self.storage_client.upload_file(local_path, remote_path)
                urls[field_name] = url
                remote_paths.append(remote_path)
            
            upload_time = time.time() - start_time
            
            result = {
                'segment_id': segment_id,
                'urls': urls,
                'remote_paths': remote_paths,
                'upload_time': upload_time,
                'status': 'success'
            }
            
            logger.info(f"Uploaded segment {segment_id} in {upload_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Failed to upload segment {segment_id}: {e}")
            return {
                'segment_id': segment_id,
                'status': 'failed',
                'error': str(e)
            }
    
    def upload_all_segments(self, segments_to_upload: List[Tuple[str, Dict[str, str]]], 
                           collection_name: str) -> List[Dict]:
        """
        Upload all segments in parallel
        
        Args:
            segments_to_upload: List of (segment_id, file_paths_dict) tuples
            collection_name: Name of the Milvus collection
            
        Returns:
            List of upload results
        """
        logger.info(f"Starting parallel upload of {len(segments_to_upload)} segments")
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all upload tasks
            futures = []
            for segment_id, file_paths in segments_to_upload:
                future = executor.submit(
                    self.upload_segment, 
                    segment_id, 
                    file_paths,
                    collection_name
                )
                futures.append(future)
            
            # Collect results with progress bar
            results = []
            with tqdm(total=len(futures), desc="Uploading segments") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
                    
                    if result['status'] == 'success':
                        pbar.set_postfix({'last_time': f"{result['upload_time']:.1f}s"})
        
        # Summary
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = sum(1 for r in results if r['status'] == 'failed')
        total_time = sum(r.get('upload_time', 0) for r in results if r['status'] == 'success')
        
        logger.info(f"Upload complete: {successful}/{len(results)} segments uploaded successfully")
        if failed > 0:
            logger.warning(f"{failed} segments failed to upload")
            for r in results:
                if r['status'] == 'failed':
                    logger.error(f"  - {r['segment_id']}: {r.get('error', 'Unknown error')}")
        
        if successful > 0:
            avg_time = total_time / successful
            logger.info(f"Average upload time per segment: {avg_time:.2f}s")
        
        return results