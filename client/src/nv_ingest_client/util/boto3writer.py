import sys
import time
import uuid
import threading
import logging
import io
import requests
from pathlib import Path
from threading import Lock, Thread
from typing import Optional, Callable, Any

import boto3
from botocore.config import Config

from pymilvus.bulk_writer import LocalBulkWriter, BulkFileType
from pymilvus.orm.schema import CollectionSchema
from pymilvus.exceptions import MilvusException

logger = logging.getLogger("bulk_writer")

# Default size constants from pymilvus
MB = 1024 * 1024

class Boto3BulkWriter(LocalBulkWriter):
    """
    A wrapper around pymilvus.bulk_writer.LocalBulkWriter that uses boto3 for S3 operations
    instead of Minio when committing files to remote storage.
    
    This implementation supports using presigned URLs for uploads, which proved to work
    better with certain S3-compatible storage backends.
    """
    
    class S3ConnectParam:
        """Connection parameters for boto3 S3 client"""
        def __init__(
            self,
            bucket_name: str,
            endpoint_url: Optional[str] = None,
            access_key: Optional[str] = None,
            secret_key: Optional[str] = None,
            region_name: Optional[str] = None,
            verify: bool = False,  
            use_presigned: bool = False,  
            **kwargs
        ):
            self.bucket_name = bucket_name
            self.endpoint_url = endpoint_url
            self.access_key = access_key
            self.secret_key = secret_key
            self.region_name = region_name
            self.verify = verify
            self.use_presigned = use_presigned
            self.kwargs = kwargs

    def __init__(
        self,
        schema: CollectionSchema,
        remote_path: str,
        connect_param: S3ConnectParam,
        chunk_size: int = 1024 * MB,
        file_type: BulkFileType = BulkFileType.PARQUET,
        config: Optional[dict] = None,
        **kwargs,
    ):
        # Initialize using parent class (LocalBulkWriter)
        local_path = Path(sys.argv[0]).resolve().parent.joinpath("bulk_writer")
        super().__init__(schema, str(local_path), chunk_size, file_type, config, **kwargs)
        
        # Set up S3-specific attributes
        self._remote_path = Path("/").joinpath(remote_path).joinpath(super().uuid)
        self._connect_param = connect_param
        self._client = None
        self._get_client()
        self._remote_files = []
        logger.info(f"Boto3 S3 buffer writer initialized, target path: {self._remote_path}")
        if self._connect_param.use_presigned:
            logger.info("Using presigned URL for uploads (recommended for GCP storage)")

    def __enter__(self):
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object):
        super().__exit__(exc_type, exc_val, exc_tb)
        if Path(self._local_path).parent.exists() and not any(
            Path(self._local_path).parent.iterdir()
        ):
            Path(self._local_path).parent.rmdir()
            logger.info(f"Delete empty directory '{Path(self._local_path).parent}'")

    def _get_client(self):
        """Initialize boto3 S3 client"""
        if self._client is not None:
            return self._client

        try:

            config = Config(
                retries={"max_attempts": 10, "mode": "standard"},
                connect_timeout=60,
                read_timeout=60,
            )
            

            self._client = boto3.Session().client(
                's3',
                endpoint_url=self._connect_param.endpoint_url,
                aws_access_key_id=self._connect_param.access_key,
                aws_secret_access_key=self._connect_param.secret_key,
                region_name=self._connect_param.region_name,
                verify=self._connect_param.verify,
                config=config
            )
            logger.info("boto3 S3 client successfully initialized")
        except Exception as err:
            logger.error(f"Failed to connect to S3 using boto3, error: {err}")
            raise

        return self._client

    def _upload_with_presigned_url(self, file_path, remote_file):
        """Upload a file using presigned URL method (better for GCP)"""
        try:
            presigned_url = self._client.generate_presigned_url(
                'put_object',
                Params={
                    'Bucket': self._connect_param.bucket_name,
                    'Key': remote_file,
                    'ContentType': 'application/octet-stream'
                },
                ExpiresIn=3600  
            )
            

            with open(file_path, 'rb') as f:
                file_data = f.read()
                
                response = requests.put(
                    presigned_url,
                    data=file_data,
                    headers={'Content-Type': 'application/octet-stream'},
                    verify=False
                )
                
                if response.status_code == 200:
                    logger.info(f"Successfully uploaded {file_path} to {remote_file} using presigned URL")
                    return True
                else:
                    logger.error(f"Upload failed with status {response.status_code}: {response.text[:100]}")
                    return False
        except Exception as e:
            logger.error(f"Failed to upload {file_path} using presigned URL: {e}")
            return False

    def _upload_with_standard_method(self, file_path, remote_file):
        """Upload a file using standard boto3 put_object method"""
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()
                self._client.put_object(
                    Bucket=self._connect_param.bucket_name,
                    Key=remote_file,
                    Body=file_data,
                    ContentType='application/octet-stream'
                )
            logger.info(f"Successfully uploaded {file_path} to {remote_file} using put_object")
            return True
        except Exception as e:
            logger.error(f"Failed to upload {file_path} using put_object: {e}")
            return False

    def _flush(self, call_back: Optional[Callable] = None):
        """Override _flush to use boto3 for S3 uploads with presigned URL support"""
        try:
            self._flush_count = self._flush_count + 1
            target_path = Path.joinpath(self._local_path, str(self._flush_count))

            old_buffer = super()._new_buffer()
            if old_buffer.row_count > 0:
                file_list = old_buffer.persist(
                    local_path=str(target_path),
                    buffer_size=self.buffer_size,
                    buffer_row_count=self.buffer_row_count,
                )
                
                remote_files = []
                for file_path in file_list:
                    local_file = Path(file_path)
                    remote_path_str = str(self._remote_path)
                    if remote_path_str.startswith('/'):
                        remote_path_str = remote_path_str[1:] 
                    
                    remote_file = f"{remote_path_str}/{local_file.name}"
                    
                    upload_success = False
                    if self._connect_param.use_presigned:
                        upload_success = self._upload_with_presigned_url(file_path, remote_file)
                    else:
                        upload_success = self._upload_with_standard_method(file_path, remote_file)
                    
                    if upload_success:
                        remote_files.append(remote_file)
                    else:
                        raise MilvusException(message=f"Failed to upload {file_path} to S3")
                
                self._remote_files.extend(remote_files)
                self._local_files.append(remote_files)  
                
                if call_back:
                    call_back(remote_files) 
        except Exception as e:
            logger.error(f"Failed to flush, error: {e}")
            raise e from e
        finally:
            del self._working_thread[threading.current_thread().name]
            logger.info(f"Flush thread finished, name: {threading.current_thread().name}")

    @property
    def batch_files(self):
        """Return list of file batches that have been uploaded"""
        return self._local_files
