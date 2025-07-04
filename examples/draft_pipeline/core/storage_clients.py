"""
Storage Client Factory for Milvus Bulk Indexing
Supports GCS and Infinia storage backends
"""

import os
import logging
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class StorageClient(ABC):
    """Abstract base class for storage clients"""
    
    @abstractmethod
    def upload_file(self, local_path: str, remote_path: str) -> str:
        """Upload file to storage and return remote URL"""
        pass
    
    @abstractmethod
    def list_files(self, prefix: str) -> List[str]:
        """List files with given prefix"""
        pass


class GCSStorageClient(StorageClient):
    """Google Cloud Storage client with default credentials support"""
    
    def __init__(self, bucket_name: str, project_id: Optional[str] = None):
        import google.auth
        from google.cloud import storage as gcs
        
        self.bucket_name = bucket_name
        
        try:
            # Check for service account key first
            if os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
                logger.info(f"Using service account from: {os.environ['GOOGLE_APPLICATION_CREDENTIALS']}")
                self.client = gcs.Client.from_service_account_json(
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'],
                    project=project_id
                )
            else:
                # Try default credentials with explicit scopes
                credentials, project = google.auth.default(
                    scopes=['https://www.googleapis.com/auth/cloud-platform',
                            'https://www.googleapis.com/auth/devstorage.full_control']
                )
                self.client = gcs.Client(
                    project=project_id or project,
                    credentials=credentials
                )
                logger.info(f"Using default credentials for project: {project_id or project}")
            
            self.bucket = self.client.bucket(self.bucket_name)
            
            # Test access
            self._test_bucket_access()
            
            logger.info(f"Initialized GCS client for bucket: {self.bucket_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize GCS client: {e}")
            logger.error("Try one of these solutions:")
            logger.error("1. Set GOOGLE_APPLICATION_CREDENTIALS to a service account key")
            logger.error("2. Run: gcloud auth application-default login --scopes=https://www.googleapis.com/auth/cloud-platform")
            raise
    
    def _test_bucket_access(self):
        """Test if we have write access to the bucket"""
        try:
            test_blob = self.bucket.blob('_test_access_check')
            test_blob.upload_from_string('test')
            test_blob.delete()
            logger.info("✓ Bucket write access verified")
        except Exception as e:
            logger.error(f"❌ Bucket write access test failed: {e}")
            if "403" in str(e) and "scope" in str(e).lower():
                logger.error("This is a scope issue. Please re-authenticate with:")
                logger.error("gcloud auth application-default login --scopes=https://www.googleapis.com/auth/cloud-platform")
            raise
    
    def upload_file(self, local_path: str, remote_path: str) -> str:
        """Upload file to GCS"""
        blob = self.bucket.blob(remote_path)
        blob.upload_from_filename(local_path)
        return f"gs://{self.bucket_name}/{remote_path}"
    
    def list_files(self, prefix: str) -> List[str]:
        """List files with given prefix"""
        return [blob.name for blob in self.bucket.list_blobs(prefix=prefix)]


class InfiniaStorageClient(StorageClient):
    """Infinia S3-compatible storage client"""
    
    def __init__(self, endpoint: str, access_key: str, secret_key: str, 
                 bucket_name: str, ca_cert_path: Optional[str] = None):
        import boto3
        from botocore.config import Config as BotoConfig
        
        self.bucket_name = bucket_name
        
        # Parse endpoint
        secure = True
        if endpoint.startswith('https://'):
            endpoint = endpoint.replace('https://', '')
            secure = True
        elif endpoint.startswith('http://'):
            endpoint = endpoint.replace('http://', '')
            secure = False
        
        self.endpoint = endpoint
        
        # Configure boto3
        session = boto3.Session()
        
        # SSL configuration
        verify = True
        if ca_cert_path and os.path.exists(ca_cert_path):
            verify = ca_cert_path
            os.environ['AWS_CA_BUNDLE'] = ca_cert_path
            logger.info(f"Using custom CA certificate: {ca_cert_path}")
        
        self.client = session.client(
            's3',
            endpoint_url=f"{'https' if secure else 'http'}://{endpoint}",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=BotoConfig(signature_version='s3v4'),
            verify=verify
        )
        
        logger.info(f"Initialized Infinia client for bucket: {bucket_name}")
    
    def upload_file(self, local_path: str, remote_path: str) -> str:
        """Upload file to Infinia Storage"""
        self.client.upload_file(local_path, self.bucket_name, remote_path)
        return f"s3://{self.bucket_name}/{remote_path}"
    
    def list_files(self, prefix: str) -> List[str]:
        """List files with given prefix"""
        response = self.client.list_objects_v2(
            Bucket=self.bucket_name,
            Prefix=prefix
        )
        return [obj['Key'] for obj in response.get('Contents', [])]


class StorageClientFactory:
    """Factory for creating storage clients"""
    
    @staticmethod
    def create_client(mode: str, config: Dict[str, Any]) -> StorageClient:
        """
        Create storage client based on mode
        
        Args:
            mode: Storage mode ('gcs' or 'infinia')
            config: Configuration dict with mode-specific parameters
            
        Returns:
            StorageClient instance
        """
        if mode == "gcs":
            return GCSStorageClient(
                bucket_name=config.get('bucket_name', 'infinia-multimodal-milvus'),
                project_id=config.get('project_id')
            )
        elif mode == "infinia":
            # Get from environment or config
            endpoint = config.get('endpoint', os.environ.get('MY_STORAGE_ENDPOINT', ''))
            access_key = config.get('access_key', os.environ.get('MY_ACCESS_KEY_ID', ''))
            secret_key = config.get('secret_key', os.environ.get('MY_SECRET_ACCESS_KEY', ''))
            bucket_name = config.get('bucket_name', os.environ.get('MY_BUCKET_NAME', 'milvus-db'))
            ca_cert = config.get('ca_cert', os.environ.get('MY_STORAGE_CA_CERT'))
            
            return InfiniaStorageClient(
                endpoint=endpoint,
                access_key=access_key,
                secret_key=secret_key,
                bucket_name=bucket_name,
                ca_cert_path=ca_cert
            )
        else:
            raise ValueError(f"Unknown storage mode: {mode}")