#!/usr/bin/env python3
"""
Storage Configuration Module
Handles different storage backends (Infinia, GCS, MinIO)
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path


class StorageConfig:
    """Manages storage backend configurations"""
    
    @staticmethod
    def load_env_file(env_file: Optional[str] = None) -> None:
        """Load environment variables from a specific .env file"""
        if env_file and Path(env_file).exists():
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        # Handle variable expansion like ${VAR}
                        if not value.startswith('${'):
                            os.environ[key.strip()] = value.strip()
    
    @staticmethod
    def get_storage_config() -> Dict[str, Any]:
        """Get storage configuration from environment variables"""
        storage_type = os.getenv('STORAGE_TYPE', 'minio').lower()
        
        if storage_type == 'infinia':
            return StorageConfig._get_infinia_config()
        elif storage_type == 'gcs':
            return StorageConfig._get_gcs_config()
        else:  # Default to minio
            return StorageConfig._get_minio_config()
    
    @staticmethod
    def _get_infinia_config() -> Dict[str, Any]:
        """Get Infinia storage configuration"""
        return {
            'type': 'infinia',
            'endpoint': os.getenv('STORAGE_ENDPOINT', '10.168.15.234:8111'),
            'bucket': os.getenv('STORAGE_BUCKET', 'milvus-db'),
            'access_key': os.getenv('STORAGE_ACCESS_KEY', ''),
            'secret_key': os.getenv('STORAGE_SECRET_KEY', ''),
            'use_ssl': os.getenv('USE_SSL', 'true').lower() == 'true',
            'ssl_cert_path': '/milvus/certs/red_ca_8A5EC7E9-B579-50C7-FA0D-197589C7D6A9.crt',
            'region': os.getenv('STORAGE_REGION', ''),
            'cloud_provider': 'aws',  # S3-compatible
            'root_path': os.getenv('STORAGE_ROOT_PATH', 'bulk-load')
        }
    
    @staticmethod
    def _get_gcs_config() -> Dict[str, Any]:
        """Get Google Cloud Storage configuration"""
        # GCS can use HMAC keys for S3-compatible access
        # Handle endpoint and port separately to match working setup
        endpoint = os.getenv('STORAGE_ENDPOINT', 'storage.googleapis.com')
        port = os.getenv('STORAGE_PORT', '443')
        
        return {
            'type': 'gcs',
            'endpoint': f"{endpoint}:{port}",
            'bucket': os.getenv('STORAGE_BUCKET', 'infinia-multimodal-milvus'),
            'access_key': os.getenv('STORAGE_ACCESS_KEY', ''),
            'secret_key': os.getenv('STORAGE_SECRET_KEY', ''),
            'use_ssl': os.getenv('USE_SSL', 'true').lower() == 'true',
            'cloud_provider': 'gcp',
            'use_iam': os.getenv('USE_IAM', 'false').lower() == 'true',
            'credentials_json': os.getenv('GCS_CREDENTIALS_JSON', ''),
            'root_path': os.getenv('STORAGE_ROOT_PATH', 'bulk-load')
        }
    
    @staticmethod
    def _get_minio_config() -> Dict[str, Any]:
        """Get MinIO storage configuration"""
        return {
            'type': 'minio',
            'endpoint': os.getenv('STORAGE_ENDPOINT', 'minio:9000'),
            'bucket': os.getenv('STORAGE_BUCKET', 'milvus-bucket'),
            'access_key': os.getenv('STORAGE_ACCESS_KEY', 'minioadmin'),
            'secret_key': os.getenv('STORAGE_SECRET_KEY', 'minioadmin'),
            'use_ssl': os.getenv('USE_SSL', 'false').lower() == 'true',
            'cloud_provider': 'aws',
            'root_path': os.getenv('STORAGE_ROOT_PATH', 'bulk-load')
        }
    
    @staticmethod
    def get_docker_volumes(storage_type: str) -> list:
        """Get Docker volume mounts based on storage type"""
        volumes = [
            './configs/milvus.yaml:/milvus/configs/milvus.yaml',
            '/var/lib/milvus/data:/var/lib/milvus'
        ]
        
        if storage_type == 'infinia':
            volumes.append('./certs:/milvus/certs:ro')
        
        return volumes
    
    @staticmethod
    def get_environment_vars(storage_config: Dict[str, Any]) -> Dict[str, str]:
        """Get environment variables for Milvus based on storage config"""
        env_vars = {
            'ETCD_ENDPOINTS': 'etcd:2379',
            'MINIO_ADDRESS': storage_config.get('endpoint', 'minio:9000'),
            'MINIO_BUCKET_NAME': storage_config.get('bucket', 'milvus-bucket'),
            'MINIO_ROOT_PATH': storage_config.get('root_path', 'bulk-load')
        }
        
        if storage_config['type'] in ['infinia', 'minio']:
            env_vars.update({
                'MINIO_ACCESS_KEY_ID': storage_config.get('access_key', ''),
                'MINIO_SECRET_ACCESS_KEY': storage_config.get('secret_key', ''),
                'MINIO_USE_SSL': str(storage_config.get('use_ssl', False)).lower()
            })
        
        if storage_config['type'] == 'infinia' and storage_config.get('use_ssl'):
            env_vars['MINIO_SSL_CACERT_FILE'] = storage_config.get('ssl_cert_path', '')
        
        if storage_config['type'] == 'gcs':
            env_vars.update({
                'MINIO_ACCESS_KEY_ID': storage_config.get('access_key', ''),
                'MINIO_SECRET_ACCESS_KEY': storage_config.get('secret_key', ''),
                'MINIO_USE_SSL': str(storage_config.get('use_ssl', True)).lower(),
                'MINIO_CLOUD_PROVIDER': 'gcp',
                'MINIO_USE_IAM': str(storage_config.get('use_iam', False)).lower()
            })
            if storage_config.get('credentials_json'):
                env_vars['MINIO_GCP_CREDENTIAL_JSON'] = storage_config['credentials_json']
        
        return env_vars