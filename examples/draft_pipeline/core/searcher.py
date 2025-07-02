"""
Embedding Searcher for querying indexed embeddings
"""

import logging
from typing import List, Dict, Optional

import numpy as np
from pymilvus import Collection, MilvusClient, connections

try:
    from nv_ingest_client.util.milvus import nvingest_retrieval
    NV_INGEST_AVAILABLE = True
except ImportError:
    NV_INGEST_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("nv_ingest_client not available. Text search will be limited.")

logger = logging.getLogger(__name__)


class EmbeddingSearcher:
    """Search indexed embeddings in Milvus"""
    
    def __init__(
        self,
        collection_name: str,
        milvus_host: str = "localhost",
        milvus_port: int = 19530,
        embedding_dim: int = 2048
    ):
        self.collection_name = collection_name
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.embedding_dim = embedding_dim
        
        self.client = None
        self.collection = None
        self.embedder = None
    
    def connect(self):
        """Connect to Milvus and load collection"""
        try:
            connections.connect("default", host=self.milvus_host, port=self.milvus_port)
            self.client = MilvusClient(uri=f"http://{self.milvus_host}:{self.milvus_port}")
            self.collection = Collection(self.collection_name)
            self.collection.load()
            
            logger.info(f"Connected to collection: {self.collection_name}")
            logger.info(f"Collection entities: {self.collection.num_entities:,}")
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            raise
    
    def search_by_embedding(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5,
        output_fields: Optional[List[str]] = None
    ) -> List[Dict]:
        """Search using an embedding vector"""
        if output_fields is None:
            output_fields = ["text", "source", "chunk_index", "page_number"]
        
        try:
            # Ensure proper shape
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            search_params = {
                "metric_type": "L2",
                "params": {"ef": 150}
            }
            
            results = self.client.search(
                collection_name=self.collection_name,
                data=query_embedding.tolist(),
                limit=top_k,
                output_fields=output_fields,
                search_params=search_params
            )
            
            return results[0] if results else []
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def search_by_text(
        self, 
        query_text: str, 
        top_k: int = 5,
        embedding_endpoint: str = "http://localhost:8012/v1",
        model_name: str = "nvidia/llama-3.2-nv-embedqa-1b-v2",
        gpu_search: bool = True
    ) -> List[Dict]:
        """Search using text query with nv-ingest retrieval"""
        if not NV_INGEST_AVAILABLE:
            logger.error("nv_ingest_client not available. Cannot perform text search.")
            return []
        
        try:
            # Use nvingest_retrieval for text search
            results = nvingest_retrieval(
                queries=[query_text],
                collection_name=self.collection_name,
                milvus_uri=f"http://{self.milvus_host}:{self.milvus_port}",
                hybrid=False,  # Dense search only
                embedding_endpoint=embedding_endpoint,
                model_name=model_name,
                top_k=top_k,
                gpu_search=gpu_search,
                output_fields=["text", "source", "content_metadata"]
            )
            
            # nvingest_retrieval returns nested list, get first query results
            return results[0] if results else []
            
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """Get collection statistics"""
        try:
            stats = {
                "collection_name": self.collection_name,
                "total_entities": self.collection.num_entities,
                "loaded": self.collection.is_loaded,
                "has_index": len(self.client.list_indexes(collection_name=self.collection_name)) > 0
            }
            
            # Get index info
            indexes = self.client.list_indexes(collection_name=self.collection_name)
            if indexes:
                index_info = self.client.describe_index(
                    collection_name=self.collection_name,
                    index_name=indexes[0]
                )
                stats["index_info"] = {
                    "type": index_info.get("index_type"),
                    "metric": index_info.get("metric_type"),
                    "state": index_info.get("state")
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def close(self):
        """Clean up connections"""
        try:
            if self.collection:
                self.collection.release()
            connections.disconnect("default")
        except Exception as e:
            logger.error(f"Error closing connection: {e}")