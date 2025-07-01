#!/usr/bin/env python3
"""
Example: Custom Milvus ingestion from serialized embeddings
This script shows how to load previously serialized embeddings and 
ingest them into Milvus with custom processing.
"""

import numpy as np
import json
from pathlib import Path
from pymilvus import (
    connections, 
    Collection, 
    FieldSchema, 
    CollectionSchema, 
    DataType,
    utility
)


def create_milvus_collection(collection_name, embedding_dim, recreate=False):
    """
    Create a Milvus collection with appropriate schema.
    
    Args:
        collection_name: Name of the collection
        embedding_dim: Dimension of embeddings
        recreate: Whether to recreate if exists
    """
    # Check if collection exists
    if utility.has_collection(collection_name):
        if recreate:
            collection = Collection(collection_name)
            collection.drop()
            print(f"Dropped existing collection: {collection_name}")
        else:
            print(f"Collection {collection_name} already exists")
            return Collection(collection_name)
    
    # Define schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="page_number", dtype=DataType.INT64),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim)
    ]
    
    schema = CollectionSchema(
        fields=fields,
        description="Document embeddings collection"
    )
    
    # Create collection
    collection = Collection(
        name=collection_name,
        schema=schema
    )
    
    print(f"Created collection: {collection_name}")
    return collection


def ingest_numpy_embeddings(embeddings_dir, collection_name, milvus_config, batch_size=1000):
    """
    Ingest numpy embeddings into Milvus.
    
    Args:
        embeddings_dir: Directory containing numpy embeddings
        collection_name: Target Milvus collection
        milvus_config: Milvus connection config
        batch_size: Batch size for insertion
    """
    # Connect to Milvus
    connections.connect(
        host=milvus_config["host"],
        port=milvus_config["port"]
    )
    
    embeddings_path = Path(embeddings_dir)
    
    # Load metadata
    metadata_file = embeddings_path / "embeddings_metadata.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    if not metadata:
        print("No embeddings found in metadata")
        return
    
    # Get embedding dimension from first file
    first_embedding = np.load(metadata[0]["filepath"])
    embedding_dim = first_embedding.shape[0]
    
    # Create collection
    collection = create_milvus_collection(collection_name, embedding_dim)
    
    # Prepare data for insertion
    total_count = len(metadata)
    inserted_count = 0
    
    for i in range(0, total_count, batch_size):
        batch_end = min(i + batch_size, total_count)
        batch_metadata = metadata[i:batch_end]
        
        # Load batch embeddings
        batch_embeddings = []
        batch_contents = []
        batch_sources = []
        batch_pages = []
        batch_chunks = []
        
        for meta in batch_metadata:
            # Load embedding
            embedding = np.load(meta["filepath"])
            batch_embeddings.append(embedding.tolist())
            
            # Prepare other fields
            batch_contents.append(meta.get("content", "")[:65535])  # Truncate if needed
            batch_sources.append(meta.get("source_file", "unknown")[:512])
            batch_pages.append(meta.get("page_number", -1) or -1)
            batch_chunks.append(meta.get("chunk_index", 0))
        
        # Insert batch
        entities = [
            batch_contents,      # content
            batch_sources,       # source_file
            batch_pages,         # page_number
            batch_chunks,        # chunk_index
            batch_embeddings     # embedding
        ]
        
        collection.insert(entities)
        inserted_count += len(batch_embeddings)
        
        print(f"Inserted batch {i//batch_size + 1}: {len(batch_embeddings)} embeddings")
    
    # Create index for better search performance
    print("Creating index...")
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index(
        field_name="embedding",
        index_params=index_params
    )
    
    # Load collection
    collection.load()
    
    print(f"\nIngestion complete!")
    print(f"Total embeddings inserted: {inserted_count}")
    print(f"Collection '{collection_name}' is ready for search")
    
    return collection


def ingest_batch_embeddings(batch_dir, collection_name, milvus_config):
    """
    Ingest batch-serialized embeddings into Milvus.
    
    Args:
        batch_dir: Directory containing batch numpy files
        collection_name: Target Milvus collection
        milvus_config: Milvus connection config
    """
    # Connect to Milvus
    connections.connect(
        host=milvus_config["host"],
        port=milvus_config["port"]
    )
    
    batch_path = Path(batch_dir)
    
    # Load batch info
    batch_info_file = batch_path / "batch_info.json"
    with open(batch_info_file, 'r') as f:
        batch_info = json.load(f)
    
    if not batch_info:
        print("No batch info found")
        return
    
    # Get embedding dimension
    first_batch = np.load(batch_info[0]["embeddings_file"])
    embedding_dim = first_batch.shape[1]
    
    # Create collection
    collection = create_milvus_collection(collection_name, embedding_dim)
    
    # Process each batch
    total_inserted = 0
    
    for batch in batch_info:
        # Load embeddings and metadata
        embeddings = np.load(batch["embeddings_file"])
        
        with open(batch["metadata_file"], 'r') as f:
            metadata = json.load(f)
        
        # Prepare data
        contents = [m.get("content", "")[:65535] for m in metadata]
        sources = [m.get("source_file", "unknown")[:512] for m in metadata]
        pages = [m.get("page_number", -1) or -1 for m in metadata]
        chunks = [m.get("chunk_index", 0) for m in metadata]
        
        # Insert
        entities = [
            contents,
            sources,
            pages,
            chunks,
            embeddings.tolist()
        ]
        
        collection.insert(entities)
        total_inserted += len(embeddings)
        
        print(f"Inserted batch {batch['batch_index']}: {len(embeddings)} embeddings")
    
    # Create index
    print("Creating index...")
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index(
        field_name="embedding",
        index_params=index_params
    )
    
    # Load collection
    collection.load()
    
    print(f"\nBatch ingestion complete!")
    print(f"Total embeddings inserted: {total_inserted}")
    
    return collection


def search_similar(collection, query_embedding, top_k=5):
    """
    Search for similar embeddings in Milvus.
    
    Args:
        collection: Milvus collection
        query_embedding: Query embedding vector
        top_k: Number of results to return
    """
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10}
    }
    
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["content", "source_file", "page_number"]
    )
    
    return results[0]


if __name__ == "__main__":
    # Milvus configuration
    milvus_config = {
        "host": "localhost",
        "port": 19530
    }
    
    # Example 1: Ingest individual numpy embeddings
    print("=== Ingesting individual embeddings ===")
    embeddings_dir = "./embeddings_individual"
    collection_name = "document_embeddings"
    
    if Path(embeddings_dir).exists():
        collection = ingest_numpy_embeddings(
            embeddings_dir,
            collection_name,
            milvus_config,
            batch_size=100
        )
    
    # Example 2: Ingest batch embeddings
    print("\n=== Ingesting batch embeddings ===")
    batch_dir = "./embeddings_batched"
    batch_collection_name = "document_embeddings_batch"
    
    if Path(batch_dir).exists():
        collection = ingest_batch_embeddings(
            batch_dir,
            batch_collection_name,
            milvus_config
        )
    
    # Example 3: Search for similar documents
    print("\n=== Example search ===")
    
    # Connect to Milvus
    connections.connect(
        host=milvus_config["host"],
        port=milvus_config["port"]
    )
    
    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
        collection.load()
        
        # Create a random query embedding (in practice, this would be from a real query)
        embedding_dim = 384  # Adjust based on your model
        query_embedding = np.random.randn(embedding_dim).astype(np.float32)
        
        # Search
        results = search_similar(collection, query_embedding.tolist(), top_k=5)
        
        print(f"\nTop {len(results)} similar documents:")
        for i, hit in enumerate(results):
            print(f"\n{i+1}. Distance: {hit.distance:.4f}")
            print(f"   Source: {hit.entity.get('source_file', 'N/A')}")
            print(f"   Page: {hit.entity.get('page_number', 'N/A')}")
            print(f"   Content: {hit.entity.get('content', '')[:200]}...")