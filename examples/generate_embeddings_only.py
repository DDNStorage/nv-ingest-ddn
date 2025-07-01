#!/usr/bin/env python3
"""
Example: Generate embeddings without uploading to vector database
This script shows how to use nv-ingest to extract content and generate embeddings
without immediately ingesting them into a vector store.
"""

import json
from nv_ingest_client import Ingestor
from nv_ingest_client.primitives import JobSpec


def generate_embeddings_only(documents, config):
    """
    Generate embeddings for documents without uploading to vector database.
    
    Args:
        documents: List of document paths
        config: NV-Ingest configuration
        
    Returns:
        List of results containing embeddings in metadata
    """
    # Create ingestor instance
    ingestor = Ingestor(
        host=config["host"],
        port=config["port"]
    )
    
    # Configure job specification
    job_spec = JobSpec(
        batch_size=config.get("batch_size", 10),
        task_timeout=config.get("task_timeout", 120)
    )
    
    # Generate embeddings without VDB upload
    # The chain: extract -> embed -> ingest (returns results)
    results = (
        ingestor
        .files(documents)
        .extract()
        .embed()  
        .ingest()
    )
    
    return results


def save_embeddings_to_file(results, output_file):
    """
    Save extracted embeddings to a JSON file for later use.
    
    Args:
        results: Results from nv-ingest containing embeddings
        output_file: Path to save embeddings
    """
    embeddings_data = []
    
    for result in results:
        if isinstance(result, dict) and "metadata" in result:
            metadata = result["metadata"]
            
            # Check if embedding exists
            if "embedding" in metadata:
                embedding_entry = {
                    "content": metadata.get("content", ""),
                    "embedding": metadata["embedding"],
                    "source_file": metadata.get("source_metadata", {}).get("source_name", ""),
                    "page_number": metadata.get("source_metadata", {}).get("page_number"),
                    "collection": metadata.get("source_metadata", {}).get("collection", "default"),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "chunk_count": metadata.get("chunk_count", 1)
                }
                embeddings_data.append(embedding_entry)
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(embeddings_data, f, indent=2)
    
    print(f"Saved {len(embeddings_data)} embeddings to {output_file}")
    return embeddings_data


if __name__ == "__main__":
    # Configuration
    config = {
        "host": "localhost",  # or your NV-Ingest host
        "port": 8082,         # or your NV-Ingest port
        "batch_size": 10,
        "task_timeout": 120
    }
    
    # Example documents
    documents = [
        "/path/to/document1.pdf",
        "/path/to/document2.docx",
        # Add more documents as needed
    ]
    
    # Generate embeddings
    print("Generating embeddings...")
    results = generate_embeddings_only(documents, config)
    
    # Save embeddings to file
    output_file = "embeddings_output.json"
    embeddings = save_embeddings_to_file(results, output_file)
    
    # Print summary
    print(f"\nProcessing complete!")
    print(f"Total documents processed: {len(documents)}")
    print(f"Total embeddings generated: {len(embeddings)}")
    
    # Example: Access individual embeddings
    if embeddings:
        first_embedding = embeddings[0]
        print(f"\nFirst embedding sample:")
        print(f"  Source: {first_embedding['source_file']}")
        print(f"  Content preview: {first_embedding['content'][:100]}...")
        print(f"  Embedding dimensions: {len(first_embedding['embedding'])}")