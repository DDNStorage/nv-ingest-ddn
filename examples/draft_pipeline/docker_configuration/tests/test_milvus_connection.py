#!/usr/bin/env python3
"""
Test Milvus Connection Script
Tests basic Milvus operations: connect, create collection, insert vectors, search, and cleanup
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
    MilvusException
)


class MilvusConnectionTest:
    """Test class for Milvus connectivity and basic operations"""
    
    def __init__(self, host: str = "localhost", port: int = 19530):
        self.host = host
        self.port = port
        self.collection_name = "test_collection_cpu_mode"
        self.dim = 128  # Vector dimension
        self.num_vectors = 100  # Number of test vectors
        
    def connect(self) -> bool:
        """Connect to Milvus server"""
        try:
            print(f"Connecting to Milvus at {self.host}:{self.port}...")
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
                timeout=30
            )
            print("✓ Successfully connected to Milvus")
            return True
        except Exception as e:
            print(f"✗ Failed to connect to Milvus: {e}")
            return False
    
    def create_collection(self) -> bool:
        """Create a test collection"""
        try:
            # Check if collection already exists
            if utility.has_collection(self.collection_name):
                print(f"Collection '{self.collection_name}' already exists, dropping it...")
                collection = Collection(self.collection_name)
                collection.drop()
                print("✓ Existing collection dropped")
            
            # Define collection schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=200)
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description="Test collection for CPU mode validation"
            )
            
            # Create collection
            print(f"Creating collection '{self.collection_name}'...")
            collection = Collection(
                name=self.collection_name,
                schema=schema,
                consistency_level="Strong"
            )
            print(f"✓ Collection created successfully")
            print(f"  - Vector dimension: {self.dim}")
            print(f"  - Fields: {[f.name for f in fields]}")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to create collection: {e}")
            return False
    
    def insert_vectors(self) -> bool:
        """Insert test vectors into the collection"""
        try:
            collection = Collection(self.collection_name)
            
            # Generate random test vectors
            print(f"\nGenerating {self.num_vectors} test vectors...")
            vectors = np.random.rand(self.num_vectors, self.dim).astype(np.float32)
            
            # Normalize vectors (common practice for embeddings)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / norms
            
            # Generate metadata
            metadata = [f"test_vector_{i}" for i in range(self.num_vectors)]
            
            # Prepare data for insertion
            data = [
                vectors.tolist(),  # embeddings
                metadata          # metadata
            ]
            
            # Insert data
            print("Inserting vectors...")
            start_time = time.time()
            result = collection.insert(data)
            insert_time = time.time() - start_time
            
            print(f"✓ Successfully inserted {len(result.primary_keys)} vectors")
            print(f"  - Insert time: {insert_time:.3f} seconds")
            print(f"  - Throughput: {self.num_vectors/insert_time:.1f} vectors/sec")
            
            # Flush to ensure data is persisted
            print("Flushing data...")
            collection.flush()
            print("✓ Data flushed successfully")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to insert vectors: {e}")
            return False
    
    def create_index(self) -> bool:
        """Create index on the vector field"""
        try:
            collection = Collection(self.collection_name)
            
            # Define index parameters
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            
            print("\nCreating index...")
            print(f"  - Index type: {index_params['index_type']}")
            print(f"  - Metric type: {index_params['metric_type']}")
            
            start_time = time.time()
            collection.create_index(
                field_name="embeddings",
                index_params=index_params
            )
            index_time = time.time() - start_time
            
            print(f"✓ Index created successfully in {index_time:.3f} seconds")
            
            # Load collection
            print("Loading collection...")
            collection.load()
            print("✓ Collection loaded successfully")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to create index: {e}")
            return False
    
    def search_vectors(self) -> bool:
        """Perform vector similarity search"""
        try:
            collection = Collection(self.collection_name)
            
            # Generate query vectors
            num_queries = 5
            print(f"\nGenerating {num_queries} query vectors...")
            query_vectors = np.random.rand(num_queries, self.dim).astype(np.float32)
            query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)
            
            # Search parameters
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10}
            }
            
            # Perform search
            print("Performing similarity search...")
            start_time = time.time()
            results = collection.search(
                data=query_vectors.tolist(),
                anns_field="embeddings",
                param=search_params,
                limit=10,
                output_fields=["metadata"]
            )
            search_time = time.time() - start_time
            
            print(f"✓ Search completed successfully")
            print(f"  - Search time: {search_time:.3f} seconds")
            print(f"  - Queries per second: {num_queries/search_time:.1f}")
            
            # Display sample results
            print("\nSample search results (first query):")
            for i, hit in enumerate(results[0][:3]):
                print(f"  {i+1}. ID: {hit.id}, Distance: {hit.distance:.4f}, "
                      f"Metadata: {hit.entity.get('metadata')}")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to search vectors: {e}")
            return False
    
    def get_collection_stats(self) -> bool:
        """Get collection statistics"""
        try:
            collection = Collection(self.collection_name)
            
            print("\nCollection Statistics:")
            print(f"  - Name: {collection.name}")
            print(f"  - Entities: {collection.num_entities}")
            # print(f"  - Loaded: {collection.is_loaded}")  # Not available in all versions
            
            # Collection stats are already displayed above
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to get collection stats: {e}")
            return False
    
    def cleanup(self) -> bool:
        """Clean up test collection"""
        try:
            print("\nCleaning up...")
            if utility.has_collection(self.collection_name):
                collection = Collection(self.collection_name)
                collection.drop()
                print(f"✓ Collection '{self.collection_name}' dropped successfully")
            
            # Disconnect
            connections.disconnect("default")
            print("✓ Disconnected from Milvus")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to cleanup: {e}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all tests in sequence"""
        print("="*60)
        print(f"   Milvus Connection Test - {self.host}:{self.port}")
        print("="*60)
        
        tests = [
            ("Connect to Milvus", self.connect),
            ("Create Collection", self.create_collection),
            ("Insert Vectors", self.insert_vectors),
            ("Create Index", self.create_index),
            ("Search Vectors", self.search_vectors),
            ("Get Collection Stats", self.get_collection_stats),
            ("Cleanup", self.cleanup)
        ]
        
        all_passed = True
        results = []
        
        for test_name, test_func in tests:
            print(f"\n{'='*40}")
            print(f"Test: {test_name}")
            print('='*40)
            
            try:
                passed = test_func()
                results.append((test_name, passed))
                if not passed:
                    all_passed = False
                    print(f"\n✗ Test '{test_name}' failed, stopping further tests")
                    break
            except Exception as e:
                print(f"\n✗ Unexpected error in '{test_name}': {e}")
                results.append((test_name, False))
                all_passed = False
                break
        
        # Print summary
        print("\n" + "="*60)
        print("                    TEST SUMMARY")
        print("="*60)
        
        for test_name, passed in results:
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"{test_name:<30} {status}")
        
        print("="*60)
        
        if all_passed:
            print("\n✓ All tests passed successfully!")
        else:
            print("\n✗ Some tests failed!")
        
        return all_passed


def main():
    parser = argparse.ArgumentParser(description='Test Milvus connection and basic operations')
    parser.add_argument('--host', type=str, default='localhost',
                        help='Milvus host (default: localhost)')
    parser.add_argument('--port', type=int, default=19530,
                        help='Milvus port (default: 19530)')
    parser.add_argument('--env', type=str, choices=['infinia', 'gcs'],
                        help='Environment being tested (for logging purposes)')
    
    args = parser.parse_args()
    
    if args.env:
        print(f"\nTesting Milvus with {args.env.upper()} storage backend")
    
    # Run tests
    tester = MilvusConnectionTest(host=args.host, port=args.port)
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()