#!/usr/bin/env python3
"""
Embedding Pipeline CLI
======================

A command-line interface for embedding generation and indexing.

Usage:
    python pipeline.py embed --input /path/to/pdfs --experiment my_experiment
    python pipeline.py index --embeddings /path/to/embeddings --collection my_collection
    python pipeline.py search --collection my_collection --query "test query"
"""

import argparse
import json
import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.embedder import EmbeddingGenerator
from core.indexer_hybrid import HybridEmbeddingIndexer
from core.searcher import EmbeddingSearcher
from core.utils import setup_logging, print_banner


def cmd_embed(args):
    """Handle embedding generation command"""
    logger = setup_logging("embedding")
    
    # Create experiment name if not provided
    if not args.experiment:
        args.experiment = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print_banner("EMBEDDING GENERATION")
    print(f"Input directory: {args.input}")
    print(f"Experiment name: {args.experiment}")
    print(f"Output directory: {args.output}")
    print(f"Batch size: {args.batch_size}")
    print(f"Checkpoints: {'enabled' if args.enable_checkpoints else 'disabled'}")
    
    # Find PDF files
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {args.input}")
        return 1
    
    pdf_files = list(input_path.glob("**/*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in: {args.input}")
        return 1
    
    print(f"Found {len(pdf_files)} PDF files")
    
    # Generate embeddings with optimized approach
    generator = EmbeddingGenerator(
        output_base_dir=args.output,
        experiment_name=args.experiment,
        nv_ingest_host=args.nv_ingest_host,
        nv_ingest_port=args.nv_ingest_port,
        batch_size=args.batch_size,
        enable_checkpoints=args.enable_checkpoints
    )
    
    try:
        metrics = generator.generate_embeddings(pdf_files)
        
        # Print summary
        print("\n" + "="*60)
        print("EMBEDDING GENERATION COMPLETE")
        print("="*60)
        print(f"Total PDFs processed: {metrics['total_documents']}")
        print(f"Total embeddings generated: {metrics['total_embeddings']}")
        print(f"Failed documents: {metrics.get('failed_documents', 0)}")
        print(f"Total time: {metrics['total_time']:.2f}s")
        if metrics['total_embeddings'] > 0:
            print(f"Average rate: {metrics.get('overall_embeddings_per_second', 0):.2f} embeddings/second")
            print(f"Documents per minute: {metrics.get('documents_per_minute', 0):.1f}")
        print(f"Output directory: {metrics['output_dir']}")
        
        if metrics.get('resumed_from_checkpoint'):
            print("(Resumed from checkpoint)")
        
        print("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return 1


def cmd_index(args):
    """Handle indexing command"""
    logger = setup_logging("indexing")
    
    print_banner("EMBEDDING INDEXING")
    print(f"Storage backend: {args.mode.upper()}")
    print(f"Embeddings directory: {args.embeddings}")
    print(f"Collection name: {args.collection}")
    print(f"Index type: {args.index_type.upper()}")
    print(f"Batch size: {args.batch_size}")
    print(f"Recreate collection: {args.recreate}")
    print(f"Enable search: {args.enable_search}")
    
    # Verify embeddings directory
    embeddings_path = Path(args.embeddings)
    if not embeddings_path.exists():
        logger.error(f"Embeddings directory does not exist: {args.embeddings}")
        return 1
    
    # Create hybrid indexer with GPU/CPU support and optimized bulk approach
    indexer = HybridEmbeddingIndexer(
        collection_name=args.collection,
        milvus_host=args.milvus_host,
        milvus_port=args.milvus_port,
        batch_size=args.batch_size,
        recreate_collection=args.recreate,
        index_type=args.index_type,
        storage_mode=args.mode,  # 'infinia' or 'gcs'
        enable_search=args.enable_search  # Include metadata fields for search
    )
    
    try:
        metrics = indexer.index_embeddings(args.embeddings)
        
        # Print summary
        print("\n" + "="*70)
        print("INDEXING COMPLETE - PERFORMANCE SUMMARY")
        print("="*70)
        print(f"Collection: {args.collection}")
        print(f"Storage backend: {args.mode.upper()}")
        print(f"Index type: {metrics.get('index_type', 'cpu').upper()} ({'GPU_CAGRA' if metrics.get('index_type', 'cpu') == 'gpu' else 'HNSW'})")
        print("-"*70)
        print("DATA METRICS:")
        print(f"  Total embeddings: {metrics['total_embeddings']:,}")
        print(f"  Successfully indexed: {metrics['processed_embeddings']:,}")
        print(f"  Failed: {metrics.get('failed_embeddings', 0):,}")
        print(f"  Success rate: {metrics['success_rate']:.2f}%")
        if 'num_segments' in metrics:
            print(f"  Number of segments: {metrics['num_segments']}")
        print(f"  Collection size: {metrics.get('collection_entities', 0):,} entities")
        print("-"*70)
        print("PERFORMANCE METRICS:")
        print(f"  Total time: {metrics['total_time']:.2f}s ({metrics['total_time']/60:.1f} minutes)")
        print(f"  Loading time: {metrics['loading_time']:.2f}s ({metrics['loading_time']/metrics['total_time']*100:.1f}%)")
        if 'aggregation_time' in metrics:
            print(f"  Aggregation time: {metrics['aggregation_time']:.2f}s ({metrics['aggregation_time']/metrics['total_time']*100:.1f}%)")
        if 'upload_time' in metrics:
            print(f"  Upload time: {metrics['upload_time']:.2f}s ({metrics['upload_time']/metrics['total_time']*100:.1f}%)")
        print(f"  Ingestion time: {metrics['ingestion_time']:.2f}s ({metrics['ingestion_time']/metrics['total_time']*100:.1f}%)")
        print(f"  Index creation time: {metrics['index_creation_time']:.2f}s ({metrics['index_creation_time']/metrics['total_time']*100:.1f}%)")
        print("-"*70)
        print("THROUGHPUT:")
        print(f"  Embeddings per second: {metrics['embeddings_per_second']:.0f}")
        print(f"  Data rate: {metrics['mb_per_second']:.2f} MB/s")
        print(f"  Average time per embedding: {metrics['total_time']/metrics['processed_embeddings']*1000:.2f}ms")
        print("="*70)
        
        # Save metrics
        metrics_file = embeddings_path / f"indexing_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {metrics_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        return 1


def cmd_search(args):
    """Handle search command"""
    logger = setup_logging("search")
    
    print_banner("EMBEDDING SEARCH")
    print(f"Collection: {args.collection}")
    print(f"Query: {args.query}")
    print(f"Top K: {args.top_k}")
    print(f"Embedding endpoint: {args.embedding_endpoint}")
    print(f"Model: {args.embedding_model}")
    
    # Create searcher
    searcher = EmbeddingSearcher(
        collection_name=args.collection,
        milvus_host=args.milvus_host,
        milvus_port=args.milvus_port
    )
    
    try:
        # Connect to Milvus
        searcher.connect()
        
        # Perform search
        results = searcher.search_by_text(
            args.query, 
            top_k=args.top_k,
            embedding_endpoint=args.embedding_endpoint,
            model_name=args.embedding_model,
            gpu_search=args.gpu_search
        )
        
        if results:
            print(f"\nFound {len(results)} results:")
            print("="*60)
            
            for idx, hit in enumerate(results):
                print(f"\n{idx+1}. Distance: {hit['distance']:.4f}")
                
                # Handle different result formats from nvingest_retrieval
                entity = hit.get('entity', hit)
                
                # Extract source info
                source_info = entity.get('source', {})
                if isinstance(source_info, dict):
                    source_name = source_info.get('source_name', 'N/A')
                else:
                    source_name = str(source_info)
                
                # Extract content metadata
                content_meta = entity.get('content_metadata', {})
                if isinstance(content_meta, dict):
                    chunk_index = content_meta.get('chunk_index', 'N/A')
                    page_number = content_meta.get('page_number', 'N/A')
                else:
                    chunk_index = 'N/A'
                    page_number = 'N/A'
                
                print(f"   Source: {source_name}")
                print(f"   Chunk: {chunk_index}")
                if page_number != 'N/A':
                    print(f"   Page: {page_number}")
                
                # Extract text
                text = entity.get('text', '')[:200]
                if text:
                    print(f"   Text: {text}...")
                print("-"*60)
        else:
            print("No results found.")
        
        return 0
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Close connection
        searcher.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Embedding Pipeline - Generate and index embeddings for PDFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate embeddings from PDFs
  python pipeline.py embed --input /path/to/pdfs --experiment exp1
  
  # Index embeddings into Milvus
  python pipeline.py index --embeddings ./embeddings/exp1_20240101_120000 --collection my_docs --mode gcs
  
  # Search in indexed collection
  python pipeline.py search --collection my_docs --query "machine learning"
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Embed command
    embed_parser = subparsers.add_parser('embed', help='Generate embeddings from PDFs')
    embed_parser.add_argument('--input', '-i', required=True, help='Input directory containing PDFs')
    embed_parser.add_argument('--experiment', '-e', help='Experiment name (default: timestamp)')
    embed_parser.add_argument('--output', '-o', default='./embeddings', help='Output base directory')
    embed_parser.add_argument('--nv-ingest-host', default='localhost', help='NV-Ingest host')
    embed_parser.add_argument('--nv-ingest-port', type=int, default=7670, help='NV-Ingest port')
    embed_parser.add_argument('--batch-size', type=int, default=10, 
                         help='Batch size for processing files (auto-adjusts based on dataset size)')
    embed_parser.add_argument('--enable-checkpoints', action='store_true', default=True,
                         help='Enable checkpoint system for resume capability (default: enabled)')
    embed_parser.add_argument('--disable-checkpoints', dest='enable_checkpoints', action='store_false',
                         help='Disable checkpoint system')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index embeddings into Milvus')
    index_parser.add_argument('--embeddings', '-e', required=True, help='Directory containing embeddings')
    index_parser.add_argument('--collection', '-c', required=True, help='Milvus collection name')
    index_parser.add_argument('--mode', '-m', required=True, help='infinia OR gcs')
    index_parser.add_argument('--batch-size', '-b', type=int, default=1000, help='Batch size for indexing')
    index_parser.add_argument('--recreate', action='store_true', help='Recreate collection if exists')
    index_parser.add_argument('--milvus-host', default='localhost', help='Milvus host')
    index_parser.add_argument('--milvus-port', type=int, default=19530, help='Milvus port')
    index_parser.add_argument('--index-type', choices=['cpu', 'gpu'], default='cpu', 
                             help='Index type: cpu (HNSW) or gpu (GPU_CAGRA)')
    index_parser.add_argument('--enable-search', action='store_true', 
                             help='Enable search capabilities by including text and metadata fields (slower indexing)')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search in indexed collection')
    search_parser.add_argument('--collection', '-c', required=True, help='Milvus collection name')
    search_parser.add_argument('--query', '-q', required=True, help='Search query')
    search_parser.add_argument('--top-k', '-k', type=int, default=5, help='Number of results')
    search_parser.add_argument('--milvus-host', default='localhost', help='Milvus host')
    search_parser.add_argument('--milvus-port', type=int, default=19530, help='Milvus port')
    search_parser.add_argument('--embedding-endpoint', default='http://localhost:8012/v1', 
                              help='Embedding service endpoint for query embedding')
    search_parser.add_argument('--embedding-model', default='nvidia/llama-3.2-nv-embedqa-1b-v2',
                              help='Embedding model name')
    search_parser.add_argument('--gpu-search', action='store_true', default=True,
                              help='Use GPU for search (if available)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    if args.command == 'embed':
        return cmd_embed(args)
    elif args.command == 'index':
        return cmd_index(args)
    elif args.command == 'search':
        return cmd_search(args)


if __name__ == "__main__":
    sys.exit(main())