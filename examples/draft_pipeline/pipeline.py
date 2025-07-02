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
from datetime import datetime
from pathlib import Path

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.embedder import EmbeddingGenerator
from core.embedder_optimized import OptimizedEmbeddingGenerator
from core.indexer import EmbeddingIndexer
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
    print(f"Library mode: {args.library_mode}")
    if args.library_mode:
        print(f"Batch size: {args.batch_size}")
    
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
    
    # Generate embeddings
    if args.library_mode:
        generator = OptimizedEmbeddingGenerator(
            output_base_dir=args.output,
            experiment_name=args.experiment,
            nv_ingest_host=args.nv_ingest_host,
            nv_ingest_port=args.nv_ingest_port,
            use_library_mode=True,
            batch_size=args.batch_size
        )
    else:
        generator = EmbeddingGenerator(
            output_base_dir=args.output,
            experiment_name=args.experiment,
            nv_ingest_host=args.nv_ingest_host,
            nv_ingest_port=args.nv_ingest_port
        )
    
    try:
        metrics = generator.generate_embeddings(pdf_files)
        
        # Print summary
        print("\n" + "="*60)
        print("EMBEDDING GENERATION COMPLETE")
        print("="*60)
        print(f"Total PDFs processed: {metrics['total_documents']}")
        print(f"Total embeddings generated: {metrics['total_embeddings']}")
        print(f"Total time: {metrics['total_time']:.2f}s")
        print(f"Output directory: {metrics['output_dir']}")
        print("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return 1
    finally:
        # Clean up resources for optimized generator
        if args.library_mode and hasattr(generator, 'close'):
            generator.close()


def cmd_index(args):
    """Handle indexing command"""
    logger = setup_logging("indexing")
    
    print_banner("EMBEDDING INDEXING")
    print(f"Embeddings directory: {args.embeddings}")
    print(f"Collection name: {args.collection}")
    print(f"Batch size: {args.batch_size}")
    print(f"Recreate collection: {args.recreate}")
    
    # Verify embeddings directory
    embeddings_path = Path(args.embeddings)
    if not embeddings_path.exists():
        logger.error(f"Embeddings directory does not exist: {args.embeddings}")
        return 1
    
    # Create indexer
    indexer = EmbeddingIndexer(
        collection_name=args.collection,
        milvus_host=args.milvus_host,
        milvus_port=args.milvus_port,
        batch_size=args.batch_size,
        recreate_collection=args.recreate
    )
    
    try:
        metrics = indexer.index_embeddings(args.embeddings)
        
        # Print summary
        print("\n" + "="*60)
        print("INDEXING COMPLETE")
        print("="*60)
        print(f"Collection: {args.collection}")
        print(f"Total embeddings: {metrics['total_embeddings']}")
        print(f"Successfully indexed: {metrics['processed_embeddings']}")
        print(f"Failed: {metrics['failed_embeddings']}")
        print(f"Success rate: {metrics['success_rate']:.2f}%")
        print(f"\nPerformance:")
        print(f"  Total time: {metrics['total_time']:.2f}s")
        print(f"  Loading time: {metrics['loading_time']:.2f}s")
        print(f"  Ingestion time: {metrics['ingestion_time']:.2f}s")
        print(f"  Index creation time: {metrics['index_creation_time']:.2f}s")
        print(f"  Throughput: {metrics['embeddings_per_second']:.2f} embeddings/s")
        print(f"  Data rate: {metrics['mb_per_second']:.2f} MB/s")
        print("="*60)
        
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
  python pipeline.py index --embeddings ./embeddings/exp1_20240101_120000 --collection my_docs
  
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
    embed_parser.add_argument('--library-mode', action='store_true', 
                             help='Use optimized library mode for better performance (requires additional dependencies)')
    embed_parser.add_argument('--batch-size', type=int, default=10, 
                             help='Batch size for processing files in library mode')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index embeddings into Milvus')
    index_parser.add_argument('--embeddings', '-e', required=True, help='Directory containing embeddings')
    index_parser.add_argument('--collection', '-c', required=True, help='Milvus collection name')
    index_parser.add_argument('--batch-size', '-b', type=int, default=1000, help='Batch size for indexing')
    index_parser.add_argument('--recreate', action='store_true', help='Recreate collection if exists')
    index_parser.add_argument('--milvus-host', default='localhost', help='Milvus host')
    index_parser.add_argument('--milvus-port', type=int, default=19530, help='Milvus port')
    
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