# Embedding Pipeline

A production-ready pipeline for generating and indexing embeddings from PDF documents using NV-Ingest and Milvus.

## Features

- **Embedding Generation**: Uses NV-Ingest to extract text and generate embeddings from PDFs
- **Efficient Indexing**: Batch processing with progress tracking and detailed metrics
- **Index Monitoring**: Real-time status updates during index creation
- **Search Capabilities**: Query indexed embeddings
- **Comprehensive Metrics**: Detailed performance metrics and logging

## Installation

1. Ensure you have the required dependencies:
```bash
pip install numpy pymilvus tqdm nv-ingest-client
```

2. Make sure NV-Ingest and Milvus are running:
- NV-Ingest: Default port 7670
- Milvus: Default port 19530

## Usage

### 1. Generate Embeddings from PDFs

```bash
python pipeline.py embed --input /path/to/pdf/folder --experiment my_experiment
```

Options:
- `--input, -i`: Directory containing PDF files (required)
- `--experiment, -e`: Experiment name (default: timestamp)
- `--output, -o`: Output base directory (default: ./embeddings)
- `--nv-ingest-host`: NV-Ingest host (default: localhost)
- `--nv-ingest-port`: NV-Ingest port (default: 7670)
- `--library-mode`: Use optimized library mode for better performance
- `--batch-size`: Batch size for processing files in library mode (default: 10)

Example:
```bash
# Standard mode
python pipeline.py embed -i ./documents -e legal_docs_v1

# Optimized library mode (recommended for large datasets)
python pipeline.py embed -i ./documents -e legal_docs_v2 --library-mode --batch-size 20
```

This will create an output directory like:
```
./embeddings/legal_docs_v1_20240315_143022/
├── embeddings_metadata.json
├── generation_metrics.json
├── document1_chunk_0000.npy
├── document1_chunk_0001.npy
└── ...
```

### 2. Index Embeddings into Milvus

```bash
python pipeline.py index --embeddings /path/to/embeddings --collection my_collection
```

Options:
- `--embeddings, -e`: Directory containing embeddings (required)
- `--collection, -c`: Milvus collection name (required)
- `--batch-size, -b`: Batch size for indexing (default: 1000)
- `--recreate`: Recreate collection if exists
- `--milvus-host`: Milvus host (default: localhost)
- `--milvus-port`: Milvus port (default: 19530)

Example:
```bash
python pipeline.py index -e ./embeddings/legal_docs_v1_20240315_143022 -c legal_docs --recreate
```

During indexing, you'll see:
- Loading progress
- Ingestion progress with batch numbers
- Real-time index building status (Started → Completed)
- Performance metrics

### 3. Search Indexed Collection

```bash
python pipeline.py search --collection my_collection --query "search terms"
```

Options:
- `--collection, -c`: Milvus collection name (required)
- `--query, -q`: Search query (required)
- `--top-k, -k`: Number of results (default: 5)
- `--milvus-host`: Milvus host (default: localhost)
- `--milvus-port`: Milvus port (default: 19530)
- `--embedding-endpoint`: Embedding service endpoint (default: http://localhost:8012/v1)
- `--embedding-model`: Embedding model name (default: nvidia/llama-3.2-nv-embedqa-1b-v2)
- `--gpu-search`: Use GPU for search if available (default: True)

Example:
```bash
python pipeline.py search -c legal_docs -q "contract termination clause" -k 10

# With custom embedding endpoint
python pipeline.py search -c legal_docs -q "machine learning algorithms" \
    --embedding-endpoint http://embedding-service:8012/v1 \
    --embedding-model nvidia/llama-3.2-nv-embedqa-1b-v2
```

**Note**: Text search requires an embedding service (NIM) to convert queries into vectors. Make sure the embedding endpoint is accessible.

## Output Structure

### Embedding Generation Output

```
embeddings/
└── experiment_name_YYYYMMDD_HHMMSS/
    ├── embeddings_metadata.json    # Metadata for all embeddings
    ├── generation_metrics.json      # Performance metrics
    └── *.npy                       # Numpy arrays of embeddings
```

### Metrics Examples

**Generation Metrics** (`generation_metrics.json`):
```json
{
  "experiment_name": "legal_docs_v1",
  "total_documents": 10,
  "total_embeddings": 523,
  "failed_documents": 0,
  "total_time": 45.23,
  "document_metrics": [...]
}
```

**Indexing Metrics** (`indexing_metrics_YYYYMMDD_HHMMSS.json`):
```json
{
  "collection_name": "legal_docs",
  "total_embeddings": 523,
  "processed_embeddings": 523,
  "success_rate": 100.0,
  "total_time": 12.45,
  "embeddings_per_second": 42.01,
  "mb_per_second": 10.23
}
```

## Performance Tips

1. **Batch Size**: Larger batches (1000-5000) are more efficient for large datasets
2. **Parallel Processing**: The pipeline uses async operations where possible
3. **Memory Usage**: Monitor memory when processing large PDF collections
4. **Index Type**: HNSW index provides good balance of speed and accuracy

### Library Mode for Embedding Generation

For small-scale workloads (fewer than 100 PDFs), use the `--library-mode` flag which provides:
- **Self-contained Pipeline**: Runs NV-Ingest locally without needing external services
- **Simplified Setup**: Uses NIMs hosted on build.nvidia.com or self-hosted
- **Quick Processing**: Ideal for testing and small document collections
- **Subprocess Isolation**: Runs pipeline in separate process with `disable_dynamic_scaling=True`

Example for small datasets:
```bash
# Process small collections efficiently
python pipeline.py embed -i ./documents -e test_docs --library-mode --batch-size 10
```

Note: Library mode is designed for workloads of fewer than 100 PDFs. For production-level performance and scalability with larger datasets, use Docker Compose or Kubernetes deployment.

## Troubleshooting

### Common Issues

1. **Connection Errors**: Ensure NV-Ingest and Milvus are running
2. **Memory Issues**: Reduce batch size or process PDFs in smaller groups
3. **Slow Indexing**: Check Milvus server resources and configuration

### Logs

The pipeline creates detailed logs:
- `pipeline_embedding_YYYYMMDD.log`
- `pipeline_indexing_YYYYMMDD.log`
- `pipeline_search_YYYYMMDD.log`

## Example Workflow

```bash
python pipeline.py embed -i ./research_papers -e arxiv_2024

# 2. Index the embeddings
python pipeline.py index -e ./embeddings/arxiv_2024_20240315_143022 -c research_papers --recreate

# 3. Search the collection
python pipeline.py search -c research_papers -q "transformer architecture attention mechanism"
```

## Architecture

```
pipeline.py (CLI)
    │
    ├── core/embedder.py    → NV-Ingest integration
    ├── core/indexer.py     → Milvus indexing with progress
    ├── core/searcher.py    → Query functionality
    └── core/utils.py       → Shared utilities
```