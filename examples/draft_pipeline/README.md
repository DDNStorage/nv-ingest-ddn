# NV-Ingest DDN Pipeline

This pipeline provides a scalable solution for generating embeddings from PDFs using NV-Ingest and indexing them into Milvus with different storage backends.

## Quick Start

### 1. Generate Embeddings
```bash
# Standard mode (uses external NV-Ingest service)
./run_embeddings.sh ./data/digital_corpora_100 test_100

# Library mode (runs NV-Ingest locally - for small datasets)
./run_embeddings.sh ./data/digital_corpora_100 test_100 ./embeddings --library-mode
```

This will generate embeddings and output the directory path, e.g.:
```
Embeddings saved to: ./embeddings/test_100_20250703_125614
```

### 2. Index Embeddings
```bash
# Use the simplified indexing script (manually deploy backend first)
./run_indexing.sh ./embeddings/test_100_20250703_125614 my_collection cpu infinia

# Or use backend-specific scripts (kept for compatibility)
./run_indexing_infinia.sh ./embeddings/test_100_20250703_125614 cpu my_collection
./run_indexing_gcs.sh ./embeddings/test_100_20250703_125614 cpu my_collection
```

## Scripts

- `run_embeddings.sh` - Generate embeddings from PDFs
- `run_indexing.sh` - Simplified indexing script (requires pre-deployed backend)
- `run_indexing_infinia.sh` - Index embeddings using Infinia storage backend (no auto-switch)
- `run_indexing_gcs.sh` - Index embeddings using GCS storage backend (no auto-switch)

## Features

- **Two Processing Modes**: Standard mode (external NV-Ingest) or library mode (local processing)
- **Hybrid Indexing**: Automatically chooses stream insert for <1000 embeddings, bulk insert for larger datasets
- **GPU/CPU Support**: Choose between HNSW (CPU) or GPU_CAGRA (GPU) indexes
- **Performance Metrics**: Detailed performance summary after each operation

## Pipeline Components

- `pipeline.py` - Main CLI interface
- `core/embedder.py` - Standard embedder using external NV-Ingest service
- `core/embedder_optimized.py` - Library mode embedder for small workloads
- `core/indexer_hybrid.py` - Hybrid indexer with stream/bulk insert support
- `core/searcher.py` - Search functionality
- `core/utils.py` - Utility functions

## Example Workflow

```bash
# 1. Generate embeddings for 100 PDFs
./run_embeddings.sh ./data/digital_corpora_100 experiment_100

# 2. Deploy your backend manually (e.g., Infinia or GCS)
# Then run indexing with the simplified script:
EMBEDDINGS_DIR=./embeddings/experiment_100_20250703_125614

# For Infinia backend
./run_indexing.sh $EMBEDDINGS_DIR infinia_cpu_collection cpu infinia
./run_indexing.sh $EMBEDDINGS_DIR infinia_gpu_collection gpu infinia

# For GCS backend (after manually switching backend)
./run_indexing.sh $EMBEDDINGS_DIR gcs_cpu_collection cpu gcs
./run_indexing.sh $EMBEDDINGS_DIR gcs_gpu_collection gpu gcs
```

## Performance Summary

After indexing, you'll see a detailed performance summary:
```
======================================================================
INDEXING COMPLETE - PERFORMANCE SUMMARY
======================================================================
Collection: my_collection
Storage backend: INFINIA
Index type: GPU (GPU_CAGRA)
Ingestion method: STREAM
----------------------------------------------------------------------
DATA METRICS:
  Total embeddings: 1,234
  Successfully indexed: 1,234
  Failed: 0
  Success rate: 100.00%
  Collection size: 1,234 entities
----------------------------------------------------------------------
PERFORMANCE METRICS:
  Total time: 45.23s (0.8 minutes)
  Loading time: 2.34s (5.2%)
  Ingestion time: 12.45s (27.5%)
  Index creation time: 30.44s (67.3%)
----------------------------------------------------------------------
THROUGHPUT:
  Embeddings per second: 27
  Data rate: 2.34 MB/s
  Average time per embedding: 36.65ms
======================================================================
```

## Direct Pipeline Usage

You can also use the pipeline directly:

### Embedding Generation
```bash
python pipeline.py embed --input /path/to/pdfs --experiment my_test
```

### Indexing
```bash
python pipeline.py index \
    --embeddings ./embeddings/my_test_20250703_125614 \
    --collection my_collection \
    --mode infinia \
    --index-type gpu \
    --recreate
```

### Search
```bash
python pipeline.py search \
    --collection my_collection \
    --query "machine learning algorithms" \
    --top-k 10
```

## Requirements

- Python 3.8+
- NV-Ingest service running (or library mode dependencies)
- Milvus instance with appropriate storage backend
- Dependencies: `pip install -r requirements.txt`

## Environment Variables

The pipeline automatically reads storage credentials from environment variables set by Docker scripts:
- For Infinia: `MULTIMODAL_*` or `MY_*` variables
- For GCS: `MY_*` variables

No manual configuration needed when using the provided scripts!

## Backend Management

The backend detection and switching logic has been removed. You need to:
1. Manually deploy the required Milvus backend (Infinia or GCS)
2. Ensure proper environment variables are set for S3 credentials
3. Run the indexing script with the appropriate storage mode

The simplified `run_indexing.sh` script accepts the storage mode as a parameter but assumes the backend is already running.