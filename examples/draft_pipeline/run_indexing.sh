#!/bin/bash
# Index embeddings using NVIDIA's optimized bulk indexing approach
# Supports GCS with default credentials and Infinia storage backends
# Assumes Milvus is already running with required backend

set -e

# Configuration
EMBEDDINGS_DIR="${1}"
COLLECTION_NAME="${2:-my_collection}"
INDEX_TYPE="${3:-cpu}"
STORAGE_MODE="${4:-gcs}"  # Default to GCS for testing

# Performance tuning parameters
export MAX_SEGMENT_ROWS="${MAX_SEGMENT_ROWS:-240000}"  # ~1GB segments
export NUM_UPLOAD_WORKERS="${NUM_UPLOAD_WORKERS:-20}"

# Validate input
if [ -z "$EMBEDDINGS_DIR" ] || [ ! -d "$EMBEDDINGS_DIR" ]; then
    echo "Error: Embeddings directory not found: $EMBEDDINGS_DIR"
    echo "Usage: $0 <embeddings_directory> [collection_name] [cpu|gpu] [gcs|infinia]"
    echo "Example: $0 ./embeddings/embeddings_20250703_125614 my_collection gpu gcs"
    exit 1
fi

if [ "$INDEX_TYPE" != "cpu" ] && [ "$INDEX_TYPE" != "gpu" ]; then
    echo "Error: Index type must be 'cpu' or 'gpu'"
    exit 1
fi

if [ "$STORAGE_MODE" != "infinia" ] && [ "$STORAGE_MODE" != "gcs" ]; then
    echo "Error: Storage mode must be 'infinia' or 'gcs'"
    exit 1
fi

# GCS-specific configuration for default credentials
if [ "$STORAGE_MODE" = "gcs" ]; then
    # Check if running in GCP environment
    if [ -n "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
        echo "Using service account credentials from: $GOOGLE_APPLICATION_CREDENTIALS"
    else
        echo "Using GCP default credentials (gcloud auth)"
        # Ensure proper scopes for GCS access
        export GOOGLE_AUTH_SCOPES="https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/devstorage.full_control"
    fi
fi

echo "============================================"
echo "OPTIMIZED BULK INDEXING"
echo "============================================"
echo "Embeddings: $EMBEDDINGS_DIR"
echo "Collection: $COLLECTION_NAME"
echo "Index type: ${INDEX_TYPE^^}"
echo "Storage mode: ${STORAGE_MODE^^}"
echo "--------------------------------------------"
echo "Performance Settings:"
echo "  Max segment rows: $MAX_SEGMENT_ROWS"
echo "  Upload workers: $NUM_UPLOAD_WORKERS"

# if [ "$STORAGE_MODE" = "infinia" ]; then
#     echo "Running Infinia Docker file..."
#     ../../../rag-nim-milvus/run_docker_compose/run_docker_multimodal_infinia.sh
# elif [ "$STORAGE_MODE" = "gcs" ]; then
#     echo "Running GCS Docker file..."
#     ../../../rag-nim-milvus/run_docker_compose/run_docker_multimodal_gcs.sh
# fi

# echo "Docker Containers are being started..."
# sleep 60

# # Check if CA certificate is configured
# if [ ! -z "$MY_STORAGE_CA_CERT" ]; then
#     echo "CA Certificate: $MY_STORAGE_CA_CERT"
# elif [ "$STORAGE_MODE" = "infinia" ]; then
#     # Set default certificate for Infinia if not already set
#     DEFAULT_CERT="/home/artemivashchenko/src/infinia-ai-workload-poc/rag-nim-milvus/certificates/red_ca_8A5EC7E9-B579-50C7-FA0D-197589C7D6A9.crt"
#     if [ -f "$DEFAULT_CERT" ]; then
#         export MY_STORAGE_CA_CERT="$DEFAULT_CERT"
#         echo "CA Certificate: $MY_STORAGE_CA_CERT (default)"
#     fi
# fi
# echo "============================================"

# Test Milvus connection
echo ""
echo "Testing Milvus connection..."
if python -c "
from pymilvus import connections
try:
    connections.connect('default', host='localhost', port=19530, timeout=10)
    print('✅ Milvus connection successful')
    connections.disconnect('default')
except Exception as e:
    print(f'❌ Milvus connection failed: {e}')
    exit(1)
"; then
    echo "Connection test passed"
else
    echo "Error: Cannot connect to Milvus. Please ensure Milvus is running."
    exit 1
fi

# Count embeddings from metadata for accuracy
if [ -f "$EMBEDDINGS_DIR/embeddings_metadata.json" ]; then
    METADATA_COUNT=$(python -c "import json; f=open('$EMBEDDINGS_DIR/embeddings_metadata.json'); m=json.load(f); print(len(m)); f.close()" 2>/dev/null || echo "0")
    FILE_COUNT=$(find "$EMBEDDINGS_DIR" -name "*.npy" | wc -l)
    echo "  Embeddings in metadata: $METADATA_COUNT"
    echo "  .npy files found: $FILE_COUNT"
    
    # Warn if there's a discrepancy
    if [ "$METADATA_COUNT" -ne "$FILE_COUNT" ]; then
        echo "  ⚠️  WARNING: File count mismatch! Metadata entries don't match .npy files"
        echo "     This may indicate incomplete processing or orphaned files"
    fi
else
    echo "  ❌ ERROR: No metadata file found!"
    FILE_COUNT=$(find "$EMBEDDINGS_DIR" -name "*.npy" | wc -l)
    echo "  .npy files found: $FILE_COUNT"
fi
echo "============================================"

# Run indexing with optimized bulk approach
echo ""
echo "Starting optimized indexing process..."

# Check if --enable-search flag is passed
if [[ " $@ " =~ " --enable-search " ]]; then
    echo "ℹ️  Search mode enabled - including metadata fields"
    python pipeline.py index \
        --embeddings "$EMBEDDINGS_DIR" \
        --collection "$COLLECTION_NAME" \
        --mode "$STORAGE_MODE" \
        --index-type "$INDEX_TYPE" \
        --recreate \
        --batch-size 5000 \
        --enable-search
else
    echo "ℹ️  Performance mode - minimal schema (id + vector only)"
    python pipeline.py index \
        --embeddings "$EMBEDDINGS_DIR" \
        --collection "$COLLECTION_NAME" \
        --mode "$STORAGE_MODE" \
        --index-type "$INDEX_TYPE" \
        --recreate \
        --batch-size 5000
fi

echo ""
echo "============================================"
echo "INDEXING COMPLETE"
echo "============================================"