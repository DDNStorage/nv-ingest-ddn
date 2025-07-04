#!/bin/bash
# Generate embeddings using NV-Ingest

set -e

# Configuration
PDF_DIR="${1:-./data/digital_corpora_100}"
EXPERIMENT_NAME="${2:-embeddings}"
OUTPUT_DIR="${3:-./embeddings}"

# Validate input
if [ ! -d "$PDF_DIR" ]; then
    echo "Error: PDF directory not found: $PDF_DIR"
    echo "Usage: $0 <pdf_directory> [experiment_name] [output_directory] [--library-mode]"
    echo "Example: $0 ./data/test_folder test_1"
    echo "Example with library mode: $0 ./data/test_folder test_1 ./embeddings --library-mode"
    exit 1
fi

# Count PDFs
PDF_COUNT=$(find "$PDF_DIR" -name "*.pdf" | wc -l)
if [ $PDF_COUNT -eq 0 ]; then
    echo "Error: No PDF files found in $PDF_DIR"
    exit 1
fi

echo "============================================"
echo "EMBEDDING GENERATION"
echo "============================================"
echo "PDFs directory: $PDF_DIR"
echo "PDF count: $PDF_COUNT"
echo "Experiment name: $EXPERIMENT_NAME"
echo "Output directory: $OUTPUT_DIR"
echo "============================================"

# Check if library mode is explicitly requested
LIBRARY_MODE=""
for arg in "$@"; do
    if [ "$arg" = "--library-mode" ]; then
        LIBRARY_MODE="--library-mode"
        echo "Library mode explicitly enabled"
        break
    fi
done

if [ -z "$LIBRARY_MODE" ]; then
    echo "Using standard mode (use --library-mode flag to enable library mode)"
fi

# Run embedding generation
echo "Starting embedding generation..."
python pipeline.py embed \
    --input "$PDF_DIR" \
    --experiment "$EXPERIMENT_NAME" \
    --output "$OUTPUT_DIR" \
    $LIBRARY_MODE

# Get the actual output directory
EMBEDDINGS_DIR=$(ls -td ${OUTPUT_DIR}/${EXPERIMENT_NAME}_* | head -1)

if [ -d "$EMBEDDINGS_DIR" ]; then
    echo ""
    echo "============================================"
    echo "EMBEDDING GENERATION COMPLETE"
    echo "============================================"
    echo "Embeddings saved to: $EMBEDDINGS_DIR"
    echo "============================================"
else
    echo "Error: Embeddings directory not found"
    exit 1
fi