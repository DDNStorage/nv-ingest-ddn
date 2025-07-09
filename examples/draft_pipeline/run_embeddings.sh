#!/bin/bash
# Generate embeddings using NV-Ingest

set -e

# Configuration
PDF_DIR="${1:-./data/digital_corpora_100}"
EXPERIMENT_NAME="${2:-embeddings}"

# Check if third argument is a flag (starts with --) or a directory
if [[ "$3" == --* ]] || [ -z "$3" ]; then
    OUTPUT_DIR="./embeddings"
    ARGS_TO_SHIFT=2
else
    OUTPUT_DIR="$3"
    ARGS_TO_SHIFT=3
fi

# Validate input
if [ ! -d "$PDF_DIR" ]; then
    echo "Error: PDF directory not found: $PDF_DIR"
    echo "Usage: $0 <pdf_directory> [experiment_name] [output_directory] [--library-mode] [--batch <size>]"
    echo "Example: $0 ./data/test_folder test_1"
    echo "Example with library mode: $0 ./data/test_folder test_1 ./embeddings --library-mode"
    echo "Example with batch size: $0 ./data/test_folder test_1 ./embeddings --batch 100"
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

# Parse optional arguments
LIBRARY_MODE=""
BATCH_ARG=""

# Shift the positional arguments we actually received
shift $ARGS_TO_SHIFT 2>/dev/null || true

# Parse remaining arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --library-mode)
            LIBRARY_MODE="--library-mode"
            echo "Library mode: enabled"
            shift
            ;;
        --batch)
            if [ -z "$2" ] || [[ "$2" == --* ]]; then
                echo "Error: --batch requires a size argument"
                exit 1
            fi
            if ! [[ "$2" =~ ^[0-9]+$ ]] || [ "$2" -eq 0 ]; then
                echo "Error: Batch size must be a positive integer"
                exit 1
            fi
            BATCH_ARG="--batch-size $2"
            echo "Batch size: $2"
            shift 2
            ;;
        *)
            echo "Warning: Unknown argument: $1"
            shift
            ;;
    esac
done

if [ -z "$LIBRARY_MODE" ]; then
    echo "Library mode: disabled (use --library-mode flag to enable)"
fi

echo "============================================"

# Run embedding generation
echo "Starting embedding generation..."
python pipeline.py embed \
    --input "$PDF_DIR" \
    --experiment "$EXPERIMENT_NAME" \
    --output "$OUTPUT_DIR" \
    $LIBRARY_MODE \
    $BATCH_ARG

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