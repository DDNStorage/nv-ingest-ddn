# Library Mode for NV-Ingest Embedding Generation

## Overview

Library mode is a self-contained way to run NV-Ingest designed for small-scale workloads (fewer than 100 PDFs). It's based on the official NV-Ingest documentation and provides a simplified setup that doesn't require external service deployment.

## Key Benefits

1. **Simple Setup**: No need to deploy external services for small workloads
2. **Self-contained**: Uses NIMs hosted on build.nvidia.com by default
3. **Quick Start**: Ideal for testing and prototyping
4. **Subprocess Isolation**: Runs the pipeline in a separate process for stability
5. **Predictable Performance**: `disable_dynamic_scaling=True` for consistent behavior

## How It Works

```python
# Library mode configuration
config = PipelineCreationSchema()
run_pipeline(
    config, 
    block=False,  # Non-blocking
    disable_dynamic_scaling=True,  # Better for controlled workloads
    run_in_subprocess=True  # Process isolation
)
```

## Usage

### Enable Library Mode

```bash
# Add --library-mode flag
python pipeline.py embed --input /path/to/pdfs --experiment my_exp --library-mode

# Adjust batch size for your workload
python pipeline.py embed --input /path/to/pdfs --experiment my_exp --library-mode --batch-size 20
```

### When to Use Library Mode

- **Small Datasets**: Fewer than 100 PDF files
- **Testing/Development**: Quick prototyping without full deployment
- **Local Processing**: When you don't have external NV-Ingest services running
- **Simple Workloads**: When you need a self-contained solution

### When NOT to Use Library Mode

- **Large Datasets**: 100+ files (use Docker Compose or Kubernetes deployment)
- **Production Workloads**: For production-level performance and scalability
- **High Throughput Needs**: When processing thousands of documents

## Performance Comparison

Based on the NV-Ingest architecture:

| Mode | Best For | Scale | Setup Complexity |
|------|----------|-------|------------------|
| Standard | External service deployment | Any size | Requires running services |
| Library | Self-contained processing | < 100 PDFs | Simple, no external deps |
| Docker/K8s | Production workloads | 100+ PDFs | Full deployment needed |

## Technical Details

### Message Broker Configuration

Library mode uses a different port (7671) and SimpleClient for message passing:

```python
client = NvIngestClient(
    message_client_allocator=SimpleClient,
    message_client_port=7671,  # Different from standard port
    message_client_hostname="localhost"
)
```

### Batch Processing

Files are processed in configurable batches:
- Default batch size: 10 files
- Recommended for small datasets: 5-20 files
- Keep total under 100 files for library mode

## Troubleshooting

### Library Mode Not Available

If you see a message that library mode is not available:
1. Install the required dependencies: `pip install nv-ingest nv-ingest-api`
2. The pipeline will automatically fall back to standard mode

### When to Use Standard Mode Instead

Use standard mode when:
- You have external NV-Ingest services already running
- You're processing very large datasets (100+ files)
- You need maximum throughput and have the infrastructure

Library mode is specifically designed for small workloads where simplicity is more important than maximum performance.

## Example Workflow

```bash
# 1. Process a small document collection (< 100 PDFs)
python pipeline.py embed \
    --input /data/test_documents \
    --experiment test_v1 \
    --library-mode \
    --batch-size 10

# 2. Monitor the output
# Library mode will show:
# - "Using library mode: True"
# - "Processing X files in library mode with batch size Y"
# - Progress for each batch

# 3. Check metrics
cat embeddings/legal_v1_*/generation_metrics.json
# Look for "library_mode": true in the output
```