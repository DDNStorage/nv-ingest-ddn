# Milvus Bulk Load Optimizer

A dynamic configuration system that automatically detects system resources and generates optimized Milvus deployments for bulk loading operations. Designed to handle workloads from millions to billions of embeddings.

## Features

- **Dynamic Resource Detection**: Automatically detects CPU, RAM, GPU, and storage capabilities
- **Simple Storage Selection**: Choose storage backend with `--env` flag (no file editing required)
- **Explicit GPU/CPU Mode**: Control hardware usage with `--gpu` or `--cpu` flags
- **Two Operation Modes**:
  - **Auto Mode**: Uses all available resources for maximum performance
  - **Target Mode**: Optimizes for specific embedding counts (e.g., 20M, 1B)
- **Multi-Storage Support**: Works with Infinia, Google Cloud Storage, and MinIO
- **Bulk Load Optimized**: All configurations tuned for high-throughput data ingestion
- **Scalable Architecture**: Dynamically scales nodes based on resources

## Quick Start

### 1. Basic Usage

```bash
# Navigate to the directory
cd docker_configuration

# Run with MinIO (default) - auto-detects GPU
./scripts/run_milvus.sh

# Run with Infinia storage on GPU
./scripts/run_milvus.sh --env infinia --gpu

# Run with GCS storage on CPU only
./scripts/run_milvus.sh --env gcs --cpu

# Run with MinIO, optimized for 20M embeddings
./scripts/run_milvus.sh --env minio --target-embeddings 20M
```

### 2. Storage Backends

The system includes pre-configured environment files for each storage backend:

- **`.env.infinia`** - Infinia S3-compatible storage
- **`.env.gcs`** - Google Cloud Storage
- **`.env.minio`** - Local MinIO storage (default)

Simply use the `--env` flag to select your storage backend. No file editing required!

### 3. GPU/CPU Selection

- **Auto-detect** (default): The system detects GPU availability automatically
- **Force GPU**: Use `--gpu` flag to ensure GPU mode
- **Force CPU**: Use `--cpu` flag to run in CPU-only mode

## Storage Configuration

### Initial Setup

For first-time setup, you'll need to configure credentials in the appropriate environment file:

#### Infinia Storage
Edit `.env.infinia`:
```env
STORAGE_ACCESS_KEY=your-access-key-here
STORAGE_SECRET_KEY=your-secret-key-here
```

#### Google Cloud Storage
Edit `.env.gcs`:
```env
STORAGE_BUCKET=your-gcs-bucket-name
# If not using IAM:
# GCS_CREDENTIALS_JSON={"type":"service_account",...}
```

#### MinIO (Local)
The default MinIO configuration in `.env.minio` works out of the box.

## Architecture

The optimizer creates a distributed Milvus cluster with:

- **Coordinator Nodes**: RootCoord, DataCoord, QueryCoord, IndexCoord
- **Data Nodes**: Scaled based on CPU cores (handles data ingestion)
- **Index Nodes**: Scaled based on GPUs (builds vector indexes)
- **Query Nodes**: Scaled based on memory and GPUs (handles searches)
- **Storage**: Etcd (metadata), Pulsar (messaging), Object Storage (data)

## Optimization Details

### Auto Mode Scaling

- **Index Nodes**: Up to 8 per GPU (GPU mode) or CPU_CORES/2 (CPU mode)
- **Data Nodes**: Up to CPU_CORES nodes (max 32)
- **Query Nodes**: 2 per GPU or CPU_CORES/4
- **Memory**: 90% of system RAM distributed across nodes
- **Parallelism**: Maximum based on available resources

### Target Mode Scaling

Calculates optimal resources based on embedding count:
- Estimates memory requirements (embedding_size × 1.5 overhead)
- Scales nodes proportionally to dataset size
- Adjusts segment sizes for optimal performance
- Configures thread pools based on workload

### Bulk Load Optimizations

- **Large Segments**: 2GB segment size for fewer files
- **High Seal Proportion**: 90% to reduce segment count
- **Disabled Auto-Compaction**: Prevents interference during load
- **Increased Buffers**: Larger insert buffers for throughput
- **Parallel Operations**: Maximum parallelism for all operations
- **Optimized Thread Pools**: Aggressive thread allocation

## Command Line Options

```bash
./scripts/run_milvus.sh [OPTIONS]

Options:
  --env TYPE             Storage environment: infinia, gcs, or minio (default: minio)
  --gpu                  Force GPU mode
  --cpu                  Force CPU mode
  --target-embeddings N  Optimize for specific number of embeddings (e.g., 20M, 1.5B)
  --dry-run             Show configuration without starting services
  --stop                Stop all services
  --clean               Stop services and clean data volumes
  --logs                Show logs from all services
  --help                Show help message

Examples:
  ./scripts/run_milvus.sh --env infinia --gpu               # GPU mode with Infinia storage
  ./scripts/run_milvus.sh --env gcs --cpu                   # CPU mode with GCS storage
  ./scripts/run_milvus.sh --env minio                       # Auto-detect GPU with MinIO
  ./scripts/run_milvus.sh --env infinia --target-embeddings 20M  # Optimize for 20M embeddings
```

## Monitoring

The deployment includes Prometheus and Grafana for monitoring:

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **Milvus API**: localhost:19530

## Python API Usage

You can also use the optimizer programmatically:

```python
from scripts.milvus_optimizer import MilvusOptimizer

# Create optimizer
optimizer = MilvusOptimizer()

# Generate configuration for 50M embeddings
summary = optimizer.generate_configuration(target_embeddings=50_000_000)

# Configuration files are generated in the current directory
```

## Performance Expectations

Based on system resources, expected bulk load performance:

- **32 CPU cores, 128GB RAM, 4 GPUs**: 
  - Auto mode: 32 data nodes, 32 index nodes, 8 query nodes
  - Can handle 100M+ embeddings efficiently
  - Parallel ingestion across all nodes

- **8 CPU cores, 32GB RAM, 1 GPU**:
  - Auto mode: 8 data nodes, 8 index nodes, 2 query nodes
  - Suitable for 10-50M embeddings
  - Good balance of resources

- **CPU-only deployment**:
  - Scales based on CPU cores
  - Suitable for smaller datasets or when GPUs unavailable

## Troubleshooting

### Out of Memory

If you encounter OOM errors:
1. Use target mode with appropriate embedding count
2. Reduce node counts in generated docker-compose.yml
3. Increase system swap space

### GPU Not Detected

Ensure nvidia-docker is installed:
```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Storage Connection Issues

- **Infinia**: Verify certificate is in `certs/` directory
- **GCS**: Ensure credentials are correct
- **MinIO**: Check if MinIO container is running

### View Logs

```bash
# All services
./scripts/run_milvus.sh --env minio --logs

# Specific service
docker-compose logs -f datanode0
```

## Advanced Configuration

### Custom Resource Limits

Edit generated `docker-compose.yml` to adjust resource limits:

```yaml
deploy:
  resources:
    limits:
      memory: 8192m  # Adjust memory limit
    reservations:
      cpus: '4'      # Reserve CPU cores
```

### Persistent Configuration

To make configuration changes permanent:
1. Generate configuration: `./scripts/run_milvus.sh --env minio --dry-run`
2. Edit `configs/milvus.yaml` as needed
3. Start services: `docker-compose up -d`

## Requirements

- Docker 20.10+
- Docker Compose 2.0+
- Python 3.8+
- nvidia-docker (for GPU support)
- 8GB+ RAM minimum
- 50GB+ free disk space

## License

This optimizer is provided as-is for use with Milvus deployments.