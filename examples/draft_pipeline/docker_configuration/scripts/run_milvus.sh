#!/bin/bash
# Milvus Runner with Dynamic Configuration
# Supports multiple storage backends and GPU/CPU modes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# Default values
ENV_TYPE="minio"
USE_GPU=""
TARGET_EMBEDDINGS=""
DRY_RUN=false
STOP=false
CLEAN=false
SHOW_LOGS=false
FORCE_CONFIG=false
CLEAN_ALL=false

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --env TYPE             Storage environment: infinia, gcs, or minio (default: minio)"
    echo "  --gpu                  Force GPU mode"
    echo "  --cpu                  Force CPU mode"
    echo "  --target-embeddings N  Optimize for specific number of embeddings (e.g., 20M, 1.5B)"
    echo "  --dry-run             Show configuration without starting services"
    echo "  --stop                Stop all services"
    echo "  --clean               Stop services and clean data volumes"
    echo "  --logs                Show logs from all services"
    echo "  --force-config        Force regeneration of configuration files"
    echo "  --clean-all           Stop ALL Docker containers (not just Milvus)"
    echo "  --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --env infinia --gpu               # GPU mode with Infinia storage"
    echo "  $0 --env gcs --cpu                   # CPU mode with GCS storage"
    echo "  $0 --env minio                       # Auto-detect GPU with MinIO"
    echo "  $0 --env infinia --target-embeddings 20M  # Optimize for 20M embeddings"
}

# Check Docker Compose command
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null; then
        if ! docker compose version &> /dev/null; then
            print_error "Docker Compose is not installed!"
            exit 1
        fi
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi
}

# Function to check dependencies
check_dependencies() {
    print_info "Checking dependencies..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed!"
        exit 1
    fi
    
    # Check Docker Compose
    check_docker_compose
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed!"
        exit 1
    fi
}

# Function to detect GPU
detect_gpu() {
    if nvidia-smi &> /dev/null; then
        if docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null 2>&1; then
            return 0  # GPU available
        fi
    fi
    return 1  # No GPU
}

# Function to clear storage-related environment variables
clear_storage_env() {
    print_info "Clearing previous storage configuration..."
    unset STORAGE_TYPE STORAGE_ENDPOINT STORAGE_PORT STORAGE_BUCKET
    unset STORAGE_ACCESS_KEY STORAGE_SECRET_KEY USE_SSL
    unset MINIO_ADDRESS MINIO_BUCKET_NAME MINIO_ACCESS_KEY_ID
    unset MINIO_SECRET_ACCESS_KEY MINIO_USE_SSL MINIO_ROOT_PATH
    unset MINIO_CLOUD_PROVIDER MINIO_USE_IAM MINIO_GCP_CREDENTIAL_JSON
    unset GCS_CREDENTIALS_JSON GCP_PROJECT USE_IAM
    unset STORAGE_REGION STORAGE_ROOT_PATH
}

# Function to load environment
load_environment() {
    local env_file="$BASE_DIR/.env.$ENV_TYPE"
    
    if [ ! -f "$env_file" ]; then
        print_error "Environment file not found: $env_file"
        print_info "Available environments: infinia, gcs, minio"
        exit 1
    fi
    
    # Clear any existing storage-related environment variables first
    clear_storage_env
    
    print_info "Loading environment from .env.$ENV_TYPE"
    
    # Load environment file
    set -a  # Export all variables
    source "$env_file"
    set +a
    
    # Handle GPU/CPU mode
    if [ -n "$USE_GPU" ]; then
        if [ "$USE_GPU" = "true" ]; then
            export MILVUS_IMAGE="$MILVUS_IMAGE_GPU"
            print_info "Using GPU mode: $MILVUS_IMAGE"
        else
            export MILVUS_IMAGE="$MILVUS_IMAGE_CPU"
            print_info "Using CPU mode: $MILVUS_IMAGE"
        fi
    else
        # Auto-detect
        if detect_gpu; then
            export MILVUS_IMAGE="$MILVUS_IMAGE_GPU"
            print_info "GPU detected, using: $MILVUS_IMAGE"
        else
            export MILVUS_IMAGE="$MILVUS_IMAGE_CPU"
            print_info "No GPU detected, using: $MILVUS_IMAGE"
        fi
    fi
    
    # Set default volume directory if not set
    export DOCKER_VOLUME_DIRECTORY=${DOCKER_VOLUME_DIRECTORY:-$BASE_DIR/volumes}
}

# Function to install Python dependencies
install_dependencies() {
    if ! python3 -c "import psutil" 2>/dev/null; then
        print_info "Installing Python dependencies..."
        cd "$BASE_DIR"
        pip3 install -q -r requirements.txt
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENV_TYPE="$2"
            shift 2
            ;;
        --gpu)
            USE_GPU="true"
            shift
            ;;
        --cpu)
            USE_GPU="false"
            shift
            ;;
        --target-embeddings)
            TARGET_EMBEDDINGS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --stop)
            STOP=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --logs)
            SHOW_LOGS=true
            shift
            ;;
        --force-config)
            FORCE_CONFIG=true
            shift
            ;;
        --clean-all)
            CLEAN_ALL=true
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main execution
cd "$BASE_DIR"

# Check dependencies
check_dependencies

# Load environment
load_environment

# Function to stop all containers (more aggressive cleanup)
stop_all_containers() {
    print_warn "Stopping ALL Docker containers..."
    docker stop $(docker ps -q) 2>/dev/null || true
    print_info "Removing all stopped containers..."
    docker rm $(docker ps -a -q) 2>/dev/null || true
    print_info "All containers stopped and removed"
}

# Function to check for existing Milvus services
check_existing_services() {
    # Check both running and stopped containers
    local existing_containers=$(docker ps -a --filter "name=milvus" --format "{{.Names}}" 2>/dev/null | wc -l)
    
    if [ "$existing_containers" -gt 0 ]; then
        print_warn "Found $existing_containers existing Milvus containers"
        print_info "Stopping and removing existing containers..."
        
        # Try to stop using docker-compose first if file exists
        if [ -f "$BASE_DIR/docker-compose.yml" ]; then
            $COMPOSE_CMD -f "$BASE_DIR/docker-compose.yml" down -v 2>/dev/null || true
        fi
        
        # Stop and remove any remaining milvus containers
        docker ps -a --filter "name=milvus" --format "{{.ID}}" | xargs -r docker rm -f >/dev/null 2>&1 || true
        
        print_info "Existing containers removed"
        sleep 2
    fi
}

# Handle clean-all command
if [ "$CLEAN_ALL" = true ]; then
    stop_all_containers
    exit 0
fi

# Check and stop existing services unless just showing logs
if [ "$SHOW_LOGS" != true ] && [ "$STOP" != true ] && [ "$CLEAN" != true ]; then
    # Always stop all containers before starting new ones
    stop_all_containers
fi

# Handle stop/clean commands
if [ "$STOP" = true ] || [ "$CLEAN" = true ]; then
    print_info "Stopping Milvus services..."
    $COMPOSE_CMD down
    
    if [ "$CLEAN" = true ]; then
        print_warn "Cleaning data volumes..."
        read -p "This will delete all data. Are you sure? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$DOCKER_VOLUME_DIRECTORY"
            print_info "Data volumes cleaned"
        fi
    fi
    exit 0
fi

# Handle logs command
if [ "$SHOW_LOGS" = true ]; then
    $COMPOSE_CMD logs -f
    exit 0
fi

# Install Python dependencies if needed
install_dependencies

# Function to generate config hash for caching
generate_config_hash() {
    # Include more details in hash to ensure proper regeneration
    local storage_details="${STORAGE_TYPE}_${STORAGE_ENDPOINT}_${STORAGE_BUCKET}_${USE_SSL}"
    local hash_input="${storage_details}_${MILVUS_IMAGE}_${TARGET_EMBEDDINGS}_${USE_GPU}"
    echo -n "$hash_input" | md5sum | cut -d' ' -f1
}

# Check if we need to regenerate configuration
CONFIG_HASH=$(generate_config_hash)
CONFIG_CACHE_FILE="$BASE_DIR/.config_cache_${CONFIG_HASH}"
COMPOSE_FILE="$BASE_DIR/docker-compose.yml"
MILVUS_CONFIG="$BASE_DIR/configs/milvus.yaml"
STORAGE_MARKER_FILE="$BASE_DIR/.current_storage_type"

need_generate_config=false

# Check if storage type has changed
if [ -f "$STORAGE_MARKER_FILE" ]; then
    PREVIOUS_STORAGE=$(cat "$STORAGE_MARKER_FILE")
    if [ "$PREVIOUS_STORAGE" != "$STORAGE_TYPE" ]; then
        print_warn "Storage type changed from $PREVIOUS_STORAGE to $STORAGE_TYPE"
        need_generate_config=true
        # Clean up old cache files when switching storage
        rm -f "$BASE_DIR"/.config_cache_* 2>/dev/null || true
    fi
fi

# Check if force config or config files don't exist
if [ "$FORCE_CONFIG" = true ]; then
    print_info "Forcing configuration regeneration (--force-config specified)"
    need_generate_config=true
elif [ ! -f "$CONFIG_CACHE_FILE" ] || [ ! -f "$COMPOSE_FILE" ] || [ ! -f "$MILVUS_CONFIG" ]; then
    print_info "Configuration files missing, generating new configuration..."
    need_generate_config=true
else
    # Check if cache is older than 24 hours
    if [ $(find "$CONFIG_CACHE_FILE" -mtime +1 2>/dev/null | wc -l) -gt 0 ]; then
        print_info "Configuration cache is older than 24 hours, regenerating..."
        need_generate_config=true
    else
        print_info "Using cached configuration (use --force-config to regenerate)"
    fi
fi

if [ "$need_generate_config" = true ]; then
    print_info "Generating optimized configuration..."
    print_info "Storage backend: $STORAGE_TYPE"
    print_info "Milvus image: $MILVUS_IMAGE"

    # Update Python path to include modules directory
    export PYTHONPATH="$BASE_DIR/modules:$PYTHONPATH"

    # Run optimizer with environment file
    OPTIMIZER_ARGS=""
    if [ -n "$TARGET_EMBEDDINGS" ]; then
        OPTIMIZER_ARGS="--target-embeddings $TARGET_EMBEDDINGS"
    fi

    # Pass CPU mode flag to optimizer
    if [ "$USE_GPU" = "false" ]; then
        export FORCE_CPU_MODE="true"
    fi

    if [ "$DRY_RUN" = true ]; then
        OPTIMIZER_ARGS="$OPTIMIZER_ARGS --dry-run"
        python3 "$SCRIPT_DIR/milvus_optimizer.py" $OPTIMIZER_ARGS
        exit 0
    fi

    # Run optimizer
    python3 "$SCRIPT_DIR/milvus_optimizer.py" $OPTIMIZER_ARGS
    
    # Create cache file on successful generation
    if [ $? -eq 0 ]; then
        touch "$CONFIG_CACHE_FILE"
        # Save current storage type
        echo "$STORAGE_TYPE" > "$STORAGE_MARKER_FILE"
        # Clean up old cache files
        find "$BASE_DIR" -name ".config_cache_*" -mtime +7 -delete 2>/dev/null || true
    else
        print_error "Configuration generation failed"
        exit 1
    fi
else
    if [ "$DRY_RUN" = true ]; then
        print_info "Using cached configuration:"
        if [ -f "$BASE_DIR/optimization-summary.json" ]; then
            cat "$BASE_DIR/optimization-summary.json"
        fi
        exit 0
    fi
fi

# Create necessary directories
print_info "Creating volume directories..."
mkdir -p "$DOCKER_VOLUME_DIRECTORY"/{milvus,etcd,pulsar_data,pulsar_logs,prometheus,grafana}

if [ "$STORAGE_TYPE" = "minio" ]; then
    mkdir -p "$DOCKER_VOLUME_DIRECTORY/minio"
fi

# Start services
print_info "Starting Milvus services..."
$COMPOSE_CMD up -d

# Wait for services to be ready
print_info "Waiting for services to be ready..."
sleep 10

# Check service health
print_info "Checking service health..."
$COMPOSE_CMD ps

# Show connection info
echo ""
print_info "Milvus cluster started successfully!"
echo ""
echo "Configuration:"
echo "  - Storage: $STORAGE_TYPE"
echo "  - Mode: $([ "$USE_GPU" = "false" ] && echo "CPU" || echo "GPU")"
echo ""
echo "Connection endpoints:"
echo "  - Milvus API: localhost:19530"
echo "  - Prometheus: http://localhost:9090"
echo "  - Grafana: http://localhost:3000 (admin/admin)"
if [ "$STORAGE_TYPE" = "minio" ]; then
    echo "  - MinIO Console: http://localhost:9001"
fi
echo ""
echo "Commands:"
echo "  View logs:    $0 --env $ENV_TYPE --logs"
echo "  Stop:         $0 --env $ENV_TYPE --stop"
echo "  Clean data:   $0 --env $ENV_TYPE --clean"