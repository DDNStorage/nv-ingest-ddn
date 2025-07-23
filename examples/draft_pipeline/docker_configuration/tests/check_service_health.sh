#!/bin/bash
# Service Health Check Script for Milvus Docker Deployment
# Checks the health of all Milvus services and ensures CPU-only operation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Function to check if a container is running
check_container() {
    local container_name=$1
    if docker ps --format "{{.Names}}" | grep -q "^${container_name}$"; then
        return 0
    else
        return 1
    fi
}

# Function to get container status
get_container_status() {
    local container_name=$1
    docker ps -a --filter "name=^${container_name}$" --format "{{.Status}}" 2>/dev/null || echo "Not found"
}

# Function to check if container is using GPU
check_gpu_usage() {
    local container_name=$1
    local image=$(docker inspect --format='{{.Config.Image}}' "$container_name" 2>/dev/null || echo "")
    if [[ "$image" == *"-gpu"* ]]; then
        return 0  # GPU image detected
    else
        return 1  # CPU image
    fi
}

echo "================================================"
echo "   Milvus Docker Service Health Check"
echo "================================================"
echo ""

# 1. Check Docker daemon
print_info "Checking Docker daemon..."
if docker info >/dev/null 2>&1; then
    print_success "Docker daemon is running"
else
    print_error "Docker daemon is not running!"
    exit 1
fi

echo ""

# 2. Check all Milvus containers
print_info "Checking Milvus containers..."
echo ""

# Define expected containers
CORE_CONTAINERS=(
    "milvus-etcd"
    "milvus-pulsar"
    "milvus-rootcoord"
    "milvus-proxy"
    "milvus-querycoord"
    "milvus-datacoord"
    "milvus-indexcoord"
)

# Data nodes (0-15)
DATA_NODES=()
for i in {0..15}; do
    DATA_NODES+=("milvus-datanode${i}")
done

# Query nodes (0-3)
QUERY_NODES=()
for i in {0..3}; do
    QUERY_NODES+=("milvus-querynode${i}")
done

# Index nodes (0-7)
INDEX_NODES=()
for i in {0..7}; do
    INDEX_NODES+=("milvus-indexnode${i}")
done

# Monitoring containers (optional)
MONITORING_CONTAINERS=(
    "milvus-prometheus"
    "milvus-grafana"
)

# Check core containers
echo "Core Services:"
all_core_running=true
for container in "${CORE_CONTAINERS[@]}"; do
    if check_container "$container"; then
        status=$(get_container_status "$container")
        print_success "$container: $status"
    else
        print_error "$container: NOT RUNNING"
        all_core_running=false
    fi
done

echo ""

# Check data nodes
echo "Data Nodes:"
data_nodes_running=0
for container in "${DATA_NODES[@]}"; do
    if check_container "$container"; then
        ((data_nodes_running++))
    fi
done
print_info "Running: $data_nodes_running/${#DATA_NODES[@]}"

# Check query nodes
echo ""
echo "Query Nodes:"
query_nodes_running=0
for container in "${QUERY_NODES[@]}"; do
    if check_container "$container"; then
        ((query_nodes_running++))
    fi
done
print_info "Running: $query_nodes_running/${#QUERY_NODES[@]}"

# Check index nodes
echo ""
echo "Index Nodes:"
index_nodes_running=0
for container in "${INDEX_NODES[@]}"; do
    if check_container "$container"; then
        ((index_nodes_running++))
    fi
done
print_info "Running: $index_nodes_running/${#INDEX_NODES[@]}"

echo ""

# 3. Check for GPU usage
print_info "Checking for GPU usage..."
gpu_containers_found=0
for container in $(docker ps --format "{{.Names}}" | grep "^milvus-"); do
    if check_gpu_usage "$container"; then
        print_warn "GPU image detected: $container"
        ((gpu_containers_found++))
    fi
done

if [ $gpu_containers_found -eq 0 ]; then
    print_success "All containers are using CPU-only images"
else
    print_warn "Found $gpu_containers_found containers using GPU images"
fi

echo ""

# 4. Check service endpoints
print_info "Checking service endpoints..."
echo ""

# Check etcd health
echo -n "etcd health: "
if curl -s http://localhost:2379/health >/dev/null 2>&1; then
    print_success "Healthy"
else
    print_error "Unreachable"
fi

# Check Pulsar health
echo -n "Pulsar health: "
if docker exec milvus-pulsar bin/pulsar-admin brokers healthcheck 2>/dev/null | grep -q "ok"; then
    print_success "Healthy"
else
    print_error "Unhealthy or unreachable"
fi

# Check Milvus proxy port
echo -n "Milvus proxy (19530): "
if nc -z localhost 19530 2>/dev/null; then
    print_success "Port open"
else
    print_error "Port closed"
fi

# Check Prometheus (optional)
echo -n "Prometheus (9090): "
if nc -z localhost 9090 2>/dev/null; then
    print_success "Port open"
else
    print_warn "Not available (optional)"
fi

# Check Grafana (optional)
echo -n "Grafana (3000): "
if nc -z localhost 3000 2>/dev/null; then
    print_success "Port open"
else
    print_warn "Not available (optional)"
fi

echo ""

# 5. Check resource usage
print_info "Checking resource usage (CPU only)..."
echo ""

# Get Docker stats without streaming
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" | grep "milvus-" | head -10 || true

echo ""

# 6. Check for container restarts
print_info "Checking for container restarts..."
restart_found=false
for container in $(docker ps --format "{{.Names}}" | grep "^milvus-"); do
    restart_count=$(docker inspect -f '{{.RestartCount}}' "$container" 2>/dev/null || echo "0")
    if [ "$restart_count" -gt 0 ]; then
        print_warn "$container has restarted $restart_count times"
        restart_found=true
    fi
done

if [ "$restart_found" = false ]; then
    print_success "No container restarts detected"
fi

echo ""

# 7. Summary
echo "================================================"
echo "                  SUMMARY"
echo "================================================"

if [ "$all_core_running" = true ]; then
    print_success "All core services are running"
else
    print_error "Some core services are not running"
fi

if [ $data_nodes_running -gt 0 ]; then
    print_success "$data_nodes_running data nodes are running"
else
    print_error "No data nodes are running"
fi

if [ $gpu_containers_found -eq 0 ]; then
    print_success "CPU-only mode confirmed"
else
    print_error "GPU containers detected - not in CPU-only mode"
fi

echo ""

# Exit with appropriate code
if [ "$all_core_running" = true ] && [ $data_nodes_running -gt 0 ]; then
    exit 0
else
    exit 1
fi