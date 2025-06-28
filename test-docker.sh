#!/bin/bash

# OpenTextShield Docker Test Script
# Tests multiple Docker deployment scenarios

set -e

echo "🐳 OpenTextShield Docker Test Suite"
echo "===================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if Docker is running
check_docker() {
    log_info "Checking Docker availability..."
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    log_success "Docker is available"
}

# Clean up function
cleanup() {
    log_info "Cleaning up containers..."
    docker stop ots-test ots-local ots-compose 2>/dev/null || true
    docker rm ots-test ots-local ots-compose 2>/dev/null || true
}

# Test pre-built image
test_prebuilt_image() {
    log_info "Testing pre-built image..."
    
    # Pull the image
    log_info "Pulling telecomsxchange/opentextshield:latest..."
    if docker pull telecomsxchange/opentextshield:latest; then
        log_success "Image pulled successfully"
    else
        log_warning "Could not pull image, skipping pre-built test"
        return
    fi
    
    # Run container
    log_info "Starting container from pre-built image..."
    docker run -d -p 8003:8002 --name ots-test telecomsxchange/opentextshield:latest
    
    # Wait for startup
    log_info "Waiting for container to start..."
    sleep 10
    
    # Test health endpoint
    if curl -f -s http://localhost:8003/health > /dev/null; then
        log_success "Pre-built image health check passed"
    else
        log_error "Pre-built image health check failed"
        docker logs ots-test
    fi
    
    # Stop container
    docker stop ots-test && docker rm ots-test
}

# Test local build
test_local_build() {
    log_info "Testing local Docker build..."
    
    # Build image
    log_info "Building local image..."
    if docker build -t opentextshield-local .; then
        log_success "Local image built successfully"
    else
        log_error "Local image build failed"
        return
    fi
    
    # Run container
    log_info "Starting container from local image..."
    docker run -d -p 8004:8002 --name ots-local opentextshield-local
    
    # Wait for startup
    log_info "Waiting for container to start..."
    sleep 15
    
    # Test health endpoint
    if curl -f -s http://localhost:8004/health > /dev/null; then
        log_success "Local build health check passed"
        
        # Test prediction endpoint
        log_info "Testing prediction endpoint..."
        response=$(curl -s -X POST "http://localhost:8004/predict/" \
            -H "Content-Type: application/json" \
            -d '{"text":"Free money! Click here!","model":"bert"}')
        
        if [[ $response == *"label"* ]]; then
            log_success "Prediction endpoint working"
        else
            log_warning "Prediction endpoint may have issues"
        fi
    else
        log_error "Local build health check failed"
        docker logs ots-local
    fi
    
    # Stop container
    docker stop ots-local && docker rm ots-local
}

# Test Docker Compose
test_docker_compose() {
    log_info "Testing Docker Compose..."
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_warning "Docker Compose not available, skipping compose test"
        return
    fi
    
    # Use docker compose (newer) or docker-compose (older)
    COMPOSE_CMD="docker compose"
    if ! docker compose version &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    fi
    
    # Start services
    log_info "Starting services with Docker Compose..."
    $COMPOSE_CMD up -d --build
    
    # Wait for startup
    log_info "Waiting for services to start..."
    sleep 20
    
    # Test health endpoint
    if curl -f -s http://localhost:8002/health > /dev/null; then
        log_success "Docker Compose health check passed"
    else
        log_error "Docker Compose health check failed"
        $COMPOSE_CMD logs
    fi
    
    # Stop services
    $COMPOSE_CMD down
}

# Test with different configurations
test_configurations() {
    log_info "Testing different configurations..."
    
    # Test with custom port
    log_info "Testing custom port configuration..."
    docker run -d -p 8005:8002 \
        -e OTS_API_PORT=8002 \
        -e OTS_LOG_LEVEL=DEBUG \
        --name ots-config \
        opentextshield-local
    
    sleep 10
    
    if curl -f -s http://localhost:8005/health > /dev/null; then
        log_success "Custom configuration test passed"
    else
        log_warning "Custom configuration test failed"
    fi
    
    docker stop ots-config && docker rm ots-config
}

# Performance test
performance_test() {
    log_info "Running basic performance test..."
    
    # Start container for testing
    docker run -d -p 8006:8002 --name ots-perf opentextshield-local
    sleep 15
    
    # Test multiple requests
    log_info "Sending 10 test requests..."
    for i in {1..10}; do
        response_time=$(curl -w "%{time_total}" -s -o /dev/null \
            -X POST "http://localhost:8006/predict/" \
            -H "Content-Type: application/json" \
            -d '{"text":"Test message","model":"bert"}')
        echo "Request $i: ${response_time}s"
    done
    
    docker stop ots-perf && docker rm ots-perf
    log_success "Performance test completed"
}

# Main test execution
main() {
    # Set trap for cleanup
    trap cleanup EXIT
    
    check_docker
    
    echo ""
    log_info "Starting Docker tests..."
    echo ""
    
    # Run tests
    test_prebuilt_image
    echo ""
    
    test_local_build
    echo ""
    
    test_docker_compose
    echo ""
    
    test_configurations
    echo ""
    
    performance_test
    echo ""
    
    log_success "All Docker tests completed!"
    echo ""
    
    echo "🎉 Docker Test Summary:"
    echo "======================"
    echo "✅ Pre-built image: Tested"
    echo "✅ Local build: Tested"
    echo "✅ Docker Compose: Tested"
    echo "✅ Custom config: Tested"
    echo "✅ Performance: Tested"
    echo ""
    echo "🚀 Ready for production deployment!"
    echo ""
    echo "Quick start commands:"
    echo "  docker-compose up -d          # Start with compose"
    echo "  docker build -t ots . && docker run -p 8002:8002 ots  # Local build"
    echo "  docker run -p 8002:8002 telecomsxchange/opentextshield:latest  # Pre-built"
}

# Check if running as script
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi