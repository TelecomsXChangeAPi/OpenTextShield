#!/bin/bash
# Check running OpenTextShield Docker containers

echo "🐳 OpenTextShield Docker Container Status"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check if Docker is installed and running
echo -n "Checking Docker installation... "
if ! command -v docker &> /dev/null; then
    echo -e "${RED}FAILED${NC} - Docker not installed"
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi
echo -e "${GREEN}PASSED${NC}"

echo -n "Checking Docker daemon... "
if ! docker info &> /dev/null; then
    echo -e "${RED}FAILED${NC} - Docker daemon not running"
    echo "Please start Docker daemon"
    exit 1
fi
echo -e "${GREEN}PASSED${NC}"

echo ""
echo "📋 Running Containers:"
echo "======================"

# List running containers
running_containers=$(docker ps --filter "ancestor=opentextshield:v2.5" --format "table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}\t{{.Names}}")

if [ -z "$running_containers" ]; then
    echo -e "${YELLOW}No OpenTextShield containers currently running${NC}"
    echo ""
    echo "🚀 To start containers, run:"
    echo "docker-compose up -d"
    echo "# or"
    echo "docker run -d --name ots-primary -p 8002:8002 -p 8080:8080 opentextshield:v2.5"
else
    echo "$running_containers"
fi

echo ""
echo "📋 All Containers (including stopped):"
echo "======================================"

# List all containers
all_containers=$(docker ps -a --filter "ancestor=opentextshield:v2.5" --format "table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}\t{{.Names}}")

if [ -z "$all_containers" ]; then
    echo -e "${YELLOW}No OpenTextShield containers found${NC}"
else
    echo "$all_containers"
fi

echo ""
echo "🔍 Container Health Checks:"
echo "==========================="

# Check each running container
container_ids=$(docker ps --filter "ancestor=opentextshield:v2.5" -q)

if [ -z "$container_ids" ]; then
    echo -e "${YELLOW}No running containers to check${NC}"
else
    for container_id in $container_ids; do
        container_name=$(docker inspect "$container_id" --format '{{.Name}}' | sed 's/\///')
        container_ports=$(docker inspect "$container_id" --format '{{range $p, $conf := .NetworkSettings.Ports}}{{if $conf}}{{"\n  "}}{{$p}} -> {{(index $conf 0).HostPort}}{{end}}{{end}}')

        echo -e "${BLUE}Container: $container_name ($container_id)${NC}"
        echo "Ports:$container_ports"

        # Get the API port
        api_port=$(docker inspect "$container_id" --format '{{(index (index .NetworkSettings.Ports "8002/tcp") 0).HostPort}}' 2>/dev/null)

        if [ -n "$api_port" ]; then
            echo -n "  Health check: "
            if curl -s "http://localhost:$api_port/health" | grep -q "healthy"; then
                echo -e "${GREEN}PASSED${NC}"
            else
                echo -e "${RED}FAILED${NC}"
            fi

            echo -n "  Model loaded: "
            if curl -s "http://localhost:$api_port/health" | grep -q '"mbert_multilingual": true'; then
                echo -e "${GREEN}PASSED${NC}"
            else
                echo -e "${RED}FAILED${NC}"
            fi

            echo -n "  Version check: "
            version=$(curl -s -X POST "http://localhost:$api_port/predict/" \
                -H "Content-Type: application/json" \
                -d '{"text":"test","model":"ots-mbert"}' | jq -r '.model_info.version' 2>/dev/null)

            if [ "$version" = "2.5" ]; then
                echo -e "${GREEN}PASSED${NC} (v$version)"
            else
                echo -e "${RED}FAILED${NC} (reported: v$version)"
            fi
        else
            echo -e "  ${RED}Could not determine API port${NC}"
        fi

        echo ""
    done
fi

echo "📊 Resource Usage:"
echo "=================="

# Show resource usage
if docker ps --filter "ancestor=opentextshield:v2.5" -q | grep -q .; then
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}" $(docker ps --filter "ancestor=opentextshield:v2.5" -q)
else
    echo -e "${YELLOW}No running containers${NC}"
fi

echo ""
echo "🔧 Management Commands:"
echo "======================="
echo "# View detailed logs"
echo "docker logs CONTAINER_NAME"
echo ""
echo "# Restart container"
echo "docker restart CONTAINER_NAME"
echo ""
echo "# Stop all containers"
echo "docker stop \$(docker ps -q --filter ancestor=opentextshield:v2.5)"
echo ""
echo "# Remove stopped containers"
echo "docker rm \$(docker ps -aq --filter ancestor=opentextshield:v2.5)"
echo ""
echo "# Clean up images"
echo "docker image prune -f"