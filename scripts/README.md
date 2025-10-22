# OpenTextShield Scripts

This directory contains deployment and utility scripts for the OpenTextShield platform.

## Deployment Scripts

### Start Scripts
- **`start.sh`** - Main startup script with auto-detection (recommended)
- **`start-local.sh`** - Local development startup with hot reload
- **`start-manual.sh`** - Manual startup for advanced users
- **`start-docker.sh`** - Docker-specific startup script

### Container Management
- **`check_docker_containers.sh`** - Check status of running Docker containers
- **`launch_multiple_containers.sh`** - Launch multiple container instances
- **`test_multiple_containers.sh`** - Test multiple container deployments

## Utility Scripts

### `utils/`
- **`enhanced_preprocessing.py`** - Enhanced text preprocessing utilities

## Usage

### Quick Start
```bash
# Start the platform (recommended)
./scripts/start.sh

# Start for local development
./scripts/start-local.sh
```

### Docker Operations
```bash
# Check container status
./scripts/check_docker_containers.sh

# Launch multiple instances
./scripts/launch_multiple_containers.sh
```

## Notes

- All scripts should be executable (`chmod +x script.sh`)
- Scripts are designed to work from any directory
- See individual script headers for detailed usage
