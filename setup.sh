#!/bin/bash
# Quick setup and validation script for BindCraft Docker

set -e

echo "=================================="
echo "🧬 BindCraft Docker Setup"
echo "=================================="
echo ""

# Check Docker
echo "✓ Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker."
    exit 1
fi
echo "  Docker version: $(docker --version)"

# Check Docker Compose
echo "✓ Checking Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose."
    exit 1
fi
echo "  Docker Compose version: $(docker-compose --version)"

# Check NVIDIA Docker
echo "✓ Checking NVIDIA Docker runtime..."
if ! docker run --rm --gpus all nvidia/cuda:12.4-runtime nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA Docker runtime not available."
    echo "  Install with: sudo apt-get install nvidia-docker2"
    exit 1
fi
echo "  NVIDIA Docker runtime: ✓"

# Check GPUs
echo "✓ Checking GPU availability..."
GPU_COUNT=$(docker run --rm --gpus all nvidia/cuda:12.4-runtime nvidia-smi --query-gpu=count --format=csv,noheader | head -1 | tr -d '\n')
if [ "$GPU_COUNT" -lt 1 ]; then
    echo "❌ No NVIDIA GPUs detected."
    exit 1
fi
echo "  GPUs available: $GPU_COUNT"

# Create necessary directories
echo "✓ Creating data directories..."
mkdir -p data/inputs data/outputs data/pdbs weights
echo "  Created: data/{inputs,outputs,pdbs}, weights"

# Verify BindCraft clone
echo "✓ Checking BindCraft repository..."
if [ ! -d "BindCraft" ]; then
    echo "❌ BindCraft not found. Run:"
    echo "  git clone https://github.com/martinpacesa/BindCraft.git"
    exit 1
fi
echo "  BindCraft: ✓"

# Copy docker scripts to BindCraft
echo "✓ Installing Docker support files..."
if [ -d "docker" ]; then
    cp docker/*.py BindCraft/ 2>/dev/null || true
    echo "  Copied scripts to BindCraft/"
fi

# Verify Dockerfile
echo "✓ Checking Dockerfile..."
if [ ! -f "Dockerfile.bindcraft" ]; then
    echo "❌ Dockerfile.bindcraft not found."
    exit 1
fi
echo "  Dockerfile.bindcraft: ✓"

# Verify docker-compose
echo "✓ Checking docker-compose.yml..."
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ docker-compose.yml not found."
    exit 1
fi
echo "  docker-compose.yml: ✓"

echo ""
echo "=================================="
echo "✅ Setup Validation Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Build Docker image:"
echo "   make build"
echo ""
echo "2. Start services:"
echo "   make up"
echo ""
echo "3. Access API:"
echo "   http://localhost:8000"
echo "   http://localhost:8000/docs"
echo ""
echo "Optional: Test GPU setup:"
echo "   make gpu-monitor"
echo ""
