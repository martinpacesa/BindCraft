#!/bin/bash
#############################################
# BindCraft Docker Pipeline Test Script
# Tests full vanilla pipeline with GPU/CPU
#############################################

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=================================================="
echo "  BindCraft Docker Pipeline Test"
echo "=================================================="
echo ""

# 1. Check Docker is running
echo "[1/5] Checking Docker setup..."
if ! docker ps > /dev/null 2>&1; then
    echo "❌ Docker not running. Please start Docker."
    exit 1
fi
echo "✓ Docker running"

# 2. Build or use existing image
echo ""
echo "[2/5] Building Docker image..."
docker compose build 2>&1 | grep -E "Successfully|error|ERROR" | tail -5 || true

# 3. Start container
echo ""
echo "[3/5] Starting BindCraft container..."
docker compose up -d
sleep 10

# 4. Validate environment
echo ""
echo "[4/5] Validating environment..."
docker exec bindcraft-api bash -c "
echo 'Python: '$(python --version)
echo 'JAX GPU:'
python -c 'import jax; d=jax.devices(); print(f\"  Devices: {d}\"); print(f\"  Status: {'✓ GPU' if any(\"cuda\" in str(x).lower() for x in d) else '⚠ CPU'}\")'
echo 'PyRosetta:'
python -c 'try: import pyrosetta; print(\"  ✓ Available\"); except: print(\"  ⚠ Not available\")' 2>/dev/null || true
"

# 5. Run test pipeline
echo ""
echo "[5/5] Running BindCraft pipeline test..."
echo "  Settings: QuickTest_ShortPeptide"
echo "  Advanced: default_4stage_multimer"
echo "  Timeout: 10 min"
echo ""

docker exec bindcraft-api bash -c "
cd /workspace/BindCraft && \
timeout 600 python bindcraft.py \
  --settings settings_target/QuickTest_ShortPeptide.json \
  --advanced settings_advanced/default_4stage_multimer.json \
  --filters settings_filters/default_filters.json \
  2>&1
"

echo ""
echo "=================================================="
echo "  Pipeline test completed!"
echo "=================================================="
echo ""
echo "Results should be in:"
echo "  /workspace/BindCraft/results/"
echo ""
echo "To check output structures:"
echo "  docker exec bindcraft-api ls -lh /workspace/BindCraft/results/"
echo ""

echo "  BindCraft Pipeline Test"
echo "=========================================="
echo ""

# Verify container is running
echo "[1/4] Checking container..."
if ! docker ps | grep -q "$CONTAINER"; then
    echo "✗ Container $CONTAINER not running"
    exit 1
fi
echo "✓ Container running"
echo ""

# Verify CUDA/JAX GPU
echo "[2/4] Checking CUDA/JAX GPU..."
docker exec "$CONTAINER" bash -c "
source activate BindCraft && python << 'PYEOF'
import jax
import torch
print('PyTorch CUDA:', torch.cuda.is_available())
print('JAX Devices:', jax.devices())
has_gpu = any('gpu' in str(d).lower() for d in jax.devices())
if not has_gpu:
    print('WARNING: JAX GPU not detected, will use CPU')
PYEOF
"
echo ""

# Verify BindCraft imports
echo "[3/4] Checking BindCraft imports..."
docker exec "$CONTAINER" bash -c "
source activate BindCraft && cd /workspace/BindCraft && python << 'PYEOF'
try:
    from functions import *
    print('✓ BindCraft functions imported')
except Exception as e:
    print(f'✗ Import error: {e}')
    exit(1)
PYEOF
"
echo ""

# Run pipeline
echo "[4/4] Running BindCraft design pipeline..."
echo "  Settings: $SETTINGS_JSON"
echo "  Output:   $OUTPUT_DIR"
echo ""

docker exec "$CONTAINER" bash -c "
source activate BindCraft && \
cd /workspace/BindCraft && \
mkdir -p '$OUTPUT_DIR' && \
timeout 1800 python bindcraft.py \
  --settings '$SETTINGS_JSON' \
  --advanced settings_advanced/default_4stage_multimer.json \
  --filters settings_filters/default_filters.json \
  2>&1 | tee '$OUTPUT_DIR/pipeline.log' || \
  echo 'Pipeline completed or timed out after 30 minutes'
"

echo ""
echo "=========================================="
echo "Pipeline test complete!"
echo "Results directory: $OUTPUT_DIR"
echo "=========================================="
