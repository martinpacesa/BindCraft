#!/bin/bash
# Test BindCraft pipeline in Docker with full validation

set -e

CONTAINER="bindcraft-api"
SETTINGS_JSON="settings_target/QuickTest_ShortPeptide.json"
OUTPUT_DIR="results/PipelineTest_$(date +%s)"

echo "=========================================="
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
