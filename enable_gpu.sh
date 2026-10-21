#!/bin/bash
# Enable JAX GPU support by installing CUDA runtime in container
# Run this AFTER docker compose up

CONTAINER="bindcraft-api"

echo "Installing JAX CUDA GPU plugin..."
docker exec "$CONTAINER" bash -c "
source activate BindCraft && \
pip install --no-cache-dir 'jax-cuda12-pjrt' 2>&1 | grep -E '(Successfully|ERROR)' || echo 'CUDA plugin attempt done (may fail in container)'
"

echo ""
echo "Testing JAX devices..."
docker exec "$CONTAINER" bash -c "
source activate BindCraft && python << 'PYEOF'
import jax
print('JAX Devices:', jax.devices())
gpu = any('gpu' in str(d).lower() for d in jax.devices())
print(f'GPU Available: {\"YES\" if gpu else \"NO (using CPU)\"}')
PYEOF
"
