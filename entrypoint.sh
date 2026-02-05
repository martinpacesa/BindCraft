#!/bin/bash
# BindCraft Docker entrypoint script
# Supports multiple modes: api, inference, batch

set -e

# Activate conda environment
source /opt/conda/bin/activate bindcraft

# Setup environment
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Create data directories
mkdir -p /data/inputs /data/outputs /data/pdbs

# Log startup
echo "=========================================="
echo "🧬 BindCraft Container Started"
echo "=========================================="
echo "Mode: ${1:-api}"
echo "CUDA Devices: $CUDA_VISIBLE_DEVICES"
echo "Workspace: /workspace"
echo "Data: /data"
echo "=========================================="
echo ""

case "${1:-api}" in
    # Start API server
    api)
        echo "🚀 Starting FastAPI server..."
        exec uvicorn api_server:app \
            --host 0.0.0.0 \
            --port 8000 \
            --log-level info \
            --access-log \
            --reload
        ;;
    
    # Run single inference
    inference)
        echo "🧪 Running inference pipeline..."
        exec python inference_pipeline.py \
            --target "${2:-/data/inputs/target.pdb}" \
            --binder-name "${3:-binder}" \
            --output /data/outputs
        ;;
    
    # Run batch processing
    batch)
        echo "📦 Running batch processing..."
        if [ -f "${2:-/data/inputs/batch.json}" ]; then
            exec python -c "
import json
from inference_pipeline import BindCraftInference
from pathlib import Path

with open('${2:-/data/inputs/batch.json}') as f:
    batch_spec = json.load(f)

pipeline = BindCraftInference()
results = pipeline.batch_design(batch_spec['targets'])

output_file = Path('/data/outputs/batch_results.json')
output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f'✅ Batch completed: {output_file}')
"
        else
            echo "❌ Batch file not found: ${2:-/data/inputs/batch.json}"
            exit 1
        fi
        ;;
    
    # Interactive bash
    bash|shell)
        echo "📍 Entering interactive shell..."
        exec bash -i
        ;;
    
    # Run custom command
    *)
        echo "🔧 Running custom command: $@"
        exec "$@"
        ;;
esac
