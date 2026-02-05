# BindCraft Docker Setup

This directory contains Docker configuration for running BindCraft on RTX GPUs (RTX 4090, RTX 5080).

## Quick Start

```bash
# 1. Validate setup
bash setup.sh

# 2. Build Docker image (first time only)
make build

# 3. Start API server
make up

# 4. Open browser to http://localhost:8000/docs
```

## API Commands

### Get Info
```bash
curl http://localhost:8000/info
```

### Upload PDB
```bash
curl -X POST -F "file=@target.pdb" http://localhost:8000/upload
```

### Design Peptide
```bash
curl -X POST http://localhost:8000/design \
  -H "Content-Type: application/json" \
  -d '{
    "target_pdb_file": "target.pdb",
    "binder_name": "binder_1",
    "target_chains": "A",
    "target_hotspot": "1-20",
    "binder_lengths": "35-50",
    "num_designs": 100
  }'
```

Returns: `{"job_id": "design_20250118_120000", ...}`

### Check Job Status
```bash
curl http://localhost:8000/jobs/design_20250118_120000
```

### Download Results
```bash
curl http://localhost:8000/jobs/design_20250118_120000/download -o results.zip
```

## Make Commands

| Command | Description |
|---------|-------------|
| `make build` | Build Docker image |
| `make build-prod` | Build optimized production image |
| `make up` | Start all services |
| `make down` | Stop services |
| `make restart` | Restart services |
| `make logs` | View API logs |
| `make shell` | Open shell in container |
| `make gpu-monitor` | Monitor GPU usage |
| `make clean` | Remove containers & cache |
| `make test` | Run tests |
| `make bench` | Performance benchmark |

## Configuration

### GPU Selection
Edit `docker-compose.yml` line 16:
```yaml
device_ids: ['0']  # Change to ['0', '1'] for both GPUs
```

### Memory Optimization
Environment variables in `docker-compose.yml`:
```yaml
environment:
  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  - CUDA_VISIBLE_DEVICES=0
```

### Model Cache
Pre-download weights to avoid first-run delay:
```bash
mkdir -p weights
# Inside container, models will cache to /weights
```

## Data Directories

```
data/
├── inputs/     ← Upload PDB files here
├── outputs/    ← Results saved here
└── pdbs/       ← PDB cache
```

## Troubleshooting

### API not accessible
```bash
# Check if service is running
docker-compose ps

# View logs
docker-compose logs bindcraft-api

# Restart
docker-compose restart
```

### GPU not detected
```bash
# Verify NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:12.4-runtime nvidia-smi

# Check system
nvidia-smi
```

### Out of Memory
```bash
# Reduce batch size or enable NVMe offloading
# See PYTORCH_CUDA_ALLOC_CONF in docker-compose.yml
```

## Performance

**RTX 4090 (24GB):**
- ~50s per design
- ~72 designs/hour
- Batch size: 4-8 per GPU

**RTX 5080 (16GB):**
- ~60s per design
- ~60 designs/hour
- Batch size: 2-4 per GPU

## Full Documentation

See main [README.md](README.md) for complete documentation.

---

**Note:** First API request may take longer as models are loaded into GPU memory.
