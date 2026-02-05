# BindCraft Docker Pipeline - Build Status

**Last Update**: 2026-01-23 00:35 UTC

## Current Status: 🔨 Building

Docker image rebuild in progress with JAX CUDA 12.1 support fix.

### What's Done ✅

1. **Infrastructure**
   - ✅ Docker container with GPU support (RTX 5080 + RTX 4090)
   - ✅ Launch scripts (bash `launch_pipeline.sh` + Python `interactive_launcher.py`)
   - ✅ Configuration system (3 levels: target/algorithm/filters)
   - ✅ Example configs ready (GLP1_FullPipeline + QuickTest)
   - ✅ PyTorch, ColabDesign, ColabFold, PyRosetta installed

2. **Documentation**
   - ✅ LAUNCH_MODES.md - Complete user guide
   - ✅ Launch scripts with auto-results copying to Desktop
   - ✅ Config templates with examples

### Current Issue & Fix 🔧

**Problem**: ColabDesign initialization fails in Docker
- Error: "No GPU device found, terminating"
- Root cause: jaxlib was CPU-only (built without CUDA)
- Impact: Pipeline can't even import ColabDesign

**Solution Applied**:
- Changed Dockerfile.bindcraft to install jaxlib via conda with CUDA 12.1 support
- Command: `conda install -y -c conda-forge jax jaxlib cuda-version=12.1`
- This ensures jaxlib has GPU bindings

### Build Process

Currently rebuilding Docker image with:
- ✅ Ubuntu 22.04 base
- ✅ Python 3.10
- ✅ PyTorch + CUDA 12.1 (GPU support)
- 🔄 JAX + jaxlib CUDA (building now...)
- ✅ ColabDesign, ColabFold
- ✅ PyRosetta
- ✅ AF2 weights (~5.3GB)

**Estimated time**: 15-20 minutes

### Next Steps (After Build)

1. Start container: `docker compose up -d`
2. Test quick pipeline: `./launch_pipeline.sh settings_target/QuickTest_ShortPeptide.json`
3. Expected: 3 designs in ~5-10 minutes
4. Check results: `results/QuickTest_ShortPeptide/structures/*.pdb`

### Commands to Test

```bash
# Start container
docker compose up -d

# Quick test (5-10 min)
./launch_pipeline.sh settings_target/QuickTest_ShortPeptide.json

# Or interactive mode
python interactive_launcher.py

# Monitor progress
docker exec bindcraft-api tail -f /workspace/BindCraft/logs/pipeline_test.log
```

### Architecture Overview

```
BindCraft Docker Pipeline
├── Docker Container (GPU-enabled)
│   ├── Environment: BindCraft conda env
│   ├── Code: /workspace/BindCraft (mounted)
│   ├── Data: /workspace/BindCraft/results (persistent)
│   └── Weights: /workspace/BindCraft/params (AF2, cached)
│
├── Launch Mechanisms
│   ├── launch_pipeline.sh (bash script)
│   ├── interactive_launcher.py (menu-driven)
│   └── Direct docker exec (manual)
│
├── Configuration System
│   ├── settings_target/*.json (PDB, chain, hotspots, lengths)
│   ├── settings_advanced/*.json (algorithm, 2/3/4-stage)
│   └── settings_filters/*.json (quality thresholds)
│
└── Pipeline Flow
    ├── ColabDesign (AF2 backprop) → sequences
    ├── ProteinMPNN (sequence design) → variants
    ├── PyRosetta (structure relax) → refined PDB
    └── Results → /results/structures/*.pdb + metrics
```

### Known Limitations

- JAX CUDA support depends on proper jaxlib installation (being fixed)
- AF2 weights are large (~5.3GB) - first download may take time
- PyRosetta requires license (free academic license accepted)

### Success Criteria

✅ Pipeline runs end-to-end without errors
✅ Generates realistic PDB structures (not templates)
✅ Produces metrics in final_design_stats.csv
✅ Results accessible both in container and on Windows Desktop

---

**Goal**: Make BindCraft work exactly like the original pipeline, but with Docker convenience.
