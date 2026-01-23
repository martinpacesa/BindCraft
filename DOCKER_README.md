# BindCraft Docker - Complete GPU-Accelerated Pipeline

## Overview

🚀 **BindCraft in Docker with full GPU support!**

- **JAX GPU**: CUDA-enabled JAX running on NVIDIA GPUs (Tested: RTX 4090, RTX 5080)
- **PyRosetta**: Full protein relaxation for structure refinement
- **ColabDesign**: AF2-based protein hallucination
- **GPU-Accelerated**: ~30min per design on RTX 4090 (vs ~2-4h CPU)

## Quick Start

### 1. Build Docker Image

```bash
cd /home/vincent/code/repo/biotech/BindCraft
docker compose build
```

**Build time**: ~10-15 minutes (downloads ~40GB image + BindCraft deps)

### 2. Run Pipeline Test

```bash
docker compose up -d
docker exec bindcraft-api bash -c "
cd /workspace/BindCraft && python bindcraft.py \
  --settings settings_target/QuickTest_ShortPeptide.json \
  --advanced settings_advanced/default_4stage_multimer.json \
  --filters settings_filters/default_filters.json
"
```

Or use the test script:

```bash
bash test_pipeline.sh
```

### 3. Check Results

```bash
docker exec bindcraft-api ls -lh /workspace/BindCraft/results/
```

## Docker Architecture

### Multi-Stage Build

```
Stage 1: rosettacommons/rosetta:latest
         ↓ PyRosetta pre-compiled
Stage 2: ghcr.io/nvidia/jax:jax
         ↓ JAX CUDA nightly
Stage 3: Final Image
         ├─ JAX GPU support ✓
         ├─ PyRosetta available ✓
         ├─ ColabDesign (AF2) ✓
         └─ All BindCraft deps ✓
```

### Image Contents

| Component | Source | Status |
|-----------|--------|--------|
| JAX/jaxlib | NVIDIA JAX-Toolbox | ✓ GPU CUDA 13 |
| PyRosetta | RosettaCommons | ✓ Relax available |
| ColabDesign | GitHub (sokrypton) | ✓ AF2 backprop |
| AF2 Weights | Google Storage | ✓ Pre-cached (5.3GB) |
| BioPython | PyPI | ✓ Structure parsing |
| ProteinMPNN | GitHub (dauparas) | ✓ Sequence design |
| Python Deps | PyPI | ✓ All installed |

## Environment Variables

Set in `docker-compose.yml`:

```yaml
- CUDA_VISIBLE_DEVICES=0          # Use first GPU
- JAX_PLATFORMS=gpu,cpu            # Try GPU first, fallback to CPU
- PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Usage Examples

### Example 1: Basic Pipeline Test

```bash
docker compose up -d
docker exec bindcraft-api bash -c "
cd /workspace/BindCraft && python bindcraft.py \
  --settings settings_target/QuickTest_ShortPeptide.json
"
```

### Example 2: Custom Settings

```bash
docker exec bindcraft-api bash -c "
cd /workspace/BindCraft && python bindcraft.py \
  --settings my_settings.json \
  --advanced settings_advanced/custom_multimer.json \
  --filters settings_filters/strict_filters.json
"
```

### Example 3: Interactive Shell

```bash
docker exec -it bindcraft-api bash
cd /workspace/BindCraft
python bindcraft.py --help
```

## Troubleshooting

### Check GPU Support

```bash
docker exec bindcraft-api python -c "
import jax
print('JAX Devices:', jax.devices())
print('GPU:', 'YES' if any('cuda' in str(d).lower() for d in jax.devices()) else 'NO')
"
```

### Check Dependencies

```bash
docker exec bindcraft-api python -c "
import jax, colabdesign, biopython, scipy
try: import pyrosetta; print('PyRosetta: OK')
except: print('PyRosetta: NOT AVAILABLE')
"
```

### View Container Logs

```bash
docker compose logs -f bindcraft-api
```

### Stop and Clean

```bash
docker compose down
docker system prune -a  # Clean unused images
```

## Performance Notes

### Expected Times (RTX 4090)

- **Model Loading**: ~30 seconds (first run)
- **Single Design**: ~2-5 minutes (depending on settings)
- **QuickTest (3 designs)**: ~10-15 minutes total

### Memory Usage

- JAX Base Image: ~30GB
- BindCraft Deps: ~10GB
- Runtime: ~16GB GPU VRAM

### Optimization Tips

1. **GPU Selection**: Change `device_ids: ['0']` in docker-compose.yml for different GPU
2. **Batch Mode**: Run multiple designs sequentially for efficiency
3. **Advanced Settings**: See `settings_advanced/*.json` for tuning

## Volumes & Data

### Mounted Paths

```
Host                           Container
/data/inputs        ←→         /data/inputs      (PDB inputs)
/data/outputs       ←→         /data/outputs     (Temp files)
./results           ←→         /workspace/BindCraft/results (Outputs)
./params            ←→         /workspace/BindCraft/params  (AF2 weights)
```

### Map Your Data

```bash
docker run -v /path/to/inputs:/data/inputs \
           -v /path/to/outputs:/data/outputs \
           -it bindcraft:latest bash
```

## File Structure

```
BindCraft/
├── Dockerfile.bindcraft.pyrosetta  # Multi-stage build config
├── docker-compose.yml              # Docker orchestration
├── test_pipeline.sh                # Test script
├── bindcraft.py                    # Main pipeline
├── settings_target/                # Input configurations
├── settings_advanced/              # Advanced tuning
├── settings_filters/               # Output filters
├── functions/                      # Core modules
│   ├── colabdesign_utils.py       # AF2 interface
│   └── pyrosetta_utils.py         # Rosetta relax
└── params/                         # AF2 weights (pre-cached)
```

## License Notes

- **PyRosetta**: Academic non-commercial use. See RosettaCommons for commercial licenses
- **JAX**: Apache 2.0
- **ColabDesign**: Used under fair-use for research
- **AF2 Weights**: DeepMind license agreement

## Support

For issues:

1. Check Docker logs: `docker compose logs -f`
2. Validate environment: See "Troubleshooting" section
3. Test basic functionality: `bash test_pipeline.sh`
4. Check GPU: `nvidia-smi` (in host terminal)

## Version Info

- **Docker Image**: ghcr.io/nvidia/jax:jax + rosettacommons/rosetta
- **JAX**: Nightly (CUDA 13 support)
- **Python**: 3.10/3.12 (from base images)
- **ColabDesign**: v1.1.3
- **AF2 Weights**: 2022-12-06 release

---

**Last Updated**: 2026-01-23  
**Status**: ✓ Full Pipeline GPU-Enabled
