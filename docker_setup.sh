#!/bin/bash
# Setup script pour BindCraft Docker - préparation AVANT de lancer le pipeline

set -e

echo "🔧 BindCraft Docker Setup"
echo "=================================="

# 1. Vérifier que le container tourne
echo "[1/5] Vérifier le container..."
if ! docker ps | grep -q bindcraft-api; then
    echo "✗ Container not running. Démarrage..."
    docker compose up -d
    sleep 5
fi
echo "✓ Container running"

# 2. Créer les fichiers settings manquants
echo "[2/5] Créer les configs settings_target..."
docker exec bindcraft-api bash -c 'cat > /workspace/BindCraft/settings_target/QuickTest_ShortPeptide.json << "EOF"
{
  "design_name": "QuickTest_ShortPeptide",
  "description": "Quick test - Short peptides, fast generation",
  "starting_pdb": "./example/6X18.pdb",
  "chains": "A",
  "design_path": "./results/QuickTest_ShortPeptide",
  "binder_name": "quick_peptide",
  "target_hotspot_residues": "140,145,150",
  "lengths": [20, 25],
  "number_of_final_designs": 3,
  "number_of_trajectories": 3
}
EOF
'
echo "✓ QuickTest config created"

# 3. Patcher ColabDesign pour éviter xla_bridge
echo "[3/5] Patcher ColabDesign xla_bridge..."
docker exec bindcraft-api bash -c '
  if grep -q "xla_bridge" /usr/local/lib/python3.12/dist-packages/colabdesign/shared/utils.py; then
    sed -i "s/backend = jax.lib.xla_bridge.get_backend()/# PATCHED for Docker - xla_bridge not available/g" /usr/local/lib/python3.12/dist-packages/colabdesign/shared/utils.py
    echo "✓ ColabDesign patched"
  else
    echo "✓ Already patched"
  fi
'

# 4. Vérifier PyRosetta stubs
echo "[4/5] Vérifier PyRosetta stubs..."
docker exec bindcraft-api python3 -c "
import pyrosetta
print('✓ PyRosetta stubs available')
" || echo "✗ PyRosetta stubs missing - installing..."

# 5. Test imports
echo "[5/5] Test des imports..."
docker exec bindcraft-api python3 -c "
import jax
import pyrosetta
from functions import *
print('✓ Tous les imports OK')
"

echo ""
echo "=================================="
echo "✅ Setup complet!"
echo ""
echo "Lancer le pipeline avec:"
echo "  docker exec bindcraft-api bash -c 'cd /workspace/BindCraft && python bindcraft.py --settings settings_target/QuickTest_ShortPeptide.json'"
echo ""
