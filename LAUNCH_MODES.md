# 🚀 BindCraft Pipeline - Tous les Modes de Lancement

## 3 Façons de Lancer le Pipeline

---

## 1️⃣ Mode INTERACTIF (GUI) - ⭐ Recommandé pour commencer

```bash
python interactive_launcher.py
```

**Fonctionnement**:
1. Menu: Sélectionner la cible (GLP1, PDL1, custom, etc.)
2. Menu: Sélectionner l'algorithme (peptide_3stage, 4stage, etc.)
3. Menu: Sélectionner les filtres de qualité
4. Confirmation avant lancement

**Avantage**: Zero configuration, tout visuel!

---

## 2️⃣ Mode SCRIPT BASH - Pour automatisation

```bash
# Lancement basique avec presets
./launch_pipeline.sh

# Avec paramètres custom
./launch_pipeline.sh \
  settings_target/GLP1_6X18_FullPipeline.json \
  settings_advanced/peptide_3stage_multimer_mpnn.json \
  settings_filters/default_filters.json
```

**Fichiers presets disponibles**:
- `GLP1_6X18_FullPipeline.json` ← Default
- `QuickTest_ShortPeptide.json` ← Test rapide
- Vos configs custom

---

## 3️⃣ Mode DIRECT Docker - Pour experts

```bash
cd /home/vincent/code/repo/biotech/BindCraft

docker exec bindcraft-api bash -c "
  source /opt/conda/bin/activate BindCraft && \
  cd /workspace/BindCraft && \
  python bindcraft.py \
    -s settings_target/YOUR_CONFIG.json \
    -a settings_advanced/YOUR_ALGORITHM.json \
    -f settings_filters/YOUR_FILTERS.json
"
```

---

## 📝 Créer une Config Personnalisée

### Option A: Via File JSON

```bash
cat > settings_target/MyDesign.json << 'EOF'
{
  "design_name": "MyDesign",
  "starting_pdb": "./example/6X18.pdb",
  "chain": "A",
  "design_path": "./results/MyDesign",
  "binder_name": "my_peptide",
  "target_hotspot_residues": "100,110,120",
  "lengths": [30, 35, 40],
  "number_of_final_designs": 10,
  "number_of_trajectories": 100,
  "number_of_mpnn_designs": 5
}
EOF

# Puis lancer
./launch_pipeline.sh settings_target/MyDesign.json
```

### Option B: Edit JSON existant

```bash
# Copier template
cp settings_target/GLP1_6X18_FullPipeline.json settings_target/MyCustom.json

# Éditer
nano settings_target/MyCustom.json

# Voir changements
diff settings_target/GLP1_6X18_FullPipeline.json settings_target/MyCustom.json

# Lancer
./launch_pipeline.sh settings_target/MyCustom.json
```

---

## 🎯 Paramètres Clés Expliqués

```json
{
  "starting_pdb": "./example/6X18.pdb",     // ← Votre protéine cible
  "chain": "A",                             // ← Chaîne à cibler
  "target_hotspot_residues": "140,143",    // ← Points clés de binding
  "lengths": [30, 35, 40],                 // ← Longueurs peptides (AA)
  "number_of_final_designs": 10,           // ← Combien de designs finaux
  "number_of_trajectories": 100,           // ← Itérations AF2 (+ = meilleur)
  "number_of_mpnn_designs": 5              // ← Variantes MPNN par design
}
```

**Presets recommandés**:
- **Quick Test** (5 min):
  ```json
  "lengths": [25],
  "number_of_final_designs": 3,
  "number_of_trajectories": 20
  ```

- **Production** (1-2h):
  ```json
  "lengths": [30, 35, 40],
  "number_of_final_designs": 10,
  "number_of_trajectories": 100
  ```

- **High Quality** (3-4h):
  ```json
  "lengths": [25, 30, 35, 40, 45],
  "number_of_final_designs": 15,
  "number_of_trajectories": 200
  ```

---

## 🧠 Choisir l'Algorithme

```
peptide_3stage_multimer.json        ← Défaut, bon équilibre
  └─ 3 phases: logits → softmax → hard
     ✓ Rapide, bon pour peptides
     ✓ Recommandé pour < 50 AA

peptide_3stage_multimer_mpnn.json   ← Avec ProteinMPNN
  └─ Inclut étape sequence design
     ✓ Meilleure diversity
     ✓ Plus coûteux en temps

default_4stage_multimer.json        ← Complexes stables
  └─ 4 phases: logits → softmax → hard → greedy
     ✓ Plus intensif
     ✓ Pour complexes > 50 AA

betasheet_4stage_multimer.json      ← Pour structures β
  └─ Optimisé pour beta-sheets
     ✓ Si vous voulez beta-sheets
     ✓ Bien pour interaction à large interface
```

---

## 📊 Exemples de Lancement

### Exemple 1: GLP-1 Simple (5 min)
```bash
# Utiliser config pré-faite
./launch_pipeline.sh
# ou
./launch_pipeline.sh settings_target/QuickTest_ShortPeptide.json
```

### Exemple 2: GLP-1 Complet (1h)
```bash
./launch_pipeline.sh settings_target/GLP1_6X18_FullPipeline.json
```

### Exemple 3: Custom PDL-1 (2h)
```bash
# 1. Créer config
cat > settings_target/PDL1_Custom.json << 'EOF'
{
  "design_name": "PDL1_Custom",
  "starting_pdb": "./example/PDL1_BMS.pdb",  # Votre PDB
  "chain": "A",
  "design_path": "./results/PDL1_Custom",
  "binder_name": "pdl1_binding_peptide",
  "target_hotspot_residues": "50,60,70,80",  # Vos hotspots
  "lengths": [30, 35, 40],
  "number_of_final_designs": 10,
  "number_of_trajectories": 100,
  "number_of_mpnn_designs": 5
}
EOF

# 2. Lancer
./launch_pipeline.sh settings_target/PDL1_Custom.json
```

### Exemple 4: Via Python interactif
```bash
python interactive_launcher.py
# → Menu...
# → Select GLP1_6X18_FullPipeline
# → Select peptide_3stage_multimer
# → Select default_filters
# → Confirm → LAUNCH!
```

---

## 📈 Monitoring Exécution

### Afficher log en temps réel
```bash
# Dans un autre terminal
docker exec bindcraft-api tail -f /workspace/BindCraft/results/GLP1_6X18_FullPipeline/trajectory_stats.csv
```

### Voir nombre de designs générés
```bash
docker exec bindcraft-api bash -c "
  ls /workspace/BindCraft/results/GLP1_6X18_FullPipeline/Trajectory/*.pdb | wc -l
"
```

### Vérifier GPU usage
```bash
docker exec bindcraft-api nvidia-smi
```

---

## 📁 Accéder aux Résultats

```bash
# Local (Linux)
ls -lh results/GLP1_6X18_FullPipeline/structures/
cat results/GLP1_6X18_FullPipeline/final_design_stats.csv

# Windows Desktop (auto-copié)
C:\Users\vincent\Desktop\BindCraft_GLP1_6X18_FullPipeline\

# Fichiers clés:
├── structures/              # PDB files ← Ouvrir dans PyMOL
├── sequences/               # FASTA ← Pour synthèse peptide
├── final_design_stats.csv   # Metrics (pLDDT, PAE, etc.)
└── trajectory_stats.csv     # Stats complètes
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| Docker not found | `docker compose up -d` |
| No target configs | Créer fichier dans `settings_target/` |
| Out of Memory | Réduire `number_of_trajectories` de 100 → 50 |
| GPU timeout | Utiliser `peptide_3stage` au lieu de `4stage` |
| PyRosetta error | `docker exec bindcraft-api conda run -n BindCraft python -c "import pyrosetta"` |

---

## 🎓 Workf low Recommandé

### Phase 1: Test Rapide ⚡
```bash
./launch_pipeline.sh settings_target/QuickTest_ShortPeptide.json
# → 5-10 min
# → 3 designs
# → Validez géométrie/hotspots
```

### Phase 2: Design Production 🚀
```bash
./launch_pipeline.sh settings_target/GLP1_6X18_FullPipeline.json
# → 1-2 heures
# → 10 designs
# → Métriques complètes
```

### Phase 3: Optimisation Fine-Tuning 🎯
```bash
# Créer config avancée
cat > settings_target/GLP1_HighQuality.json << 'EOF'
{
  ...
  "number_of_trajectories": 200,
  "number_of_mpnn_designs": 8
}
EOF

./launch_pipeline.sh settings_target/GLP1_HighQuality.json
# → 3-4 heures
# → 15+ designs
# → Meilleure qualité
```

---

## 💡 Quick Reference Card

```bash
# Lancer interactif (recommandé)
python interactive_launcher.py

# Lancer script simple
./launch_pipeline.sh

# Lancer avec params custom
./launch_pipeline.sh \
  settings_target/MyConfig.json \
  settings_advanced/peptide_3stage_multimer_mpnn.json \
  settings_filters/default_filters.json

# Voir résultats
ls results/*/structures/

# Voir métriques
cat results/*/final_design_stats.csv

# Copier vers Desktop
cp results/*/structures/*.pdb /mnt/c/Users/vincent/Desktop/
```

---

**Prêt? Lance `python interactive_launcher.py` pour commencer! 🚀**
