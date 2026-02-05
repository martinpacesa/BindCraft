# BindCraft Docker Pipeline - Guide Complet

## 📋 Table des matières
1. [Vue d'ensemble](#vue-densemble)
2. [Prérequis](#prérequis)
3. [Installation & Build](#installation--build)
4. [Lancer la Pipeline](#lancer-la-pipeline)
5. [Configuration](#configuration)
6. [Entrées & Sorties](#entrées--sorties)
7. [Exemples d'utilisation](#exemples-dutilisation)
8. [Troubleshooting](#troubleshooting)
9. [Possibilités & Limitations](#possibilités--limitations)

---

## 🎯 Vue d'ensemble

BindCraft est une pipeline **complète et automatisée** de design de peptides binders utilisant:
- **ColabDesign** : Hallucination + optimisation via AlphaFold2
- **PyRosetta** : Relaxation et analyse structurale
- **ProteinMPNN** : Optimisation de séquence
- **AlphaFold2 v3** : Prédiction de structures

**Tout fonctionne en Docker sur GPU NVIDIA** 🚀

---

## 📦 Prérequis

### Hardware
- **GPU NVIDIA** : RTX 4090, 5080, A100, etc. (minimum 24GB VRAM recommandé)
- **CPU** : 8+ cores
- **RAM** : 32GB minimum
- **Disk** : 100GB libre (35GB Docker image + 65GB résultats)

### Software
```bash
docker --version      # >= 20.10
nvidia-docker --version  # Ou runtime nvidia configuré
```

### Vérifier GPU dans Docker
```bash
docker run --rm --runtime=nvidia nvidia/cuda:12.2.0-base nvidia-smi
```

---

## 🔧 Installation & Build

### 1. Cloner le repo
```bash
cd /path/to/your/workspace
git clone https://github.com/martinpacesa/BindCraft.git
cd BindCraft
```

### 2. Builder l'image Docker
```bash
cd /path/to/biotech  # Dossier parent contenant docker-compose.yml
docker compose build
```

**Durée** : ~45 min (première fois)
**Résultat** : Image `bindcraft:native` (35.5GB)

Ou utiliser l'image pré-construite :
```bash
docker pull <votre-registry>/bindcraft:native
```

### 3. Vérifier l'image
```bash
docker images | grep bindcraft
# OUTPUT: bindcraft  native  fadf1eb71774  4 days ago  35.5GB
```

---

## 🚀 Lancer la Pipeline

### Option 1 : Via docker-compose (Recommandé)

```bash
cd /path/to/biotech  # Dossier contenant docker-compose.yml

# Démarrer
docker compose up -d

# Vérifier l'état
docker logs bindcraft-pipeline -f

# Arrêter
docker compose down
```

**docker-compose.yml inclut:**
- Configuration CUDA_VISIBLE_DEVICES=0
- Volume mounting pour données persistantes
- GPU reservation
- Entrypoint avec validation d'environnement

### Option 2 : Direct via docker run

```bash
docker run -d \
  --name bindcraft-pipeline \
  --runtime=nvidia \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v $(pwd)/BindCraft/results:/workspace/BindCraft/results \
  bindcraft:native \
  python bindcraft.py --settings settings_target/THP_NoFilters.json
```

### Vérifier le démarrage

```bash
# Logs directs
docker logs bindcraft-pipeline

# Résultat attendu (envs bien chargés)
=== BindCraft Environment (Native Conda) ===
✓ JAX: [CudaDevice(id=0)]
✓ PyRosetta
✓ ColabDesign
=============================
```

---

## ⚙️ Configuration

### Structure des Configs

Chaque config est un fichier JSON en `settings_target/`:

```json
{
  "design_path": "/workspace/BindCraft/results/THP_NoFilters/",
  "binder_name": "THP_NoFilters",
  "starting_pdb": "/workspace/BindCraft/example/THP_variantXG.pdb",
  "chains": "A",
  "target_hotspot_residues": "15-25",
  "lengths": [15, 20, 25],
  "number_of_final_designs": 5,
  "number_of_trajectories": 5,
  "sampling_temp": 1.0,
  "design_steps": 100,
  "relax_steps": 50,
  "filters": "no_filters"
}
```

### Paramètres Clés

| Paramètre | Description | Exemple |
|-----------|-------------|---------|
| `starting_pdb` | Cible (chemin Docker) | `/workspace/BindCraft/example/THP_variantXG.pdb` |
| `chains` | Chaîne à designer | `"A"` ou `"A,B"` |
| `target_hotspot_residues` | Résidus importants | `"15-25"` ou `"56,57,58"` |
| `lengths` | Longueurs peptide (aa) | `[10, 15, 20, 25]` |
| `number_of_final_designs` | Designs à générer | `5` (recommandé) |
| `number_of_trajectories` | Essais par design | `5` (plus = meilleur) |
| `sampling_temp` | Température (diversité) | `1.0` (normal) |
| `design_steps` | Itérations ColabDesign | `100` (par défaut) |
| `relax_steps` | Itérations PyRosetta | `50` (par défaut) |
| `filters` | Filtres qualité | `"default_filters"` ou `"no_filters"` |

### Configs Disponibles

```
✅ THP_Native.json       → THP strict (filters actifs)
✅ THP_NoFilters.json    → THP relaxé (tous acceptés)
✅ GLP1_Permissive.json  → GLP1 avec hotspots simples
✅ PDL1_Docker.json      → PDL1 variant
✅ QuickTest_ShortPeptide.json → Rapide (10-15min)
```

### Changer de Config

**Éditer `docker-compose.yml`:**
```yaml
command: python bindcraft.py --settings settings_target/GLP1_Permissive.json
```

**Ou relancer directement:**
```bash
docker exec bindcraft-pipeline \
  python bindcraft.py --settings settings_target/GLP1_Permissive.json
```

---

## 📥 📤 Entrées & Sorties

### Entrées

#### 1. **PDB Cible** (Structure protéine)
- **Localisation Docker** : `/workspace/BindCraft/example/`
- **Format** : `.pdb` standard (ATOM records)
- **Ajout personnalisé** :
  ```bash
  # Sur l'hôte
  cp /chemin/local/ma_cible.pdb ./BindCraft/example/
  
  # Dans config JSON
  "starting_pdb": "/workspace/BindCraft/example/ma_cible.pdb"
  ```

#### 2. **Config JSON** (Paramètres)
- **Localisation** : `BindCraft/settings_target/`
- **Créer une config personnalisée** :
  ```bash
  cp settings_target/THP_NoFilters.json settings_target/MonProjet.json
  # Éditer avec vos paramètres
  ```

#### 3. **Données de Poids** (AF2)
- **Pré-téléchargées** : Dans l'image Docker (5.3GB)
- **Aucune action requise** ✓

### Sorties

#### Structure de Résultats
```
results/
└── MonProjet/              # Nom du run
    ├── Accepted/           # ✅ Structures acceptées
    │   ├── Ranked/         # Triées par qualité
    │   ├── Pickle/         # Objets Python
    │   ├── Animation/      # Vidéos trajectoire
    │   └── Plots/          # Graphes métriques
    │
    ├── Trajectory/         # Toutes les trajectoires
    │   ├── Relaxed/        # ✓ PyRosetta réussies
    │   ├── LowConfidence/  # ⚠ Qualité faible
    │   ├── Clashing/       # ✗ Chocs atomiques
    │   ├── Pickle/         # Sauvegardes
    │   ├── Animation/      # Vidéos MD
    │   └── Plots/          # Graphes
    │
    ├── MPNN/               # Séquences optimisées
    │   ├── Sequences/      # FASTA optimisés
    │   ├── Binder/         # PDB redessinés
    │   ├── Relaxed/        # PyRosetta +MPNN
    │   └── ...
    │
    ├── trajectory_stats.csv     # Tous les designs
    ├── mpnn_design_stats.csv    # MPNN + stats
    ├── final_design_stats.csv   # Résumé final
    └── failure_csv.csv          # Rejets + raison
```

#### Fichiers Clés

**CSV : `trajectory_stats.csv`**
```
Design,Sequence,pLDDT,pTM,i_pTM,pAE,i_pAE,dG,RMSD,n_InterfaceResidues,n_InterfaceHbonds,...
THP_NoFilters_l25_s746091,MNKERKIEKTLSKTFPGLYRVYKEM,0.83,0.61,0.61,0.23,0.20,-29.55,1.45,9,0,...
...
```

**PDB : `Trajectory/Relaxed/MonProjet_l25_s746091.pdb`**
```
ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00 50.00           N
...
END
```

**Rapport : `failure_csv.csv`**
```
Design,Reason
THP_NoFilters_l15_s123456,Low pLDDT: 0.65
THP_NoFilters_l20_s789012,Severe clashes detected
```

---

## 🔬 Exemples d'utilisation

### Exemple 1 : Rapide (QuickTest - 10min)

```bash
# Modifier docker-compose.yml
command: python bindcraft.py --settings settings_target/QuickTest_ShortPeptide.json

# Lancer
docker compose down && docker compose up -d

# Vérifier après 10min
cd BindCraft/results/QuickTest_ShortPeptide
ls -la Trajectory/Relaxed/   # ~3-5 structures
head trajectory_stats.csv
```

### Exemple 2 : Production (THP_NoFilters - 8h)

```bash
# Lancé overnight
docker compose up -d

# Le matin, analyser les résultats
python3 << 'EOF'
import csv
with open('BindCraft/results/THP_NoFilters/trajectory_stats.csv') as f:
    reader = csv.DictReader(f)
    designs = sorted(reader, key=lambda x: float(x['pLDDT']), reverse=True)
    
print("TOP 3 DESIGNS")
for design in designs[:3]:
    print(f"{design['Design']}: pLDDT={design['pLDDT']}, Seq={design['Sequence']}")
EOF
```

### Exemple 3 : Cible Personnalisée

**1. Préparer le PDB:**
```bash
# Télécharger de PDBe ou AlphaFoldDB
wget https://alphafold.ebi.ac.uk/files/AF-P01234-F1-model_v4.pdb
cp AF-P01234-F1-model_v4.pdb BindCraft/example/MyTarget.pdb
```

**2. Créer la config:**
```bash
cat > BindCraft/settings_target/MyProject.json << 'EOF'
{
  "design_path": "/workspace/BindCraft/results/MyProject/",
  "binder_name": "MyProject",
  "starting_pdb": "/workspace/BindCraft/example/MyTarget.pdb",
  "chains": "A",
  "target_hotspot_residues": "10-20",
  "lengths": [12, 15, 18],
  "number_of_final_designs": 5,
  "number_of_trajectories": 5,
  "filters": "no_filters"
}
EOF
```

**3. Lancer:**
```bash
docker exec bindcraft-pipeline python bindcraft.py --settings settings_target/MyProject.json
```

### Exemple 4 : Analyse des Résultats

```bash
cd BindCraft/results/THP_NoFilters

# Statistiques résumées
wc -l trajectory_stats.csv    # Nombre de designs
head -2 trajectory_stats.csv  # En-têtes CSV
tail -5 trajectory_stats.csv  # Derniers designs

# Top 5 par pLDDT
python3 << 'EOF'
import csv
with open('trajectory_stats.csv') as f:
    reader = csv.DictReader(f)
    designs = sorted(reader, key=lambda x: float(x['pLDDT']), reverse=True)
    for i, d in enumerate(designs[:5], 1):
        print(f"{i}. {d['Design']}: pLDDT={d['pLDDT']} Energy={d['dG']}")
EOF

# Exporter les top designs
for pdb in Trajectory/Relaxed/THP_NoFilters_l25*.pdb; do
    cp "$pdb" /chemin/export/
done
```

---

## 🛠️ Troubleshooting

### Erreur : `RuntimeError: CUDA out of memory`
```
Solution:
1. Réduire design_steps: 100 → 50
2. Réduire longueurs: [25] → [20]
3. Réduire trajectories: 5 → 3
```

### Erreur : `AttributeError: 'list' object has no attribute 'split'`
```
Cause: Format "chains" mal formé dans JSON
Solution:
"chains": ["A"]  ❌
"chains": "A"    ✓
```

### Container crash après 30min
```
Cause: Mémoire GPU insuffisante
Solution:
1. docker compose down && docker compose up -d  # Redémarrer
2. Réduire les paramètres (voir au-dessus)
3. Vérifier nvidia-smi: gpu memory libre?
```

### Zéro designs acceptés
```
Cause possible: Filters trop stricts OU cible incompatible
Solution:
1. Utiliser "filters": "no_filters"
2. Tester avec QuickTest (structure test simple)
3. Ajuster hotspots (moins restrictifs)
```

### Comment vérifier logs en temps réel?
```bash
docker logs -f bindcraft-pipeline

# Ou filtrer
docker logs bindcraft-pipeline 2>&1 | grep "Stage\|successful\|Starting"
```

---

## 🎯 Possibilités & Limitations

### ✅ Possibilités

| Fonctionnalité | Détail |
|---|---|
| **Multiprotéines** | Complexes protéine-protéine (chaînes A,B) |
| **Taillesvarées** | Peptides 10-30aa |
| **Hotspots** | Cibler régions spécifiques (binding sites) |
| **MPNN** | Optimisation de séquence post-design |
| **Itérations** | Relancer sur même cible = nouveaux designs |
| **Parallelisation** | Plusieurs GPU (avec modifications docker) |
| **Stockage** | Résultats persistants sur disque |
| **Qualité** | Structures pLDDT 0.7-0.9 typiques |

### ⚠️ Limitations

| Limitation | Description | Contournement |
|---|---|---|
| **GPU unique** | 1 seule GPU support natif | Modifier docker-compose pour multi-GPU |
| **Vitesse** | ~2-3 min par trajet (5 = 15min) | Réduire trajectories/steps |
| **Mémoire** | RTX 4090 min, A100 optimal | Réduire design_steps |
| **AlphaFold2 age** | v3 (2022), pas v4 | Attendre mise à jour BindCraft |
| **MPNN** | Optimisation légère seulement | Pas de redesign complet |
| **Pas d'affinité** | AF2 prédit structure, pas binding affinity | Validation computationnelle externe requise |
| **Hotspots statiques** | Fixe au run, pas adaptatif | Relancer avec hotspots différents |
| **Complexité max** | Complexes jusqu'à ~5 chaînes | Beyond = timeout possible |

### 🎯 Cas d'Usage Recommandés

✅ **Idéal pour:**
- Conception rapide de binders peptidiques
- Screening de hotspots
- Générer candidats pour validation expérimentale
- Exploration de espace design (variations longueur, position)
- Démonstration prototype MLOps/biotech

❌ **Pas idéal pour:**
- Prédiction d'affinité (besoin validation) 
- Designs avec contraintes spatiales complexes
- Production d'échelle (centaines de cibles)
- Petites molécules (AF2 ≈ protéines)

---

## 📊 Performance Typique

**Matériel:** RTX 4090 (24GB VRAM)
```
Startup:           15 sec
Per trajectory:    2-3 min
Per design (5 traj): 12-15 min
5 designs total:   60-75 min
```

**Résultats typiques:**
```
Lancé: 5 designs × 5 trajectoires = 25 tentatives
Relaxed: ~15-20 (60-80%) passent PyRosetta
Acceptés: 0-5 (0-20%, dépend filtres)
MPNN: ~40-50 séquences optimisées
```

---

## 📚 Références & Extensions

### Repos Originaux
- [BindCraft](https://github.com/martinpacesa/BindCraft)
- [ColabDesign](https://github.com/sokrypton/ColabDesign)
- [ProteinMPNN](https://github.com/dauparas/ProteinMPNN)

### Où Modifier

**Paramètres:** `settings_target/*.json`
**Cibles:** `example/*.pdb`
**Code:** `functions/*.py`
**Pipeline:** `bindcraft.py`

---

## ✉️ Support & Versioning

**Version Docker:** 35.5GB, 4 jours old
**Image:** `bindcraft:native` (Miniforge + JAX CUDA 0.4-0.6)
**Git:** BindCraft_DockerReady branch (à jour)

---

## 📝 Checkliste Rapide

```
⬜ 1. Vérifier GPU: nvidia-smi
⬜ 2. Builder image: docker compose build
⬜ 3. Préparer config: settings_target/MonProjet.json
⬜ 4. Ajouter cible: example/MaCible.pdb (optionnel)
⬜ 5. Éditer docker-compose.yml (--settings path)
⬜ 6. Lancer: docker compose up -d
⬜ 7. Monitorer: docker logs -f bindcraft-pipeline
⬜ 8. Analyser: results/MonProjet/trajectory_stats.csv
⬜ 9. Exporter designs: cp results/MonProjet/Relaxed/*.pdb ./export/
⬜ 10. Archiver: git commit results/
```

---

**Questions?** Voir logs détaillés:
```bash
docker logs bindcraft-pipeline 2>&1 | tail -100
```
