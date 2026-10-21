#!/bin/bash
# 🚀 BindCraft Pipeline Launcher via Docker
# Usage: ./launch_pipeline.sh [TARGET] [ALGORITHM] [FILTERS]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Defaults
TARGET_CONFIG="${1:-settings_target/GLP1_6X18_FullPipeline.json}"
ALGORITHM_CONFIG="${2:-settings_advanced/peptide_3stage_multimer.json}"
FILTERS_CONFIG="${3:-settings_filters/default_filters.json}"

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  🧬 BindCraft Pipeline Launcher       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}\n"

# Verify files exist
echo -e "${YELLOW}📋 Checking configuration files...${NC}"

if [ ! -f "$TARGET_CONFIG" ]; then
    echo -e "${RED}✗ Target config not found: $TARGET_CONFIG${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Target: $TARGET_CONFIG${NC}"

if [ ! -f "$ALGORITHM_CONFIG" ]; then
    echo -e "${RED}✗ Algorithm config not found: $ALGORITHM_CONFIG${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Algorithm: $ALGORITHM_CONFIG${NC}"

if [ ! -f "$FILTERS_CONFIG" ]; then
    echo -e "${RED}✗ Filters config not found: $FILTERS_CONFIG${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Filters: $FILTERS_CONFIG${NC}"

# Check Docker
echo -e "\n${YELLOW}🐳 Checking Docker...${NC}"
if ! docker ps --filter "name=bindcraft" | grep -q "bindcraft-api"; then
    echo -e "${RED}✗ Docker container not running${NC}"
    echo -e "${YELLOW}Starting Docker container...${NC}"
    docker compose up -d
    sleep 3
fi
echo -e "${GREEN}✓ Docker container active${NC}"

# Extract design name from target config
DESIGN_NAME=$(grep '"design_name"' "$TARGET_CONFIG" | cut -d'"' -f4)
DESIGN_PATH=$(grep '"design_path"' "$TARGET_CONFIG" | cut -d'"' -f4)

echo -e "\n${BLUE}════════════════════════════════════════${NC}"
echo -e "📊 Design Configuration:"
echo -e "  Design Name: ${GREEN}$DESIGN_NAME${NC}"
echo -e "  Output Dir:  ${GREEN}$DESIGN_PATH${NC}"
echo -e "  Algorithm:   ${GREEN}$(basename $ALGORITHM_CONFIG)${NC}"
echo -e "${BLUE}════════════════════════════════════════${NC}\n"

# Launch pipeline
echo -e "${YELLOW}🚀 Launching BindCraft pipeline...${NC}"
echo -e "${YELLOW}(This may take 30min - 2h depending on parameters)${NC}\n"

docker exec bindcraft-api bash -c "
  source /opt/conda/bin/activate BindCraft && \
  cd /workspace/BindCraft && \
  CUDA_VISIBLE_DEVICES=1 python bindcraft.py \
    -s $TARGET_CONFIG \
    -a $ALGORITHM_CONFIG \
    -f $FILTERS_CONFIG 2>&1 | tee /workspace/BindCraft/logs/pipeline_${DESIGN_NAME}.log
"

PIPELINE_EXIT_CODE=$?

if [ $PIPELINE_EXIT_CODE -eq 0 ]; then
    echo -e "\n${GREEN}✅ Pipeline completed successfully!${NC}"
    echo -e "${BLUE}════════════════════════════════════════${NC}"
    echo -e "📊 Results available at:"
    echo -e "  ${GREEN}$DESIGN_PATH${NC}"
    echo -e "\nKey files:"
    echo -e "  • ${YELLOW}structures/${NC}       - Final PDB structures"
    echo -e "  • ${YELLOW}sequences/${NC}        - FASTA sequences"
    echo -e "  • ${YELLOW}final_design_stats.csv${NC} - Design metrics"
    echo -e "${BLUE}════════════════════════════════════════${NC}"
    
    # Copy to Windows Desktop (optional)
    if [ -d "/mnt/c/Users/vincent/Desktop" ]; then
        echo -e "\n${YELLOW}💾 Copying results to Desktop...${NC}"
        mkdir -p "/mnt/c/Users/vincent/Desktop/BindCraft_${DESIGN_NAME}"
        cp -v "$DESIGN_PATH"/structures/*.pdb "/mnt/c/Users/vincent/Desktop/BindCraft_${DESIGN_NAME}/" 2>/dev/null || true
        echo -e "${GREEN}✓ Results copied to Desktop${NC}"
    fi
else
    echo -e "\n${RED}❌ Pipeline failed with exit code $PIPELINE_EXIT_CODE${NC}"
    exit 1
fi
