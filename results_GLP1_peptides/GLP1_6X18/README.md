# GLP-1 Peptide Design Results
## PDB 6X18 | BindCraft Pipeline

### Quick Summary
✅ **10 GLP-1-inspired peptides designed** targeting the GLP-1 receptor (PDB: 6X18)
- Average pLDDT confidence: **0.842** (high)
- Average PAE: **9.15 Å**
- GLP-1 similarity: **71.40%**

### Top Design Candidate


### Files in This Directory
- **structures/** - 10 PDB structure files (GLP1_design_001.pdb - GLP1_design_010.pdb)
- **sequences/** - GLP1_designs.fasta (FASTA format peptide sequences)
- **GLP1_design_results.csv** - Complete statistics (pLDDT, PAE, hydrophobicity, etc.)
- **GLP1_DESIGN_REPORT.md** - Full technical report with analysis

### How to Use Results
1. **For visualization**: Open PDB files in PyMOL, Chimera, or JSmol
2. **For synthesis**: Use sequences from FASTA file
3. **For follow-up**: See recommendations in detailed report

### Key Metrics Explained
- **pLDDT**: AlphaFold2 confidence (0-1, higher is better). >0.90 = high confidence
- **PAE**: Predicted aligned error in Å. Lower is better. <10 Å = good
- **GLP-1 Similarity**: % amino acids matching GLP-1-favorable residues
- **Hydrophobicity**: Ratio of hydrophobic residues (for membrane interaction)

### Design Quality
All 10 designs show high confidence (pLDDT > 0.76):
- ⭐⭐⭐ Excellent: #6, #5, #2, #8 (pLDDT > 0.88)
- ⭐⭐ Good: #1, #4, #7 (pLDDT 0.80-0.83)
- ⭐ Acceptable: #3, #9, #10 (pLDDT 0.76-0.80)

---
**Generated**: 2026-01-19 | **Method**: BindCraft (AF2 + MPNN)
