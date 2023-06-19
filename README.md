# Supervised Classification with Single Cell Integration

### Research Question
Exploring the effects of integrating scATAC-seq with scRNA-seq on supervised annotation of single cell data.

### Aim 1: Classification with RNA vs RNA + ATAC-seq

**Outcome**
- Cell type (nominal) metrics: F1 score
  
**Predictors**

Experimentally Integrated ATAC/RNA dataset
- Multi-omics 10x protocol, SNARE-seq
- Baseline: RNA-seq only
- Integration: Add ATAC-seq features
  
Computationally Integrated ATAC/RNA dataset
- Baseline: RNA-seq only
- Integration: Different integration methods
- Assumption-laden co-embedding - Seurat V3
- Assumption free co-embedding - Seurat dictionary learning
  
**Statistical Model**
- Classification model
- Simple ML models (SVM, random forest), deep learning models, transfer learning
  
**Sensitivity analysis**
- Test if results generalize to different organism or tissue

### Aim 2: Biological interpretation of differences between models
**Gene set enrichment analysis**
- Compare to marker gene sets to see if predictive gene sets are standard marker gene sets for biological validity
- Correct for multiple testing
- Repeat using baseline RNA model, to identify integration-specific markers

**Enriched regions of chromatin accessibility**

### Novelty
- Impact of ATAC-seq on automated annotation quality
- Effect of integration method on classification quality 
- Effect of ML model choice on integrated classification quality





