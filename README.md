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

## Data
[PBMC 10K Data](https://www.10xgenomics.com/resources/datasets/pbmc-from-a-healthy-donor-granulocytes-removed-through-cell-sorting-10-k-1-standard-1-0-0) (Filtered feature barcode matrix) and corresponding manual
[annotations](https://figshare.manchester.ac.uk/articles/dataset/TriTan_An_efficient_triple_non-negative_matrix_factorisation_method_for_integrative_analysis_of_single-cell_multiomics_data/23283998/1)

[Alzheimers Mouse Data (Not using due to lack of cell name annotations)](https://www.10xgenomics.com/resources/datasets/multiomic-integration-neuroscience-application-note-single-cell-multiome-rna-atac-alzheimers-disease-mouse-model-brain-coronal-sections-from-one-hemisphere-over-a-time-course-1-standard)

[Purified PBMC Data (Best annotations but scRNA only)](https://github.com/10XGenomics/single-cell-3prime-paper/tree/master/pbmc68k_analysis)
## Files
Old Scripts - Contains pre-processing and exploration of alzheimers mouse data

Pre-processing.py - Contains pre-processing and exploration of annotated PBMC 10K data. Used this [muon tutorial](https://muon-tutorials.readthedocs.io/en/latest/single-cell-rna-atac/index.html) to guide pre-processing.




