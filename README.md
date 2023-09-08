# Combining Single-cell ATAC and RNA Sequencing for Supervised Cell Annotation

## Research Question
How can scATAC-seq be used to improve supervised annotation?

### Aim 1: Compare the effects of the choice of dimensionality reduction method on ATAC utility

### Aim 2: Compare the effects of classifier choice on ATAC utility

### Aim 3: Investigate how ATAC utility differs for different cell types and granularity levels

### Novelty
- Impact of ATAC-seq on automated annotation quality

## Methods (Modelling)
**Outcome**
- Cell-type

**Predictors**
Experimentally Integrated ATAC/RNA dataset
- Multi-omics 10x protocol
- Baseline: RNA-seq only
- Combined model: RNA + ATAC-seq

**Metrics**
- Macro metrics: F1 score, precision, recall
- Cell type (nominal) metrics: F1 score, precision, recall, Proportion of Ambiguous Predictions (PAP)

**Statistical Model**
- Dimensionality reduction: PCA, scVI autoencoder
- Classifiers: Support vector machine (SVM), random forest (RF), logistic regression (RF)
  
## Data
[PBMC 10K Data](https://www.10xgenomics.com/resources/datasets/pbmc-from-a-healthy-donor-granulocytes-removed-through-cell-sorting-10-k-1-standard-1-0-0) (Filtered feature barcode matrix)

[B Cell Lymphoma Data](https://www.10xgenomics.com/resources/datasets/fresh-frozen-lymph-node-with-b-cell-lymphoma-14-k-sorted-nuclei-1-standard-2-0-0)

## Files
ML_models.py - Supervised ML algorithms, feature importance and other visualisations

QC PBMC10K.py - Contains pre-processing and exploration of annotated PBMC 10K data. This [muon tutorial](https://muon-tutorials.readthedocs.io/en/latest/single-cell-rna-atac/index.html) was used to guide pre-processing.

Train_test PBMC10K - Dimensionality reduction of pre-processed PBMC 10K data to output training and test data ready for input into supervised ML workflows.

Utils.py - Utility functions used in other Python scripts












