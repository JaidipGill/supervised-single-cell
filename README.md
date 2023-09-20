# Combining Single-cell ATAC and RNA Sequencing for Supervised Cell Annotation

## Research Question
How can scATAC-seq be used to improve supervised annotation?

### Aim 1: Compare the effects of the choice of dimensionality reduction method on ATAC utility

### Aim 2: Compare the effects of classifier choice on ATAC utility

### Aim 3: Investigate how ATAC utility differs for different cell types and granularity levels

### Novelty
- Impact of ATAC-seq on automated annotation quality

## Methods (Modelling)

![Alt text](image.png)

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

The pipeline can be run by:
1. Download [PBMC 10K Data](https://www.10xgenomics.com/resources/datasets/pbmc-from-a-healthy-donor-granulocytes-removed-through-cell-sorting-10-k-1-standard-1-0-0) and place in 'PBMC 10k multiomic' folder
1. QC of feature-barcode matrix + ground truth annotation -> **Pre-processing.py**
2. Generation of bootstrap X and y datasets -> **boostrap.py**
3. Process results for further analysis -> **Results Analysis.py**
4. Statistical testing of processed results -> **stats_test_bootstrap.R**

```
C:.
|   .gitignore
|   .Rhistory
|   bootstrap.py                # Generates bootstrapped embeddings and classification results using original dataset
|   bootstrap_utils.py 
|   Interpretation dataset.py   # Generates dataset for interpretation
|   Interpretation Model.R      # Performs interpreation analysis
|   ML_models.py                # Supervised ML algorithms, feature importance and other visualisations
|   output.txt
|   Pre-processing.py           # QC of original dataset. This [muon tutorial](https://muon-tutorials.readthedocs.io/en/latest/single-cell-rna-atac/index.html) was used to guide pre-processing.
|   README.md
|   Result Analysis.py          # Processing of results
|   Results.xlsx                
|   stats_tests_bootstrap.R     # Statistical test of classificationr results
|   Train_test PBMC10k.py       # Dimensionality reduction of pre-processed PBMC 10K data to output training and test data ready for input into supervised ML workflows.
|   Utils.py                    # Utility functions used in other Python scripts
+---Data
|   +---B cell lymphoma         # Additional cancer dataset analysis, not included in paper
|   |   +---Bootstrap_X
|   |   \---Bootstrap_y
|   +---Figures
|   \---PBMC 10k multiomic      # Main PMBC dataset, includes annotation files
|       +---Bootstrap_X         # Bootstrapped feature sets
|       +---Bootstrap_y         # OOB label sets
|       +---Final Results
|       +---Interpretation
|       +---processed_data
|       |   +---Dim_Red
|       |   +---PCA
|       |   \---X_Matrices
|       \---Silhouette Scores
+---Supervised Models
|   +---Macro Metrics
|   +---PCA
|   +---Saved Models
|   \---scVI
|       
```













