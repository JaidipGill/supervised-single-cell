# %% ----------------------------------------------------------------
import mudata as md
import scanpy as sc
import muon as mu
import anndata as ad
import h5py as h5
import pandas as pd
from muon import atac as ac
import Utils as ut
from importlib import reload

# %% ----------------------------------------------------------------
# CONFIGURATION

DATA = 'AD' # 'pbmc' or 'cancer' or 'AD

if DATA == 'pbmc':
    RAW = 'Data/PBMC 10k multiomic/pbmc_filtered.h5'
    QC = 'Data/PBMC 10k multiomic/QC-pbmc10k.h5mu'
    UPPER_GENES = 5000
    UPPER_UNIQUE_PEAKS = 15000
    TOTAL_COUNTS = 15000
elif DATA == 'cancer':
    RAW = 'Data/B cell lymphoma/lymph_node_lymphoma_14k_filtered_feature_bc_matrix.h5'
    QC = 'Data/B cell lymphoma/QC-bcell.h5mu'
    UPPER_GENES = 5000
    UPPER_UNIQUE_PEAKS = 15000
    TOTAL_COUNTS = 15000
elif DATA == 'AD':
    RAW = "Data/Alz multiomic/Downsampled10%_matrix.h5mu"
    QC = 'Data/Alz multiomic/Downsampled10%_matrix_processed.h5mu'
    UPPER_GENES = 10000
    UPPER_UNIQUE_PEAKS = 25000
    TOTAL_COUNTS = 25000

# %% ----------------------------------------------------------------
# Quality Control

ut.quality_control(data = DATA, input_file = RAW, upper_genes = UPPER_GENES, total_counts = TOTAL_COUNTS, output_file = QC, upper_unique_peaks = UPPER_UNIQUE_PEAKS, lower_unique_peaks=2000, lower_total_peaks=4000, upper_total_peaks=40000)

# %% ----------------------------------------------------------------
# LOAD DATA FOR RE-ANNOTATION OF PBMC DATASET

mdata=mu.read_h5mu(QC)

# %% ----------------------------------------------------------------
# WNN CLUSTERING

mdata = ut.wnn_cluster(mdata)

# %% ----------------------------------------------------------------
# ANNOTATION BY SUBTYPE LEVEL 

ut.annotate_clusters(DATA, mdata, level=1, modality = 'rna')
ut.annotate_clusters(DATA, mdata, level=1, modality = 'wnn')

# %% 
# UPDATE WNNL1 ANNOTATIONS BY AGGREGATING WNNL2

# Define the mapping dictionary
label_dict = {
    'B intermediate': 'B',
    'B naive': 'B',
    'CD14 Mono': 'Mono',
    'CD16 Mono': 'Mono',
    'CD4 Naive': 'CD4 T',
    'CD4 TCM': 'CD4 T',
    'CD4 TEM': 'CD4 T',
    'CD8 Naive': 'CD8 T',
    'CD8 TEM': 'CD8 T',
    'NK': 'NK',
    'Plasmablast': 'Plasmablast',
    'cDC2': 'Dendritic',
    'dnT': 'dnT',
    'pDC': 'Dendritic'
}

# Iterate through y_train and test files
for sample in range(10):
    for embedding in ['PCA', 'scVI']:
        for split in ['train', 'test']:
            # Load the y_train file
            y = pd.read_pickle(f'Data/PBMC 10k multiomic/Bootstrap_y/y_{split}_{embedding}_{sample}.pkl')
            # Replace the values in '1' column based on the mapping from '2'
            y['1'] = y['2'].map(label_dict)
            # Save the file
            y.to_pickle(f'Data/PBMC 10k multiomic/Bootstrap_y/y_{split}_{embedding}_{sample}.pkl')
# %%
