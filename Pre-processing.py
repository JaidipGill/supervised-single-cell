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

DATA = 'cancer' # 'pbmc' or 'cancer' or 'AD

if DATA == 'pbmc':
    RAW = 'Data/PBMC 10k multiomic/pbmc_filtered.h5'
    QC = 'Data/PBMC 10k multiomic/QC-pbmc10k.h5mu'
    UPPER_GENES = 5000
    UPPER_UNIQUE_PEAKS = 15000
    LOWER_UNIQUE_PEAKS = 2000
    TOTAL_COUNTS = 15000
    LOWER_TOTAL_PEAKS=4000
    UPPER_TOTAL_PEAKS=40000
elif DATA == 'cancer':
    RAW = 'Data/B cell lymphoma/lymph_node_lymphoma_14k_filtered_feature_bc_matrix.h5'
    QC = 'Data/B cell lymphoma/QC-bcell.h5mu'
    UPPER_GENES = 5000
    UPPER_UNIQUE_PEAKS = 30000
    LOWER_UNIQUE_PEAKS = 500
    TOTAL_COUNTS = 15000
    LOWER_TOTAL_PEAKS=500
    UPPER_TOTAL_PEAKS=70000
elif DATA == 'AD':
    RAW = "Data/Alz multiomic/Downsampled10%_matrix.h5mu"
    QC = 'Data/Alz multiomic/Downsampled10%_matrix_processed.h5mu'
    UPPER_GENES = 10000
    UPPER_UNIQUE_PEAKS = 25000
    LOWER_UNIQUE_PEAKS = 2000
    TOTAL_COUNTS = 25000
    LOWER_TOTAL_PEAKS=4000
    UPPER_TOTAL_PEAKS=40000

# %% ----------------------------------------------------------------
# Quality Control

ut.quality_control(data = DATA, input_file = RAW, upper_genes = UPPER_GENES, total_counts = TOTAL_COUNTS, output_file = QC, upper_unique_peaks = UPPER_UNIQUE_PEAKS, lower_unique_peaks=LOWER_UNIQUE_PEAKS, lower_total_peaks=LOWER_TOTAL_PEAKS, upper_total_peaks=UPPER_TOTAL_PEAKS)

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
# %% ----------------------------------------------------------------
# CREATE MIXED ANNOTATIONS FOR B CELL LYMPHOMA DATASET

# Load the annotations
annotations_t = pd.read_csv('Data/B cell lymphoma/T-cell Subtypes.csv')
annotations_o = pd.read_csv('Data/B cell lymphoma/Original Cell Types.csv')

# Set 'Barcode' as index for df1
annotations_t.set_index('Barcode', inplace=True)
# Merge the dataframes on the index (which is 'Barcode' for df1 and 'index' for df2)
merged_df = annotations_o.merge(annotations_t, left_on='index', right_index=True, how='left')
# Update the 'x' column in df2 with 'T-cell Subtypes' from merged_df
annotations_o['x'] = merged_df['T-cell Subtypes']
# Fill the NaN values in 'x' column with 'Original Cell Types' from merged_df
annotations_o['x'].fillna(merged_df['x'], inplace=True)
# Save the updated annotations
annotations_o.to_csv('Data/B cell lymphoma/Complete Cell Types.csv', index=False)

# %%
