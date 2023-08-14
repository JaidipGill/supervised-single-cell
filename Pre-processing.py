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

'''
#CONVERTING MATRIX DIRECTORY TO MUDATA FORMAT
# Load the .mtx file
matrix = scipy.io.mmread('Data/PBMC 10k multiomic/filtered_feature_bc_matrix/matrix.mtx.gz')
# Load barcodes file
barcodes = pd.read_csv('Data/PBMC 10k multiomic/filtered_feature_bc_matrix/barcodes.tsv.gz', header=None, sep='\t')
# Load genes file
features = pd.read_csv('Data/PBMC 10k multiomic/filtered_feature_bc_matrix/features.tsv.gz', header=None, sep='\t')

# Split the features into genes and peaks based on the type column
genes = features[features[2] == 'Gene Expression']
peaks = features[features[2] == 'Peaks']

# Split the matrix into two based on the features
matrix = matrix.tocsr()  # Convert to CSR format for efficient row slicing
rna_matrix = matrix[genes.index, :]
atac_matrix = matrix[peaks.index, :]
# Transpose the matrices
rna_matrix = rna_matrix.transpose()
atac_matrix = atac_matrix.transpose()

# Create two separate AnnData objects
rna = ad.AnnData(X=rna_matrix, obs=barcodes, var=genes)
atac = ad.AnnData(X=atac_matrix, obs=barcodes, var=peaks)
# Combine the AnnData objects into a MuData object
mdata = mu.MuData({'rna': rna, 'atac': atac})
#mdata['atac'].var_names = [f"Peak_{i:d}" for i in range(mdata['atac'].n_vars)]
#mdata['rna'].var_names = [f"Gene_{i:d}" for i in range(mdata['rna'].n_vars)]
mdata['rna'].var_names = mdata['rna'].var[1]
mdata['atac'].var_names = mdata['atac'].var[1]
print(mdata.obs_names[:10])
'''
# %% ----------------------------------------------------------------
# CONFIGURATION

DATA = 'cancer' # 'pbmc' or 'cancer'

if DATA == 'pbmc':
    RAW = 'Data/PBMC 10k multiomic/pbmc_filtered.h5'
    QC = 'Data/PBMC 10k multiomic/QC-pbmc10k.h5mu'
elif DATA == 'cancer':
    RAW = 'Data/B cell lymphoma/lymph_node_lymphoma_14k_filtered_feature_bc_matrix.h5'
    QC = 'Data/B cell lymphoma/QC-bcell.h5mu'

# %% ----------------------------------------------------------------
# Quality Control

ut.quality_control(input_file = RAW, output_file = QC, upper_unique_peaks = 15000, lower_unique_peaks=2000, lower_total_peaks=4000, upper_total_peaks=40000)

# %% ----------------------------------------------------------------
# LOAD DATA FOR RE-ANNOTATION

mdata=mu.read_h5mu(QC)

# %% ----------------------------------------------------------------
# WNN CLUSTERING

mdata = ut.wnn_cluster(mdata)

# %% ----------------------------------------------------------------
# ANNOTATION BY SUBTYPE LEVEL 

ut.annotate_clusters(DATA, mdata, level=1, modality = 'rna')
ut.annotate_clusters(DATA, mdata, level=1, modality = 'wnn')

# %% 
