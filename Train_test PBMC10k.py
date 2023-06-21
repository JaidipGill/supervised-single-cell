# %% ----------------------------------------------------------------
import scanpy as sc
import muon as mu
import anndata as ad
import h5py as h5
import pandas as pd
from muon import atac as ac
import numpy as np
from sklearn.model_selection import train_test_split

# %% ----------------------------------------------------------------
# FUNCTIONS

def train_test_split_mdata(mdata_obj):

    '''
    Split MuData object into train and test sets
    '''

    # Get the indices of cells
    indices = np.arange(mdata_obj['rna'].shape[0])

    # Split indices into train and test
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    # Create train and test MuData objects
    mdata_train = mu.MuData({mod: mdata_obj[mod][train_idx] for mod in mdata_obj.mod.keys()}) 
    mdata_test = mu.MuData({mod: mdata_obj[mod][test_idx] for mod in mdata_obj.mod.keys()})

    # Convert views to AnnData objects
    mdata_train = mdata_train.copy()
    mdata_test = mdata_test.copy()

    return mdata_train, mdata_test

def pre_process(adata):

    '''
    Pre-process MuData object for either RNA or ATAC modality
    '''

    # Filter out low-frequency features
    mu.pp.filter_var(adata, 'n_cells_by_counts', lambda x: x >= 10) 

    # Saving raw counts
    adata.layers["counts"] = adata.X # Store unscaled counts in .layers["counts"] attribute

    # Normalizing peaks/genes
    print(f"Total peaks/genes in random cell: {adata.X[1,:].sum()}")
    sc.pp.normalize_total(adata, target_sum=1e4) # Normalize counts per cell
    print(f"Total peaks/genes in random cell: {adata.X[1,:].sum()}")
    sc.pp.log1p(adata) # Logarithmize + 1
    print(f"Total peaks/genes in random cell: {adata.X[1,:].sum()}") # Sanity check for normalization - should be 1e4

    # Filtering features
    sc.pp.highly_variable_genes(adata) # Select highly variable genes
    sc.pl.highly_variable_genes(adata) # Plot highly variable genes
    print(f"Number of highly variable peaks/genes: {np.sum(adata.var.highly_variable)}")
    print(f"Number of peaks/genes before filtering: {adata.n_vars}")
    #adata = adata[:, adata.var.highly_variable] # Filter out non-highly variable genes
    print(f"Number of peaks/genes after filtering: {adata.n_vars}")

    # Scaling
    adata.raw = adata # Store unscaled counts in .raw attribute
    sc.pp.scale(adata) # Scale to unit variance and shift to zero mean
    print(f"Min Scaled peaks/genes value:{adata.X.min()}") 
    print(f"Min unscaled peaks/genes value: {adata.raw.X.min()}") # Sanity check for scaling - should not be negative

    return adata

# %% ----------------------------------------------------------------
#LOAD DATA

# Loading QC data
mdata=mu.read_h5mu("Data/PBMC 10k multiomic/QC-pbmc10k.h5mu")
# %% ----------------------------------------------------------------
# TRAIN-TEST SPLIT

mdata_train, mdata_test = train_test_split_mdata(mdata)
# %% ----------------------------------------------------------------
# PRE-PROCESSING RNA

mdata_train.mod['rna'] = pre_process(mdata_train['rna'])
mdata_test.mod['rna'] = pre_process(mdata_test['rna'])

# %% ----------------------------------------------------------------
# PRE-PROCESSING ATAC

mdata_train.mod['atac'] = pre_process(mdata_train['atac'])
mdata_test.mod['atac'] = pre_process(mdata_test['atac'])

# %% ----------------------------------------------------------------
# Dimensionality reduction of RNA
sc.tl.pca(mdata_train.mod['rna'])
sc.pl.pca(mdata_train.mod['rna'], color=['CD2', 'CD79A', 'KLF4', 'IRF8'])
sc.tl.pca(mdata_test.mod['rna'])
sc.pl.pca(mdata_test.mod['rna'], color=['CD2', 'CD79A', 'KLF4', 'IRF8'])




# %%
