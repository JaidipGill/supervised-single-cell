# %% ----------------------------------------------------------------
import scanpy as sc
import muon as mu
import anndata as ad
import h5py as h5
import pandas as pd
from muon import atac as ac
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import time
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import Utils as ut
from importlib import reload
# %% ----------------------------------------------------------------
#LOAD DATA

# Loading QC data
mdata=mu.read_h5mu("Data/PBMC 10k multiomic/processed_data/QC-pbmc10k.h5mu")

# %% ----------------------------------------------------------------
# TRAIN-TEST SPLIT

mdata_train, mdata_test = ut.train_test_split_mdata(mdata)

# %% ----------------------------------------------------------------
# PRE-PROCESSING RNA

mdata_train.mod['rna'] = ut.pre_process_train(mdata_train['rna'])
mdata_test.mod['rna'] = ut.pre_process_test(mdata_test['rna'], mdata_train['rna'])
# %% ----------------------------------------------------------------
# PRE-PROCESSING ATAC

mdata_train.mod['atac'] = ut.pre_process_train(mdata_train['atac'])
mdata_test.mod['atac'] = ut.pre_process_test(mdata_test['atac'], mdata_train['atac'])

# %% ----------------------------------------------------------------
# SAVE PROCESSED DATA

mdata_train.write_h5mu("Data/PBMC 10k multiomic/processed_data/pre-proc_mdata_train.h5mu")
mdata_test.write_h5mu("Data/PBMC 10k multiomic/processed_data/pre-proc_mdata_test.h5mu")

# %% ----------------------------------------------------------------
# LOAD PROCESSED DATA

mdata_train = mu.read_h5mu("Data/PBMC 10k multiomic/processed_data/pre-proc_mdata_train.h5mu")
mdata_test = mu.read_h5mu("Data/PBMC 10k multiomic/processed_data/pre-proc_mdata_test.h5mu")

# %% ----------------------------------------------------------------
# DIMENSIONALITY REDUCTION - CCA

sample_train = mu.pp.sample_obs(mdata_train,0.1)[:,0:30000]
sample_test = mu.pp.sample_obs(mdata_test,0.1)[:,0:30000]

sample_train, sample_test = ut.perform_cca(sample_train, sample_test, n_components=20)
# %% ----------------------------------------------------------------
# DIMENSIONALITY REDUCTION - PCA

mdata_train, mdata_test, pca = ut.perform_pca(mdata_train, mdata_test)
# RNA ELBOW = 10 PCS
# ATAC ELBOW = 12 PCS

# %% ----------------------------------------------------------------
# SAVE PROCESSED DATA

mdata_train.write_h5mu("Data/PBMC 10k multiomic/processed_data/Dim_Red/unann_mdata_train.h5mu")
mdata_test.write_h5mu("Data/PBMC 10k multiomic/processed_data/Dim_Red/unann_mdata_test.h5mu")

# %% ----------------------------------------------------------------
# LOAD PROCESSED DATA

mdata_train = mu.read_h5mu("Data/PBMC 10k multiomic/processed_data/Dim_Red/unann_mdata_train.h5mu")
mdata_test = mu.read_h5mu("Data/PBMC 10k multiomic/processed_data/Dim_Red/unann_mdata_test.h5mu")
# %% ----------------------------------------------------------------
# ADDING ANNOTATIONS

for df, name in zip([mdata_train, mdata_test],['train', 'test']):

    # Loading annotations
    annotations = pd.read_csv('Data\PBMC 10k multiomic\PBMC-10K-celltype.txt', sep='\t', header=0, index_col=0)

    # Take intersection of cell barcodes in annotations and mdata
    common_barcodes = annotations.index.intersection(df.obs_names)

    # Filter annotations and mdata to keep only common barcodes
    annotations = annotations.loc[common_barcodes]

    # Add the annotations to the .obs DataFrame
    df.obs = pd.concat([df.obs, annotations], axis=1)
    df.obs.rename(columns={'x': 'cell_type'}, inplace=True)
    df.mod['rna'].obs['cell_type'] = df.obs['cell_type']
    df.mod['atac'].obs['cell_type'] = df.obs['cell_type']

    # Count number of NAs in cell_type column
    print(f"{name} cell_type NAs: {df.obs['cell_type'].isna().sum()}")

# %% ----------------------------------------------------------------
# GENERATING LABEL SETS

# Generating labels for training set
y_train = mdata_train.obs['cell_type'].values

# Generating labels for test set
y_test = mdata_test.obs['cell_type'].values

# CREATING DIFFERENT LABEL SETS

# MAJOR CELL TYPE SET
# Define your mapping dictionary
removing_nans = {'Platelets':np.nan, 'Double negative T cell':np.nana}
dict_map = {'CD8 Naive': 'CD8','CD8 Effector':'CD8', 'CD4 Memory': 'CD4', 'CD4 Naive': 'CD4',
            'pre-B cell':'B cell progenitor', 'CD16+ Monocytes':'Monoblast-Derived', 
            'CD14+ Monocytes':'Monoblast-Derived','Dendritic Cells':'Monoblast-Derived',
            'pDC':'Monoblast-Derived'}
# Create a vectorized function
vfunc = np.vectorize(lambda x: dict_map.get(x, x))
rem_nan_func = np.vectorize(lambda x: removing_nans.get(x, x))
# Apply the vectorized function to your array
y_train = rem_nan_func(y_train)
y_test = rem_nan_func(y_test)
y_train_mjr = vfunc(y_train)
y_test_mjr = vfunc(y_test)

# %% ----------------------------------------------------------------
# GENERATING FEATURE MATRICES

Xpca_train, Xpca_test, y_train, y_test = ut.generate_feature_matrix(mdata_train, mdata_test, 
                                                   y_train, y_test, 'PCA', 
                                                   n_components_rna=10, n_components_atac=12)   
'''
Xcca_train, Xcca_test = ut.generate_feature_matrix(mdata_train, mdata_test,
                                                   y_train, y_test, 'CCA',
                                                   n_components_rna=10, n_components_atac=12)
'''
# %% ----------------------------------------------------------------
# SAVE ML MATRICES AS PICKLES

Xpca_train.to_pickle("Data/PBMC 10k multiomic/processed_data/X_Matrices/Xpca_train.pkl")
Xpca_test.to_pickle("Data/PBMC 10k multiomic/processed_data/X_Matrices/Xpca_test.pkl")

# %% ---------------------------------------------------------------- 
# SAVE LABELS AS NUMPY ARRAYS
np.save("Data/PBMC 10k multiomic/y_train.npy", y_train)
np.save("Data/PBMC 10k multiomic/y_test.npy", y_test)
np.save("Data/PBMC 10k multiomic/y_train_mjr.npy", y_train_mjr)
np.save("Data/PBMC 10k multiomic/y_test_mjr.npy", y_test_mjr)

# %% ----------------------------------------------------------------
# VISUALISE GROUND TRUTH ANNOTATIONS RNA

sc.set_figure_params(dpi=200) # set the dpi
for df, name in zip([mdata_train.mod['rna'], mdata_test.mod['rna']],['train', 'test']):
    sc.pp.neighbors(df, n_neighbors=10, n_pcs=10) 
    sc.tl.umap(df, spread=1., min_dist=.5, random_state=11)
    sc.pl.umap(df, color='cell_type', legend_loc='on data',legend_fontsize = 4, title=f'{name} UMAP (RNA)')

# %% ----------------------------------------------------------------
#VISUALISE GROUND TRUTH ANNOTATIONS ATAC

for df, name in zip([mdata_train.mod['atac'], mdata_test.mod['atac']],['train', 'test']):
    sc.pp.neighbors(df, n_neighbors=10, n_pcs=12) 
    sc.tl.umap(df, spread=1., min_dist=.5, random_state=11)
    sc.pl.umap(df, color='cell_type', legend_loc='on data',legend_fontsize = 4, title=f'{name} UMAP (ATAC)')