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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import scvi

import Utils as ut
from importlib import reload
# %% ----------------------------------------------------------------
#LOAD DATA

# Loading QC data
mdata=mu.read_h5mu("Data/PBMC 10k multiomic/processed_data/QC-pbmc10k.h5mu")
# %% ----------------------------------------------------------------
# WNN CLUSTERING

mdata = ut.wnn_cluster(mdata)

# %%
# ANNOTATION BY SUBTYPE LEVEL

ut.annotate_clusters(mdata, level=2)

# %% ----------------------------------------------------------------
# RELOAD DATA

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

sample_train, sample_test = ut.perform_cca(sample_train, sample_test, n_components=50)
# %% ----------------------------------------------------------------
# DIMENSIONALITY REDUCTION - PCA

mdata_train, mdata_test, pca = ut.perform_pca(mdata_train, mdata_test, raw=False, components = 50)
# RNA ELBOW = 10 PCS
# ATAC ELBOW = 12 PCS

# %% ----------------------------------------------------------------
# SAVE PROCESSED DATA

mdata_train.write_h5mu("Data/PBMC 10k multiomic/processed_data/Dim_Red/unann_mdata_train.h5mu")
mdata_test.write_h5mu("Data/PBMC 10k multiomic/processed_data/Dim_Red/unann_mdata_test.h5mu")

# %% ----------------------------------------------------------------
# LOAD DIMENSIONALITY REDUCTION - SCVI

#mdata_train, mdata_test = ut.scvi_process(mdata_train, mdata_test, epochs=1)

mdata_train = mu.read_h5mu("Data/PBMC 10k multiomic/processed_data/Dim_Red/scvi_mdata_train_35.h5mu")
mdata_test = mu.read_h5mu("Data/PBMC 10k multiomic/processed_data/Dim_Red/scvi_mdata_test_35.h5mu")
# %% 
# LOAD DIMENSIONALITY REDUCTION - PCA

mdata_train = mu.read_h5mu("Data/PBMC 10k multiomic/processed_data/Dim_Red/unann_mdata_train.h5mu")
mdata_test = mu.read_h5mu("Data/PBMC 10k multiomic/processed_data/Dim_Red/unann_mdata_test.h5mu")

# %%
# LOAD DIMENSIOANLITY REDUCTION - CCA

mdata_train = mu.read_h5mu("Data/PBMC 10k multiomic/processed_data/Dim_Red/CCA_mdata_train.h5mu")
mdata_test = mu.read_h5mu("Data/PBMC 10k multiomic/processed_data/Dim_Red/CCA_mdata_test.h5mu")
# %% ----------------------------------------------------------------
# ADDING ANNOTATIONS

mdata_train, mdata_test = ut.add_annon(mdata_train, mdata_test, wnn = 2)

# %% ----------------------------------------------------------------
# GENERATING LABEL SETS

# Generating labels
y_train = mdata_train.obs['cell_type'].values
y_test = mdata_test.obs['cell_type'].values

# Converting to pandas series
y_train = pd.Series(y_train)
y_test = pd.Series(y_test)

# CREATING DIFFERENT LABEL SETS

# MAJOR CELL TYPE SET
# Define your mapping dictionary

adding_nans = {'Platelets':np.nan, 'Double negative T cell':np.nan}
'''
dict_map = {'CD8 Naive': 'CD8','CD8 Effector':'CD8', 'CD4 Memory': 'CD4', 'CD4 Naive': 'CD4',
            'pre-B cell':'B cell progenitor', 'CD16+ Monocytes':'Monoblast-Derived', 
            'CD14+ Monocytes':'Monoblast-Derived','Dendritic Cells':'Monoblast-Derived',
            'pDC':'Monoblast-Derived'}

# Apply the mapping dictionaries to your Series
y_train_mjr = y_train.replace(dict_map)
y_test_mjr = y_test.replace(dict_map)
'''
# Remove NaN values again if any were introduced
#y_train_mjr.replace(adding_nans, inplace=True)
#y_test_mjr.replace(adding_nans, inplace=True)
y_train.replace(adding_nans, inplace=True)
y_test.replace(adding_nans, inplace=True)

# %% ----------------------------------------------------------------
# GENERATING FEATURE MATRICES

Xpca_train, Xpca_test, y_train, y_test = ut.generate_feature_matrix(mdata_train, mdata_test, 
                                                   y_train, y_test, 'PCA', 
                                                   n_components_rna=35, n_components_atac=35)   
# %% 
Xcca_train, Xcca_test, y_train, y_test = ut.generate_feature_matrix(mdata_train, mdata_test,
                                                   y_train, y_test, 'CCA',
                                                   n_components_rna=35, n_components_atac=35)

# %% 
XscVI_train, XscVI_test, y_train, y_test = ut.generate_feature_matrix(mdata_train, mdata_test, 
                                                   y_train, y_test, 'scVI', 
                                                   n_components_rna=35, n_components_atac=35) 
# %% ----------------------------------------------------------------
# SAVE FULL DATA

ut.save_data('wnnL2', 'scVI', Xpca_train=None, Xpca_test=None, y_train=y_train, y_test=y_test, XscVI_train=XscVI_train, 
          XscVI_test=XscVI_test, Xcca_train=None, Xcca_test=None)
# %% ---------------------------------------------------------------- 
# SAVE MAJOR CELL TYPE DATA
'''
Xpca_train_mjr, Xpca_test_mjr, y_train_mjr, y_test_mjr = ut.generate_feature_matrix(mdata_train, mdata_test, 
                                                   y_train_mjr, y_test_mjr, 'PCA', 
                                                   n_components_rna=10, n_components_atac=12)   
Xpca_train_mjr.to_pickle("Data/PBMC 10k multiomic/processed_data/X_Matrices/Xpca_train_mjr.pkl")
Xpca_test_mjr.to_pickle("Data/PBMC 10k multiomic/processed_data/X_Matrices/Xpca_test_mjr.pkl")
np.save("Data/PBMC 10k multiomic/y_train_mjr.npy", y_train_mjr)
np.save("Data/PBMC 10k multiomic/y_test_mjr.npy", y_test_mjr)
'''
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