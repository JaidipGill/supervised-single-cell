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

from Utils import train_test_split_mdata, pre_process
# %% ----------------------------------------------------------------
#LOAD DATA

# Loading QC data
mdata=mu.read_h5mu("Data/PBMC 10k multiomic/processed_data/QC-pbmc10k.h5mu")

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
# DIMENSIONALITY REDUCTION OF RNA

for df, name in zip([mdata_train, mdata_test],['train', 'test']):
    st=time.process_time()
    sc.tl.pca(df.mod['rna'])
    et=time.process_time()
    print(f"{name} PCA (RNA) took {et-st} seconds")
    sc.pl.pca_overview(mdata_train.mod['rna'])
    sc.pl.pca(df.mod['rna'], color=['CD2', 'CD79A', 'KLF4', 'IRF8'])

# %% ----------------------------------------------------------------
# DIMENSIONALITY REDUCTION OF ATAC

for df, name in zip([mdata_train, mdata_test],['train', 'test']):
    st=time.process_time()
    sc.tl.pca(df.mod['atac'])
    et=time.process_time()
    print(f"{name} PCA (ATAC) took {et-st} seconds")
    sc.pl.pca_overview(df.mod['atac'])

# %% ----------------------------------------------------------------
# CHOOSING N_PCS

sc.pl.pca_variance_ratio(mdata_train.mod['atac'], log=True) # 14 PCs
print(np.cumsum(mdata_train.mod['atac'].uns['pca']['variance_ratio'])[:14]) # Explains 6.3% of variance
sc.pl.pca_variance_ratio(mdata_train.mod['rna'], log=True) # 12 PCs
print(np.cumsum(mdata_train.mod['rna'].uns['pca']['variance_ratio'])[:12]) # Explains 10.8% of variance

# %% ----------------------------------------------------------------
# SAVE PROCESSED DATA

mdata_train.write_h5mu("Data/PBMC 10k multiomic/processed_data/PCA/unann_mdata_train.h5mu")
mdata_test.write_h5mu("Data/PBMC 10k multiomic/processed_data/PCA/unann_mdata_test.h5mu")

# %% ----------------------------------------------------------------
# LOAD PROCESSED DATA

mdata_train = mu.read_h5mu("Data/PBMC 10k multiomic/processed_data/PCA/unann_mdata_train.h5mu")
mdata_test = mu.read_h5mu("Data/PBMC 10k multiomic/processed_data/PCA/unann_mdata_test.h5mu")
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
# VISUALISE ANNOTATIONS RNA

sc.set_figure_params(dpi=200) # set the dpi
for df, name in zip([mdata_train.mod['rna'], mdata_test.mod['rna']],['train', 'test']):
    sc.pp.neighbors(df, n_neighbors=10, n_pcs=12) 
    sc.tl.umap(df, spread=1., min_dist=.5, random_state=11)
    sc.pl.umap(df, color='cell_type', legend_loc='on data',legend_fontsize = 4, title=f'{name} UMAP (RNA)')

# %% ----------------------------------------------------------------
#VISUALISE ANNOTATIONS ATAC

for df, name in zip([mdata_train.mod['atac'], mdata_test.mod['atac']],['train', 'test']):
    sc.pp.neighbors(df, n_neighbors=10, n_pcs=14) 
    sc.tl.umap(df, spread=1., min_dist=.5, random_state=11)
    sc.pl.umap(df, color='cell_type', legend_loc='on data',legend_fontsize = 4, title=f'{name} UMAP (ATAC)')

# %% ----------------------------------------------------------------
# GENERATING ML MATRICES

# Generating labels for training set
y_train = mdata_train.obs['cell_type'].values

# Generating labels for test set
y_test = mdata_test.obs['cell_type'].values

# Generating feature matrix for training set
X_train = np.concatenate((mdata_train.mod['rna'].obsm['X_pca'][:,:12], mdata_train.mod['atac'].obsm['X_pca'][:,:14]), axis=1)
# Convert to dataframe
X_train = pd.DataFrame(X_train, columns=[f"RNA PC{i}" for i in range(1,13)] + [f"ATAC PC{i}" for i in range(1,15)])
print(X_train.head())

# Generating feature matrix for test set
X_test = np.concatenate((mdata_test.mod['rna'].obsm['X_pca'][:,:12], mdata_test.mod['atac'].obsm['X_pca'][:,:14]), axis=1)
# Convert to dataframe
X_test = pd.DataFrame(X_test, columns=[f"RNA PC{i}" for i in range(1,13)] + [f"ATAC PC{i}" for i in range(1,15)])
print(X_test.head())

# %% ----------------------------------------------------------------
# SAVE ML MATRICES AS PICKLES

X_train.to_pickle("Data/PBMC 10k multiomic/processed_data/PCA/X_train.pkl")
X_test.to_pickle("Data/PBMC 10k multiomic/processed_data/PCA/X_test.pkl")
np.save("Data/PBMC 10k multiomic/processed_data/PCA/y_train.npy", y_train)
np.save("Data/PBMC 10k multiomic/processed_data/PCA/y_test.npy", y_test)
# %%
