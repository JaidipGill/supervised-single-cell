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
# DIMENSIONALITY REDUCTION - AUTOENCODER

# Convert numpy array to torch tensor
input_data_train = torch.from_numpy(mdata_train['rna'].X).float()

# Specify batch size
batch_size = 100
# Assume you have input data X and its dimension is 1000
input_dim = 3124
encoding_dim = 32
n_epochs = 100

# Create DataLoader
train_data = torch.utils.data.TensorDataset(input_data_train, input_data_train)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
            nn.Sigmoid(), 
            nn.Linear(128, 64), 
            nn.Sigmoid(), 
            nn.Linear(64, encoding_dim),
            nn.Sigmoid(),
            nn.Linear(encoding_dim, encoding_dim),
            nn.Sigmoid(),
            nn.Linear(encoding_dim, encoding_dim),
            nn.Sigmoid(),
            nn.Linear(encoding_dim, encoding_dim),
            nn.Sigmoid(),
            nn.Linear(encoding_dim, encoding_dim),
            nn.Sigmoid(),
            nn.Linear(encoding_dim, encoding_dim),
            nn.Sigmoid(),
            nn.Linear(encoding_dim, encoding_dim),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim),
            nn.Sigmoid(),
            nn.Linear(encoding_dim, encoding_dim),
            nn.Sigmoid(),
            nn.Linear(encoding_dim, encoding_dim),
            nn.Sigmoid(),
            nn.Linear(encoding_dim, encoding_dim),
            nn.Sigmoid(),
            nn.Linear(encoding_dim, encoding_dim),
            nn.Sigmoid(),
            nn.Linear(encoding_dim, encoding_dim),
            nn.Sigmoid(),
            nn.Linear(encoding_dim, 64),
            nn.Sigmoid(),
            nn.Linear(64, 128),
            nn.Sigmoid(),
            nn.Linear(128, 256),
            nn.Sigmoid(), 
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def encode(self, x):
        return self.encoder(x)


# Create the autoencoder model
autoencoder = Autoencoder(input_dim, encoding_dim)

# Set loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# If you have a GPU, you might need to send the data to GPU
if torch.cuda.is_available():
    input_data_train = input_data_train.cuda()
    autoencoder = autoencoder.cuda()
    print('using GPU')

# Define a list to hold the losses
losses = []

# Now you can use this tensor as input to your model
for epoch in range(n_epochs):
    epoch_loss = 0
    for batch in train_loader:
        inputs, _ = batch 
        output = autoencoder(inputs)
        loss = criterion(output, inputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    average_epoch_loss = epoch_loss / len(train_loader)
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, n_epochs, average_epoch_loss))
    losses.append(average_epoch_loss)

# Extract latent features
latent_features = autoencoder.encode(input_data_train).detach().cpu().numpy()
# convert numpy array to pandas dataframe
latent_features = pd.DataFrame(latent_features)

plt.figure()
plt.plot(range(n_epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over time')
plt.show()

print('Training is finished!')

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

def add_annon(mdata_train, mdata_test, wnn):
    for df, name in zip([mdata_train, mdata_test],['train', 'test']):

        # Loading annotations
        if wnn == False:
            annotations = pd.read_csv('Data\PBMC 10k multiomic\PBMC-10K-celltype.txt', sep='\t', header=0, index_col=0)
        elif wnn == True:
            annotations = pd.read_csv('Data\PBMC 10k multiomic\WNN-PBMC-10K-celltype.csv', sep='\t', header=0, index_col='index')
        # Take intersection of cell barcodes in annotations and mdata
        print(annotations)
        common_barcodes = annotations.index.intersection(df.obs_names)
        print(annotations.index)
        # Filter annotations and mdata to keep only common barcodes
        annotations = annotations.loc[common_barcodes]
        print(annotations)
        # Add the annotations to the .obs DataFrame
        df.obs = pd.concat([df.obs, annotations], axis=1)
        df.obs.rename(columns={'x': 'cell_type'}, inplace=True)
        df.mod['rna'].obs['cell_type'] = df.obs['cell_type']
        df.mod['atac'].obs['cell_type'] = df.obs['cell_type']

        # Count number of NAs in cell_type column
        print(f"{name} cell_type NAs: {df.obs['cell_type'].isna().sum()}")
    return mdata_train, mdata_test

mdata_train, mdata_test = add_annon(mdata_train, mdata_test, wnn = False)

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

def save_data(labels, embedding, Xpca_train, Xpca_test, y_train, y_test, XscVI_train, XscVI_test):
    if embedding == 'PCA' and labels == 'rna':
        Xpca_train.to_pickle("Data/PBMC 10k multiomic/processed_data/X_Matrices/Xpca_train_35_RAW.pkl")
        Xpca_test.to_pickle("Data/PBMC 10k multiomic/processed_data/X_Matrices/Xpca_test_35_RAW.pkl")
        np.save("Data/PBMC 10k multiomic/y_train.npy", y_train)
        np.save("Data/PBMC 10k multiomic/y_test.npy", y_test)
    elif embedding == 'scVI' and labels == 'rna':
        XscVI_train.to_pickle("Data/PBMC 10k multiomic/processed_data/X_Matrices/XscVI_train_35.pkl")
        XscVI_test.to_pickle("Data/PBMC 10k multiomic/processed_data/X_Matrices/XscVI_test_35.pkl")
        np.save("Data/PBMC 10k multiomic/y_train.npy", y_train)
        np.save("Data/PBMC 10k multiomic/y_test.npy", y_test)
    elif embedding == 'PCA' and labels == 'wnn':
        Xpca_train.to_pickle("Data/PBMC 10k multiomic/processed_data/X_Matrices/Xpca_train_35_wnn.pkl")
        Xpca_test.to_pickle("Data/PBMC 10k multiomic/processed_data/X_Matrices/Xpca_test_35_wnn.pkl")
        np.save("Data/PBMC 10k multiomic/y_train_wnn.npy", y_train)
        np.save("Data/PBMC 10k multiomic/y_test_wnn.npy", y_test)
    elif embedding == 'scVI' and labels =='wnn':
        np.save("Data/PBMC 10k multiomic/y_train_wnn.npy", y_train)
        np.save("Data/PBMC 10k multiomic/y_test_wnn.npy", y_test)
        XscVI_train.to_pickle("Data/PBMC 10k multiomic/processed_data/X_Matrices/XscVI_train_35_wnn.pkl")
        XscVI_test.to_pickle("Data/PBMC 10k multiomic/processed_data/X_Matrices/XscVI_test_35_wnn.pkl")
    elif embedding == 'CCA' and labels == 'rna':
        np.save("Data/PBMC 10k multiomic/y_train.npy", y_train)
        np.save("Data/PBMC 10k multiomic/y_test.npy", y_test)
        Xcca_train.to_pickle("Data/PBMC 10k multiomic/processed_data/X_Matrices/Xcca_train_35.pkl")
        Xcca_test.to_pickle("Data/PBMC 10k multiomic/processed_data/X_Matrices/Xcca_test_35.pkl")
    elif embedding == 'CCA' and labels == 'wnn':
        np.save("Data/PBMC 10k multiomic/y_train_wnn.npy", y_train)
        np.save("Data/PBMC 10k multiomic/y_test_wnn.npy", y_test)
        Xcca_train.to_pickle("Data/PBMC 10k multiomic/processed_data/X_Matrices/Xcca_train_35_wnn.pkl")
        Xcca_test.to_pickle("Data/PBMC 10k multiomic/processed_data/X_Matrices/Xcca_test_35_wnn.pkl")
save_data('rna', 'CCA', Xpca_train=None, Xpca_test=None, y_train=y_train, y_test=y_test, XscVI_train=None, 
          XscVI_test=None, Xcca_train=Xcca_train, Xcca_test=Xcca_test)
# %% ---------------------------------------------------------------- 
# SAVE MAJOR CELL TYPE DATA
Xpca_train_mjr, Xpca_test_mjr, y_train_mjr, y_test_mjr = ut.generate_feature_matrix(mdata_train, mdata_test, 
                                                   y_train_mjr, y_test_mjr, 'PCA', 
                                                   n_components_rna=10, n_components_atac=12)   
Xpca_train_mjr.to_pickle("Data/PBMC 10k multiomic/processed_data/X_Matrices/Xpca_train_mjr.pkl")
Xpca_test_mjr.to_pickle("Data/PBMC 10k multiomic/processed_data/X_Matrices/Xpca_test_mjr.pkl")
np.save("Data/PBMC 10k multiomic/y_train_mjr.npy", y_train_mjr)
np.save("Data/PBMC 10k multiomic/y_test_mjr.npy", y_test_mjr)

# %% ----------------------------------------------------------------
# WNN CLUSTERING

# Pre-process entire dataset
mdata.mod['rna'] = ut.pre_process_train(mdata['rna'])
mdata.mod['atac'] = ut.pre_process_train(mdata['atac'])
# %%
# PCA
sc.tl.pca(mdata.mod['rna'])
sc.tl.pca(mdata.mod['atac'])
# %%
# Calculate weighted nearest neighbors
sc.pp.neighbors(mdata['rna'])
sc.pp.neighbors(mdata['atac'])
mu.pp.neighbors(mdata, key_added='wnn', add_weights_to_modalities = True)
# %%
# PLot WNN UMAP
mdata.uns['wnn']['params']['use_rep']
mu.tl.umap(mdata, neighbors_key='wnn', random_state=10)
mu.pl.umap(mdata, color=['rna:mod_weight', 'atac:mod_weight'], cmap='RdBu')
# %% 
# Clustering WNN
sc.tl.leiden(mdata, resolution=1.0, neighbors_key='wnn', key_added='leiden_wnn')
sc.pl.umap(mdata, color='leiden_wnn', legend_loc='on data')
sc.pl.violin(mdata, groupby='leiden_wnn', keys='atac:mod_weight')
# %%
# Annotations
mdata.mod['rna'].obs['leiden_wnn']=mdata.obs['leiden_wnn']
# Differential expression analysis between the identified clusters
sc.tl.rank_genes_groups(mdata.mod['rna'], 'leiden_wnn', method='wilcoxon')
result = mdata.mod['rna'].uns['rank_genes_groups']
groups = result['names'].dtype.names
pd.set_option('display.max_columns', 50)
# Create a DataFrame that contains the top 10 genes for each cluster, along with their corresponding p-values. 
# Each cluster's results are in separate columns. 
pd.DataFrame(
    {group + '_' + key[:1]: result[key][group]
    for group in groups for key in ['names', 'pvals']}).head(10) 
# %%
marker_genes = {
    'CD4+ Naive T': {'TCF7', 'CD4', 'CCR7', 'IL7R', 'FHIT', 'LEF1', 'MAL', 'NOSIP', 'LDHB', 'PIK3IP1'},
    'CD14+ Monocytes': {'S100A9', 'CTSS', 'S100A8', 'LYZ', 'VCAN', 'S100A12', 'IL1B', 'CD14', 'G0S2', 'FCN1'},
    'CD16+ Monocyte': {'CDKN1C', 'FCGR3A', 'PTPRC', 'LST1', 'IER5', 'MS4A7', 'RHOC', 'IFITM3', 'AIF1', 'HES4'},
    'CD8+ Naive T': {'CD8B', 'S100B', 'CCR7', 'RGS10', 'NOSIP', 'LINC02446', 'LEF1', 'CRTAM', 'CD8A', 'OXNAD1'},
    'NK cell': {'NKG7', 'KLRD1', 'TYROBP', 'GNLY', 'FCER1G', 'PRF1', 'CD247', 'KLRF1', 'CST7', 'GZMB'},
    'Dendritic Cells': {'CD74', 'HLA-DPA1', 'HLA-DPB1', 'HLA-DQA1', 'CCDC88A', 'HLA-DRA', 'HLA-DMA', 'CST3', 'HLA-DQB1', 'HLA-DRB1'},
    'pre-B cell': {'CD10', 'CD22', 'CD34', 'CD38', 'CD48', 'CD79a', 'CD127', 'CD184', 'RAG', 'TdT', 'Vpre-B', 'Pax5', 'EBF'},
    'CD8+ Effector Memory T': {'CCL5', 'GZMH', 'CD8A', 'TRAC', 'KLRD1', 'NKG7', 'GZMK', 'CST7', 'CD8B', 'TRGC2'},
    'pDC': {'ITM2C', 'PLD4', 'SERPINF1', 'LILRA4', 'IL3RA', 'TPM2', 'MZB1', 'SPIB', 'IRF4', 'SMPD3'},
    'CD4+ Central Memory T': {'IL7R', 'TMSB10', 'CD4', 'ITGB1', 'LTB', 'TRAC', 'AQP3', 'LDHB', 'IL32', 'MAL'}
} # Marker genes from Azimuth: https://azimuth.hubmapconsortium.org/references/#Human%20-%20PBMC
sc.tl.marker_gene_overlap(mdata.mod['rna'], marker_genes,method='overlap_count')
# %%
# Relabel
new_cluster_names = {
    "0": "CD4+ Naive T", "1": "CD8+ Naive T", "2": "CD14+ Monocytes", "3": "CD14+ Monocytes", 
    "4": "CD4+ Naive T", "5": "CD14+ Monocytes", "6": "CD8+ Effector Memory T", "7": "CD4+ Central Memory T", 
    "8": "CD4+ Central Memory T", "9": "Dendritic Cells", "10": "Dendritic Cells", 
    "11": "NK cell", "12": "CD8+ Effector Memory T", "13": "Dendritic Cells", "14": "CD8+ Naive T",
    "15": "CD14+ Monocytes", "16": "Dendritic Cells", "17": "CD14+ Monocytes", "18": "CD8+ Effector Memory T",
    "19": "pDC", "20": "pDC", "21": "pDC"
}
mdata.mod['rna'].obs['cell_type'] = mdata.mod['rna'].obs.leiden_wnn.astype("str").values
mdata.mod['rna'].obs.cell_type = mdata.mod['rna'].obs.celltype.replace(new_cluster_names)
'''
mdata.mod['rna'].obs.celltype = mdata.mod['rna'].obs.celltype.astype("category")
mdata.mod['rna'].obs.celltype.cat.reorder_categories([
    'CD4+ Naive T', 'CD4+ Central Memory T', 'CD8+ Naive T',
    'CD8+ Effector Memory T', 'NK cell', 'pre-B cell',
    'CD14+ Monocytes', 'CD16+ Monocyte',
    'pDC', 'Dendritic Cells','???'])
'''
mdata.obs['cell_type']=mdata.mod['rna'].obs['cell_type']
sc.pl.umap(mdata, color="cell_type", legend_loc="on data")
# %%
# Generate alternate annotation txt file
# Extract index and a column
df_to_save = mdata.obs.reset_index()[['index', 'cell_type']]
# Save to a txt file, separator can be specified, here it's a space
df_to_save.to_csv('Data\PBMC 10k multiomic\WNN-PBMC-10K-celltype.csv', index=True, header=True, sep='\t')
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