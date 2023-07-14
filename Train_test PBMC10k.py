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

sample_train, sample_test = ut.perform_cca(sample_train, sample_test, n_components=20)
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
# DIMENSIONALITY REDUCTION - SCVI

mdata_train, mdata_test = ut.scvi_process(mdata_train, mdata_test, epochs=3)
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
dict_map = {'CD8 Naive': 'CD8','CD8 Effector':'CD8', 'CD4 Memory': 'CD4', 'CD4 Naive': 'CD4',
            'pre-B cell':'B cell progenitor', 'CD16+ Monocytes':'Monoblast-Derived', 
            'CD14+ Monocytes':'Monoblast-Derived','Dendritic Cells':'Monoblast-Derived',
            'pDC':'Monoblast-Derived'}

# Apply the mapping dictionaries to your Series
y_train_mjr = y_train.replace(dict_map)
y_test_mjr = y_test.replace(dict_map)

# Remove NaN values again if any were introduced
y_train_mjr.replace(adding_nans, inplace=True)
y_test_mjr.replace(adding_nans, inplace=True)
y_train.replace(adding_nans, inplace=True)
y_test.replace(adding_nans, inplace=True)

# %% ----------------------------------------------------------------
# GENERATING FEATURE MATRICES

Xpca_train, Xpca_test, y_train, y_test = ut.generate_feature_matrix(mdata_train, mdata_test, 
                                                   y_train, y_test, 'PCA', 
                                                   n_components_rna=25, n_components_atac=25)   
'''
Xcca_train, Xcca_test = ut.generate_feature_matrix(mdata_train, mdata_test,
                                                   y_train, y_test, 'CCA',
                                                   n_components_rna=10, n_components_atac=12)
'''
# %% ----------------------------------------------------------------
# SAVE FULL DATA

Xpca_train.to_pickle("Data/PBMC 10k multiomic/processed_data/X_Matrices/Xpca_train_25.pkl")
Xpca_test.to_pickle("Data/PBMC 10k multiomic/processed_data/X_Matrices/Xpca_test_25.pkl")
np.save("Data/PBMC 10k multiomic/y_train.npy", y_train)
np.save("Data/PBMC 10k multiomic/y_test.npy", y_test)
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