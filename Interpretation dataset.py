# %% ----------------------------------------------------------------
import muon as mu
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.utils import resample
import torch
from xgboost import XGBClassifier
from sklearn.metrics import silhouette_score
import scvi
from sklearn.linear_model import LogisticRegression

import Utils as ut
from importlib import reload
from sklearn.utils import resample
import bootstrap_utils as boot
# %% ----------------------------------------------------------------
# CONFIGURATION

xgb=XGBClassifier(random_state=42)
rf=RandomForestClassifier(random_state=42, class_weight='balanced')
svm_model=svm.SVC(random_state=42, class_weight='balanced')
log_reg=LogisticRegression(random_state=42, class_weight='balanced')

EMBEDDING = 'scVI' # Choose from: PCA, CCA, scVI
GROUND_TRUTH = 'wnnL2' # PBMC: wnnL2, wnnL1, rna    Cancer: wnnL2, wnnL1, rna
CELL_TYPE = 'All' # Choose from: All, B cells, T cells, Monoblast-Derived   # Choose from: 10, 35
CL = rf # Choose from: xgb, rf, svm_model, log_reg
N_COMPONENTS_TO_TEST = 35 # Choose from: 10, 35
DATA = 'pbmc' # Choose from: pbmc, cancer

if DATA == 'pbmc':
    INPUT_ADDRESS = "PBMC 10k multiomic/QC-pbmc10k.h5mu"
    GROUND_TRUTH_SUFFIX = ''
    OUTCOME = 'multi'
elif DATA == 'cancer':
    INPUT_ADDRESS = "B cell lymphoma/QC-bcell.h5mu"
    OUTCOME = 'binary'
    if GROUND_TRUTH == 'wnnL1':
        GROUND_TRUTH_SUFFIX = '_wnnL1'
    elif GROUND_TRUTH == 'rna':
        GROUND_TRUTH_SUFFIX = '_rna'
if EMBEDDING == 'PCA':
    OBSM = 'X_pca'
elif EMBEDDING == 'CCA':
    OBSM = 'cca'
elif EMBEDDING == 'scVI':
    OBSM = 'X_scVI'
N_COMPONENTS = 35

# %% ----------------------------------------------------------------

# Loading QC data
mdata=mu.read_h5mu(f'Data/{INPUT_ADDRESS}')

#Load annotations
mdata= boot.add_annon(mdata, subset=False, data=DATA, GROUND_TRUTH = GROUND_TRUTH)
mdata.mod['rna'] = ut.pre_process_train(mdata['rna'])
mdata.mod['atac'] = ut.pre_process_train(mdata['atac'])

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Train scVI model
for mod in ['rna', 'atac']:
    # Setup the anndata object
    scvi.model.SCVI.setup_anndata(mdata.mod[mod], layer="counts")
    # Create a model
    vae = scvi.model.SCVI(mdata.mod[mod], n_latent=N_COMPONENTS)
    # Train the model
    vae.train(max_epochs=None)

    # Extract the low-dimensional representations
    mdata.mod[mod].obsm["X_scVI"] = vae.get_latent_representation(mdata.mod[mod])

# Generate labels
y = {}
if DATA == 'pbmc':
    labels = ['0','1','2']
elif DATA == 'cancer':
    labels = ['0']
for x in labels:
    # Generating labels
    y[x] = mdata.mod['rna'].obs[f'cell_type{x}'].values
y = pd.DataFrame.from_dict(y)

# Generating feature matrix for training set
X = np.concatenate((mdata.mod['rna'].obsm[OBSM][:,:N_COMPONENTS], mdata.mod['atac'].obsm[OBSM][:,:N_COMPONENTS]), axis=1)
# Convert to dataframe
X= pd.DataFrame(X, columns=[f"RNA Comp{i}" for i in range(1,N_COMPONENTS+1)] + [f"ATAC Comp{i}" for i in range(1,N_COMPONENTS+1)])
print(X.head())

# Save the feature and label datasets
X.to_csv(f'Data/PBMC 10k multiomic/Interpretation/X_{GROUND_TRUTH_SUFFIX}.csv')
# Save label dataframe
y.to_csv(f'Data/PBMC 10k multiomic/Interpretation/y_{GROUND_TRUTH_SUFFIX}.csv')