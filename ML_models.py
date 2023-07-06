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
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import Utils as ut
from importlib import reload

# %% ----------------------------------------------------------------
# LOAD PICKLED ML MATRICES

Xpca_train = pd.read_pickle('Data/PBMC 10k multiomic/processed_data/X_Matrices/Xpca_train.pkl')
Xpca_test = pd.read_pickle('Data/PBMC 10k multiomic/processed_data/X_Matrices/Xpca_test.pkl')

y_train = np.load('Data/PBMC 10k multiomic/y_train.npy', allow_pickle=True)
y_test = np.load('Data/PBMC 10k multiomic/y_test.npy', allow_pickle=True)
y_train_mjr = np.load('Data/PBMC 10k multiomic/y_train_mjr.npy', allow_pickle=True)
y_test_mjr = np.load('Data/PBMC 10k multiomic/y_test_mjr.npy', allow_pickle=True)
# %% ----------------------------------------------------------------
#ML Models

xgb=XGBClassifier(random_state=42)
rf=RandomForestClassifier(random_state=42, class_weight='balanced')
svm_model=svm.SVC(random_state=42, class_weight='balanced')

# %% ----------------------------------------------------------------
# RNA ONLY FEATURE SET

Xpca_train_rna = Xpca_train.filter(like='RNA')
Xpca_test_rna = Xpca_test.filter(like='RNA')

# %% ----------------------------------------------------------------
# RANDOM FOREST RNA ONLY

model_cl, y_pred_test = ut.model_test_main(rf,Xpca_train_rna,y_train,
                                           Xpca_test_rna,y_test, 
                                           subset = False)
ut.save_model(model_cl,'PCA\RF_RNA_only', y_pred_test, y_test)
# %% ----------------------------------------------------------------
# RANDOM FOREST RNA + ATAC

model_cl, y_pred_test = ut.model_test_main(rf,Xpca_train,y_train,
                                           Xpca_test,y_test, 
                                           subset = False)
ut.save_model(model_cl,'PCA\RF', y_pred_test, y_test)

# %% ----------------------------------------------------------------
# SVM RNA ONLY

model_cl, y_pred_test = ut.model_test_main(svm_model,Xpca_train_rna,y_train,
                                           Xpca_test_rna,y_test, 
                                           subset = True)
ut.save_model(model_cl,'PCA\RF_RNA_only', y_pred_test, y_test)
# %% ----------------------------------------------------------------
# SVM RNA + ATAC

model_cl, y_pred_test = ut.model_test_main(svm_model,Xpca_train,y_train,
                                           Xpca_test,y_test, 
                                           subset = False)
ut.save_model(model_cl,'PCA\RF', y_pred_test, y_test)
# %%
