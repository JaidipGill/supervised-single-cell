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

# %% ----------------------------------------------------------------
# LOAD PICKLED ML MATRICES

X_train = pd.read_pickle('Data/PBMC 10k multiomic/processed_data/PCA/X_train.pkl')
X_test = pd.read_pickle('Data/PBMC 10k multiomic/processed_data/PCA/X_test.pkl')
y_train = np.load('Data/PBMC 10k multiomic/processed_data/PCA/y_train.npy', allow_pickle=True)
y_test = np.load('Data/PBMC 10k multiomic/processed_data/PCA/y_test.npy', allow_pickle=True)

# %% ----------------------------------------------------------------
#ML Models

xgb=XGBClassifier(random_state=42)
rf=RandomForestClassifier(random_state=42, class_weight='balanced')
svm_model=svm.SVC(random_state=42, class_weight='balanced')

# %% ----------------------------------------------------------------
# REMOVE NANS

# Create DataFrame from X_train and X_test
train_data = pd.DataFrame(X_train)
test_data = pd.DataFrame(X_test)

# Add y_train and y_test as columns
train_data['label'] = y_train
test_data['label'] = y_test

# Removing rows with NaN values
train_data = train_data.dropna(subset=['label'])
test_data = test_data.dropna(subset=['label'])

# Separate X_train, y_train, X_test, and y_test from the updated DataFrame
X_train = train_data.iloc[:, :-1]
y_train = train_data['label'].to_numpy()

X_test = test_data.iloc[:, :-1]
y_test = test_data['label'].to_numpy()

# %% ----------------------------------------------------------------
# RNA ONLY FEATURE SET

X_train_rna = X_train.filter(like='RNA')
X_test_rna = X_test.filter(like='RNA')

# %% ----------------------------------------------------------------
# RANDOM FOREST RNA + ATAC

model_cl, y_pred_test = ut.model_test_main(rf,X_train,y_train,
                                           X_test,y_test, 
                                           subset = False)
ut.save_model(model_cl,'PCA\RF', y_pred_test, y_test)
# %% ----------------------------------------------------------------
# RANDOM FOREST RNA ONLY

model_cl, y_pred_test = ut.model_test_main(rf,X_train_rna,y_train,
                                           X_test_rna,y_test, 
                                           subset = False)
ut.save_model(model_cl,'PCA\RF_RNA_only', y_pred_test, y_test)

# %% ----------------------------------------------------------------
# SVM RNA + ATAC

model_cl, y_pred_test = ut.model_test_main(svm_model,X_train,y_train,
                                           X_test,y_test, 
                                           subset = False)
ut.save_model(model_cl,'PCA\RF', y_pred_test, y_test)
# %% ----------------------------------------------------------------
# SVM RNA ONLY

model_cl, y_pred_test = ut.model_test_main(svm_model,X_train_rna,y_train,
                                           X_test_rna,y_test, 
                                           subset = False)
ut.save_model(model_cl,'PCA\RF_RNA_only', y_pred_test, y_test)
# %%
