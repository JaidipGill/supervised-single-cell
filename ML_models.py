# %% ----------------------------------------------------------------
import pickle
import scanpy as sc
import muon as mu
import anndata as ad
import h5py as h5
import pandas as pd
from muon import atac as ac
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import LocalOutlierFactor
from scipy.spatial import distance
import shap


import Utils as ut
from importlib import reload

# %% ----------------------------------------------------------------
# LOAD PICKLED ML MATRICES

Xpca_train = pd.read_pickle('Data/PBMC 10k multiomic/processed_data/X_Matrices/Xpca_train_35.pkl')
Xpca_test = pd.read_pickle('Data/PBMC 10k multiomic/processed_data/X_Matrices/Xpca_test_35.pkl')
Xpca_train_mjr = pd.read_pickle('Data/PBMC 10k multiomic/processed_data/X_Matrices/Xpca_train_mjr.pkl')
Xpca_test_mjr = pd.read_pickle('Data/PBMC 10k multiomic/processed_data/X_Matrices/Xpca_test_mjr.pkl')

y_train = np.load('Data/PBMC 10k multiomic/y_train.npy', allow_pickle=True)
y_test = np.load('Data/PBMC 10k multiomic/y_test.npy', allow_pickle=True)
y_train_mjr = np.load('Data/PBMC 10k multiomic/y_train_mjr.npy', allow_pickle=True)
y_test_mjr = np.load('Data/PBMC 10k multiomic/y_test_mjr.npy', allow_pickle=True)
# %% ----------------------------------------------------------------
#ML Models

xgb=XGBClassifier(random_state=42)
rf=RandomForestClassifier(random_state=42, class_weight='balanced')
svm_model=svm.SVC(random_state=42, class_weight='balanced')
log_reg=LogisticRegression(random_state=42, class_weight='balanced')

# %% ----------------------------------------------------------------
# RNA ONLY FEATURE SET

Xpca_train_rna = Xpca_train.filter(like='RNA')
Xpca_test_rna = Xpca_test.filter(like='RNA')
Xpca_train_rna_mjr = Xpca_train_mjr.filter(like='RNA')
Xpca_test_rna_mjr = Xpca_test_mjr.filter(like='RNA')

# %% ----------------------------------------------------------------
# CHOOSE FEATURE/LABEL SET

def choose_feature_set(feature_set, resample):
    if feature_set == 'PCA MAJOR':
        FEATURES_COMB_TRAIN = Xpca_train_mjr
        FEATURES_COMB_TEST = Xpca_test_mjr
        FEATURES_RNA_TRAIN = Xpca_train_rna_mjr
        FEATURES_RNA_TEST = Xpca_test_rna_mjr
        LABELS_TRAIN = y_train_mjr
        LABELS_TEST = y_test_mjr
    elif feature_set == 'PCA MINOR':
        FEATURES_COMB_TRAIN = Xpca_train
        FEATURES_COMB_TEST = Xpca_test
        FEATURES_RNA_TRAIN = Xpca_train_rna
        FEATURES_RNA_TEST = Xpca_test_rna
        LABELS_TRAIN = y_train
        LABELS_TEST = y_test
    if resample == True:
        smote = SMOTE(random_state=42)
        FEATURES_COMB_TRAIN, LABELS_TRAIN = smote.fit_resample(FEATURES_COMB_TRAIN, LABELS_TRAIN)
        FEATURES_RNA_TRAIN = FEATURES_COMB_TRAIN.filter(like='RNA')
    return FEATURES_COMB_TRAIN, FEATURES_COMB_TEST, FEATURES_RNA_TRAIN, FEATURES_RNA_TEST, LABELS_TRAIN, LABELS_TEST

FEATURES_COMB_TRAIN, FEATURES_COMB_TEST, FEATURES_RNA_TRAIN, FEATURES_RNA_TEST, LABELS_TRAIN, LABELS_TEST = choose_feature_set('PCA MINOR', resample = False)
# %% ----------------------------------------------------------------
# RANDOM FOREST RNA ONLY

model_cl, y_pred_test = ut.model_test_main(rf,FEATURES_RNA_TRAIN,LABELS_TRAIN,
                                           FEATURES_RNA_TEST,LABELS_TEST, 
                                           subset = False)
ut.save_model(model_cl,'PCA\RF_RNA_only', y_pred_test, LABELS_TEST)
# %% ----------------------------------------------------------------
# RANDOM FOREST RNA + ATAC

model_cl, y_pred_test = ut.model_test_main(rf,FEATURES_COMB_TRAIN,LABELS_TRAIN,
                                           FEATURES_COMB_TEST,LABELS_TEST, 
                                           subset = False)
ut.save_model(model_cl,'PCA\RF', y_pred_test, LABELS_TEST)

# %% ----------------------------------------------------------------
# SVM RNA ONLY

model_cl, y_pred_test = ut.model_test_main(svm_model,FEATURES_RNA_TRAIN,LABELS_TRAIN,
                                           FEATURES_RNA_TEST,LABELS_TEST, 
                                           subset = False)
ut.save_model(model_cl,'PCA\SVM_RNA_only', y_pred_test, LABELS_TEST)
# %% ----------------------------------------------------------------
# SVM RNA + ATAC

model_cl, y_pred_test = ut.model_test_main(svm_model,FEATURES_COMB_TRAIN,LABELS_TRAIN,
                                           FEATURES_COMB_TEST,LABELS_TEST, 
                                           subset = False)
ut.save_model(model_cl,'PCA\SVM', y_pred_test, LABELS_TEST)
# %% ----------------------------------------------------------------
# Log Reg RNA ONLY

model_cl, y_pred_test = ut.model_test_main(log_reg,FEATURES_RNA_TRAIN,LABELS_TRAIN,
                                           FEATURES_RNA_TEST,LABELS_TEST, 
                                           subset = False)
ut.save_model(model_cl,'PCA\LOGREG_RNA_only', y_pred_test, LABELS_TEST)
# %% ----------------------------------------------------------------
# Log Reg RNA + ATAC

model_cl, y_pred_test = ut.model_test_main(log_reg,FEATURES_COMB_TRAIN,LABELS_TRAIN,
                                           FEATURES_COMB_TEST,LABELS_TEST, 
                                           subset = False)
ut.save_model(model_cl,'PCA\LOGREG', y_pred_test, LABELS_TEST)

# %% ----------------------------------------------------------------
# VISUALIZE EMBEDDINGS

embedding = ut.visualise_embeddings(FEATURES_COMB_TRAIN, LABELS_TRAIN)

# %% ----------------------------------------------------------------
# FEATURE IMPORTANCE 

# Load model
# open a file, where you stored the pickled data
#file = open('Supervised Models\PCA\RF_RNA_only.pickle', 'rb')
# dump information to that file
#model_cl = pickle.load(file)

shap_values, feat_imp = ut.feature_importance(model_cl, FEATURES_COMB_TEST, mdata_train)

# %% ----------------------------------------------------------------
# REMOVING OUTLIERS

filtered_df = ut.remove_outliers('LOF',FEATURES_COMB_TRAIN, LABELS_TRAIN, FEATURES_COMB_TEST, LABELS_TEST, 95)

# %%

ut.visualise_embeddings(filtered_df.drop(['label'], axis =1), filtered_df['label'])
# %%

sns.scatterplot(x=FEATURES_COMB_TRAIN['ATAC PC1'], y=FEATURES_COMB_TRAIN['ATAC PC2'], hue=LABELS_TRAIN)
plt.show()
# %%

corr_matrix = FEATURES_COMB_TRAIN.corr()
sns.heatmap(corr_matrix)
plt.show()

# %%
FEATURES_COMB_TRAIN['label'] = LABELS_TRAIN.tolist()
FEATURES_COMB_TEST['label'] = LABELS_TEST.tolist()
df = pd.concat([FEATURES_COMB_TRAIN, FEATURES_COMB_TEST], axis=0)
ut.visualise_embeddings(df.drop(['label'],axis=1), df['label'])
# %%
