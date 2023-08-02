# %% ----------------------------------------------------------------
import scanpy as sc
import muon as mu
import anndata as ad
import h5py as h5
import pandas as pd
from muon import atac as ac
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.utils import resample
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from collections import defaultdict
import time

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

#for cells in ['B cells', 'T cells', 'Monoblast-Derived', 'All']:
EMBEDDING = 'PCA' # Choose from: PCA, CCA, scVI
GROUND_TRUTH = 'wnnL2' # Choose from: wnnL2, wnnL1, rna
CELL_TYPE = 'All' # Choose from: B cells, T cells, Monoblast-Derived, All
N_COMPONENTS = 35   # Choose from: 10, 35
CL = svm_model # Choose from: xgb, rf, svm_model, log_reg
DATA = 'pbmc' # Choose from: pbmc, cancer

if DATA == 'pbmc':
    INPUT_ADDRESS = "PBMC 10k multiomic/processed_data/QC-pbmc10k.h5mu"
elif DATA == 'cancer':
    INPUT_ADDRESS = "B cell lymphoma/QC-bcell.h5mu"
if EMBEDDING == 'PCA':
    OBSM = 'X_pca'
elif EMBEDDING == 'CCA':
    OBSM = 'cca'
elif EMBEDDING == 'scVI':
    OBSM = 'X_scVI'
SUFFIX = f'{DATA}_{CL.__class__.__name__}_{EMBEDDING}_{GROUND_TRUTH}_{CELL_TYPE}_{N_COMPONENTS}'
# %% ----------------------------------------------------------------
# BOOTSTRAP SAMPLES

N = 10

for i in range(0,N):
    
    # Loading QC data
    mdata=mu.read_h5mu(f'Data/{INPUT_ADDRESS}')

    #Load annotations
    mdata= boot.add_annon(mdata, subset=False, data=DATA)

    # Bootstrap sample and pre-process without data leakage
    mdata_train, mdata_test = boot.train_test_split_mdata(mdata, seed=i)
    mdata_train.mod['rna'] = ut.pre_process_train(mdata_train['rna'])
    mdata_test.mod['rna'] = ut.pre_process_test(mdata_test['rna'], mdata_train['rna'])
    mdata_train.mod['atac'] = ut.pre_process_train(mdata_train['atac'])
    mdata_test.mod['atac'] = ut.pre_process_test(mdata_test['atac'], mdata_train['atac'])
    print(mdata_train)
    print(mdata_test)

    #Generate embedding
    if EMBEDDING == 'PCA':
        mdata_train, mdata_test, pca = ut.perform_pca(mdata_train, mdata_test, raw=False, components = N_COMPONENTS)
    elif EMBEDDING == 'scVI':  
        mdata_train, mdata_test = ut.scvi_process(mdata_train, mdata_test, epochs=None, n_latent=N_COMPONENTS)

    # Generate labels
    y_train = {}
    y_test = {}
    if DATA == 'pbmc':
        labels = ['0','1','2']
    elif DATA == 'cancer':
        labels = ['0']
    for x in labels:
        # Generating labels
        y_train[x] = mdata_train.mod['rna'].obs[f'cell_type{x}'].values
        y_test[x] = mdata_test.mod['rna'].obs[f'cell_type{x}'].values
    y_train = pd.DataFrame.from_dict(y_train)
    y_test = pd.DataFrame.from_dict(y_test)

    # Generating feature matrix for training set
    X_train = np.concatenate((mdata_train.mod['rna'].obsm[OBSM][:,:N_COMPONENTS], mdata_train.mod['atac'].obsm[OBSM][:,:N_COMPONENTS]), axis=1)
    # Convert to dataframe
    X_train = pd.DataFrame(X_train, columns=[f"RNA Comp{i}" for i in range(1,N_COMPONENTS+1)] + [f"ATAC Comp{i}" for i in range(1,N_COMPONENTS+1)])
    print(X_train.head())

    # Generating feature matrix for test set
    X_test = np.concatenate((mdata_test.mod['rna'].obsm[OBSM][:,:N_COMPONENTS], mdata_test.mod['atac'].obsm[OBSM][:,:N_COMPONENTS]), axis=1)
    # Convert to dataframe
    X_test = pd.DataFrame(X_test, columns=[f"RNA Comp{i}" for i in range(1,N_COMPONENTS+1)] + [f"ATAC Comp{i}" for i in range(1,N_COMPONENTS+1)])

    # Standardization
    sclr = StandardScaler().fit(X_train)
    X_train = sclr.transform(X_train)
    X_test = sclr.transform(X_test)

    print(X_test.head())
    print(X_train)
    print(X_test)
    print(y_train)
    print(y_test)

    # Save the feature and label datasets
    X_train.to_pickle(f'Data/{INPUT_ADDRESS.split("/")[0]}/Bootstrap_X/X_train_{EMBEDDING}_{i}.pkl')
    X_test.to_pickle(f'Data/{INPUT_ADDRESS.split("/")[0]}/Bootstrap_X/X_test_{EMBEDDING}_{i}.pkl')
    # Save label dataframe
    y_train.to_pickle(f'Data/{INPUT_ADDRESS.split("/")[0]}/Bootstrap_y/y_train_{EMBEDDING}_{i}.pkl')
    y_test.to_pickle(f'Data/{INPUT_ADDRESS.split("/")[0]}/Bootstrap_y/y_test_{EMBEDDING}_{i}.pkl')
# %% ----------------------------------------------------------------
# RUN MODELS ON BOOTSTRAP SAMPLES

#for GROUND_TRUTH in ['wnnL2', 'wnnL1', 'rna']:
#    for CL in [svm_model, rf]:
# Get the classes
f1_scores_per_class = defaultdict(list)
f1_scores_overall = []
pap_scores_per_class = defaultdict(list)

f1_scores_per_class_rna = defaultdict(list)
f1_scores_overall_rna = []
pap_scores_per_class_rna = defaultdict(list)

N = 10
for i in range(0,N):
    print(f"Bootstrap sample {i}/{N-1}")
    boot_time = time.process_time()

    # Loading features
    X_train=pd.read_pickle(f'Data/{INPUT_ADDRESS.split("/")[0]}/Bootstrap_X/X_train_{EMBEDDING}_{i}.pkl')
    X_test=pd.read_pickle(f'Data/{INPUT_ADDRESS.split("/")[0]}/Bootstrap_X/X_test_{EMBEDDING}_{i}.pkl')
    # Load labels
    y_train=pd.read_pickle(f'Data/{INPUT_ADDRESS.split("/")[0]}/Bootstrap_y/y_train_{EMBEDDING}_{i}.pkl')
    y_test=pd.read_pickle(f'Data/{INPUT_ADDRESS.split("/")[0]}/Bootstrap_y/y_test_{EMBEDDING}_{i}.pkl')


    # Pre-process labels and features
    # Converting to pandas series
    if GROUND_TRUTH == 'wnnL2':
        col = '2'
    elif GROUND_TRUTH == 'wnnL1':
        col = '1'
    elif GROUND_TRUTH == 'rna':
        col = '0'
    y_train = y_train[col]
    y_test = y_test[col]
    # CREATING DIFFERENT LABEL SETS
    adding_nans = {'Platelets':np.nan, 'Double negative T cell':np.nan}
    y_train.replace(adding_nans, inplace=True)
    y_test.replace(adding_nans, inplace=True)
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

    # Filtering cells to specified cell type
    FEATURES_COMB_TRAIN, FEATURES_COMB_TEST, LABELS_TRAIN, LABELS_TEST = ut.remove_cells(GROUND_TRUTH, CELL_TYPE, X_train, X_test, y_train, y_test)

    # RNA ONLY FEATURE SET
    FEATURES_RNA_TRAIN = FEATURES_COMB_TRAIN.filter(like='RNA')
    FEATURES_RNA_TEST = FEATURES_COMB_TEST.filter(like='RNA')

    # Get the classes for generating per-class metrics
    classes = np.unique(LABELS_TRAIN)
    print(classes)

    # CLASSIFIER RNA ONLY
    model_cl, y_pred_test, rna_pap_scores_per_class, rna_f1_scores_per_class, rna_f1_scores_overall = boot.model_test_main(CL, FEATURES_RNA_TRAIN,LABELS_TRAIN,
                                            FEATURES_RNA_TEST,LABELS_TEST, classes=classes, f1_scores_per_class = f1_scores_per_class,
                                            f1_scores_overall = f1_scores_overall, pap_scores_per_class = pap_scores_per_class,
                                            subset = False)
    
    # CLASSIFIER RNA + ATAC
    model_cl, y_pred_test, comb_pap_scores_per_class, comb_f1_scores_per_class, comb_f1_scores_overall = boot.model_test_main(CL, FEATURES_COMB_TRAIN,LABELS_TRAIN,
                                            FEATURES_COMB_TEST,LABELS_TEST, classes=classes, f1_scores_per_class=f1_scores_per_class_rna,
                                            f1_scores_overall = f1_scores_overall_rna, pap_scores_per_class = pap_scores_per_class_rna,
                                            subset = False)


# Save pap_scores_per_class and f1_scores_per_class from each model
with open(f'Data/{INPUT_ADDRESS.split("/")[0]}/comb_pap_{SUFFIX}.pkl', 'wb') as f:
    pickle.dump(comb_pap_scores_per_class, f)
with open(f'Data/{INPUT_ADDRESS.split("/")[0]}/comb_f1_{SUFFIX}.pkl', 'wb') as f:
    pickle.dump(comb_f1_scores_per_class, f)
with open(f'Data/{INPUT_ADDRESS.split("/")[0]}/comb_f1_overall_{SUFFIX}.pkl', 'wb') as f:
    pickle.dump(comb_f1_scores_overall, f) 

with open(f'Data/{INPUT_ADDRESS.split("/")[0]}/rna_pap_{SUFFIX}.pkl', 'wb') as f:
    pickle.dump(rna_pap_scores_per_class, f)
with open(f'Data/{INPUT_ADDRESS.split("/")[0]}/rna_f1_{SUFFIX}.pkl', 'wb') as f:
    pickle.dump(rna_f1_scores_per_class, f)
with open(f'Data/{INPUT_ADDRESS.split("/")[0]}/rna_f1_overall_{SUFFIX}.pkl', 'wb') as f:
    pickle.dump(rna_f1_scores_overall, f)

boot_taken=(time.process_time() - boot_time)
print(f'CPU time for boostrap ({SUFFIX}): {boot_taken} seconds or {boot_taken/60} mins or {boot_taken/(60*60)} hrs')

# %% ----------------------------------------------------------------
# LOAD PAP AND F1 SCORES

# Load pap_scores_per_class and f1_scores_per_class
with open(f'Data/{INPUT_ADDRESS.split("/")[0]}/comb_pap_{SUFFIX}.pkl', 'rb') as f:
    comb_pap_scores_per_class = pickle.load(f)
with open(f'Data/{INPUT_ADDRESS.split("/")[0]}/comb_f1_{SUFFIX}.pkl', 'rb') as f:
    comb_f1_scores_per_class = pickle.load(f)
with open(f'Data/{INPUT_ADDRESS.split("/")[0]}/comb_f1_overall_{SUFFIX}.pkl', 'rb') as f:
    comb_f1_scores_overall = pickle.load(f) 

with open(f'Data/{INPUT_ADDRESS.split("/")[0]}/rna_pap_{SUFFIX}.pkl', 'rb') as f:
    rna_pap_scores_per_class = pickle.load(f)
with open(f'Data/{INPUT_ADDRESS.split("/")[0]}/rna_f1_{SUFFIX}.pkl', 'rb') as f:
    rna_f1_scores_per_class = pickle.load(f)
with open(f'Data/{INPUT_ADDRESS.split("/")[0]}/rna_f1_overall_{SUFFIX}.pkl', 'rb') as f:
    rna_f1_scores_overall = pickle.load(f) 

# %% ------------------------------------------------
# PROCESS METRICS

rna_results = boot.analyse_metrics(rna_f1_scores_per_class, rna_pap_scores_per_class, rna_f1_scores_overall)
comb_results = boot.analyse_metrics(comb_f1_scores_per_class, comb_pap_scores_per_class, comb_f1_scores_overall)
# %%
# VISUALISE A DATASET

embedding = ut.visualise_embeddings(X_train, y_train)

# %%
