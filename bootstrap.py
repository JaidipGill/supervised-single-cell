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
from sklearn.metrics import silhouette_score

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

EMBEDDING = 'PCA' # Choose from: PCA, CCA, scVI
GROUND_TRUTH = 'wnnL2' # PBMC: wnnL2, wnnL1, rna    Cancer: wnnL2, wnnL1, rna, T cells   AD: wnnL2
CELL_TYPE = 'All' # Choose from: All, B cells, T cells, Monoblast-Derived   # Choose from: 10, 35
CL = svm_model # Choose from: xgb, rf, svm_model, log_reg
N_COMPONENTS_TO_TEST = 35 # Choose from: 10, 35
DATA = 'cancer' # Choose from: pbmc, cancer, AD

if DATA == 'pbmc':
    INPUT_ADDRESS = "PBMC 10k multiomic/QC-pbmc10k.h5mu"
    GROUND_TRUTH_SUFFIX = ''
    OUTCOME = 'multi'
elif DATA == 'cancer':
    INPUT_ADDRESS = "B cell lymphoma/lymph_node_lymphoma_14k_filtered_feature_bc_matrix.h5"
    OUTCOME = 'multi'
    if GROUND_TRUTH == 'wnnL1':
        GROUND_TRUTH_SUFFIX = '_wnnL1'
    elif GROUND_TRUTH == 'rna':
        GROUND_TRUTH_SUFFIX = '_rna'
    elif GROUND_TRUTH == 'T cells':
        GROUND_TRUTH_SUFFIX = '_Tcells'
elif DATA == 'AD':
    INPUT_ADDRESS = "Alz multiomic/Downsampled10%_matrix_processed.h5mu"
    GROUND_TRUTH_SUFFIX = '_wnnL2'
    OUTCOME = 'multi'
if EMBEDDING == 'PCA':
    OBSM = 'X_pca'
elif EMBEDDING == 'CCA':
    OBSM = 'cca'
elif EMBEDDING == 'scVI':
    OBSM = 'X_scVI'
N_COMPONENTS = 35

# %% ----------------------------------------------------------------
# BOOTSTRAP SAMPLES

N = 5 # Number of bootstrap samples

for i in range(0,N):
    
    # Loading QC data
    if DATA == 'AD':
        mdata = mu.read_h5mu(f'Data/{INPUT_ADDRESS}')
        mdata.var_names_make_unique()
    elif DATA == 'cancer':
        mdata = mu.read_10x_h5(f'Data/{INPUT_ADDRESS}')
        mdata.var_names_make_unique()
        #Load annotations
        mdata= boot.add_annon(mdata, subset=False, data=DATA, GROUND_TRUTH = GROUND_TRUTH)
    else:
        mdata = mu.read_h5mu(f'Data/{INPUT_ADDRESS}')
        #Load annotations
        mdata= boot.add_annon(mdata, subset=False, data=DATA, GROUND_TRUTH = GROUND_TRUTH)

    # Bootstrap sample and pre-process without data leakage
    mdata_train, mdata_test = boot.train_test_split_mdata(mdata, seed=i)
    mdata_train.mod['rna'] = ut.pre_process_train(mdata_train['rna'], DATA)
    mdata_test.mod['rna'] = ut.pre_process_test(mdata_test['rna'], mdata_train['rna'])
    mdata_train.mod['atac'] = ut.pre_process_train(mdata_train['atac'], DATA)
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
    elif (DATA == 'cancer')|(DATA == 'AD'):
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
    
    # Convert back to dataframe format with column names
    X_train = pd.DataFrame(X_train, columns=[f"RNA PC{i}" for i in range(1,N_COMPONENTS+1)] + [f"ATAC PC{i}" for i in range(1,N_COMPONENTS+1)])
    X_test = pd.DataFrame(X_test, columns=[f"RNA PC{i}" for i in range(1,N_COMPONENTS+1)] + [f"ATAC PC{i}" for i in range(1,N_COMPONENTS+1)])

    print(X_test.head())
    print(X_train)
    print(X_test)
    print(y_train)
    print(y_test)

    # Save the feature and label datasets
    X_train.to_pickle(f'Data/{INPUT_ADDRESS.split("/")[0]}/Bootstrap_X/X_train_{EMBEDDING}_{i}{GROUND_TRUTH_SUFFIX}.pkl')
    X_test.to_pickle(f'Data/{INPUT_ADDRESS.split("/")[0]}/Bootstrap_X/X_test_{EMBEDDING}_{i}{GROUND_TRUTH_SUFFIX}.pkl')
    # Save label dataframe
    y_train.to_pickle(f'Data/{INPUT_ADDRESS.split("/")[0]}/Bootstrap_y/y_train_{EMBEDDING}_{i}{GROUND_TRUTH_SUFFIX}.pkl')
    y_test.to_pickle(f'Data/{INPUT_ADDRESS.split("/")[0]}/Bootstrap_y/y_test_{EMBEDDING}_{i}{GROUND_TRUTH_SUFFIX}.pkl')
    
# %% ----------------------------------------------------------------
# RUN MODELS ON BOOTSTRAP SAMPLES

for GROUND_TRUTH in ['T cells']: # ['wnnL2', 'wnnL1', 'rna', 'T cells']
    for EMBEDDING in ['PCA','scVI']: # ['PCA', 'scVI']
        for CL in [svm_model]: # [rf, svm_model, log_reg]
            # Get the classes
            SUFFIX = f'{DATA}_{CL.__class__.__name__}_{EMBEDDING}_{GROUND_TRUTH}_{CELL_TYPE}_{N_COMPONENTS}'
            f1_scores_per_class = defaultdict(list)
            f1_scores_overall = []
            pap_scores_per_class = defaultdict(list)

            f1_scores_per_class_rna = defaultdict(list)
            f1_scores_overall_rna = []
            pap_scores_per_class_rna = defaultdict(list)

            precision_scores_per_class = defaultdict(list)
            precision_scores_overall = []
            recall_scores_per_class = defaultdict(list)
            recall_scores_overall = []

            precision_scores_per_class_rna = defaultdict(list)
            precision_scores_overall_rna = []
            recall_scores_per_class_rna = defaultdict(list)
            recall_scores_overall_rna = []

            boot_time = time.process_time()

            N = 5
            for i in range(0,N):
                print(f"Bootstrap sample {i}/{N-1}")

                # Loading features
                X_train=pd.read_pickle(f'Data/{INPUT_ADDRESS.split("/")[0]}/Bootstrap_X/X_train_{EMBEDDING}_{i}{GROUND_TRUTH_SUFFIX}.pkl')
                X_test=pd.read_pickle(f'Data/{INPUT_ADDRESS.split("/")[0]}/Bootstrap_X/X_test_{EMBEDDING}_{i}{GROUND_TRUTH_SUFFIX}.pkl')
                # Load labels
                y_train=pd.read_pickle(f'Data/{INPUT_ADDRESS.split("/")[0]}/Bootstrap_y/y_train_{EMBEDDING}_{i}{GROUND_TRUTH_SUFFIX}.pkl')
                y_test=pd.read_pickle(f'Data/{INPUT_ADDRESS.split("/")[0]}/Bootstrap_y/y_test_{EMBEDDING}_{i}{GROUND_TRUTH_SUFFIX}.pkl')
                print(y_train)

                if N_COMPONENTS_TO_TEST == 10:
                    X_train.iloc[:, list(range(0, 10)) + list(range(35, 45))]
                    X_test.iloc[:, list(range(0, 10)) + list(range(35, 45))]

                # Pre-process labels and features
                # Converting to pandas series
                if DATA == 'pbmc':
                    if GROUND_TRUTH == 'wnnL2':
                        col = '2'
                    elif GROUND_TRUTH == 'wnnL1':
                        col = '1'
                    elif GROUND_TRUTH == 'rna':
                        col = '0'
                elif DATA == 'cancer':
                    col = '0'
                elif DATA == 'AD':
                    col = '0'
                y_train = y_train[col]
                y_test = y_test[col]
                # Create DataFrame from X_train and X_test
                train_data = pd.DataFrame(X_train)
                test_data = pd.DataFrame(X_test)
                # Add y_train and y_test as columns
                train_data['label'] = y_train
                test_data['label'] = y_test
                # Separate X_train, y_train, X_test, and y_test from the updated DataFrame
                X_train = train_data.iloc[:, :-1]
                y_train = train_data['label'].to_numpy()
                X_test = test_data.iloc[:, :-1]
                y_test = test_data['label'].to_numpy()

                # Filtering cells to specified cell type
                FEATURES_COMB_TRAIN, FEATURES_COMB_TEST, LABELS_TRAIN, LABELS_TEST = ut.remove_cells(DATA, GROUND_TRUTH, CELL_TYPE, X_train, X_test, y_train, y_test)

                # RNA ONLY FEATURE SET
                FEATURES_RNA_TRAIN = FEATURES_COMB_TRAIN.filter(like='RNA')
                FEATURES_RNA_TEST = FEATURES_COMB_TEST.filter(like='RNA')

                # Get the classes for generating per-class metrics
                classes = np.unique(LABELS_TRAIN)
                print(classes)

                # CLASSIFIER RNA ONLY
                
                print("RNA only")
                model_cl, y_pred_test, rna_pap_scores_per_class, rna_f1_scores_per_class, rna_f1_scores_overall, rna_precision_scores_per_class, rna_precision_scores_overall, rna_recall_scores_per_class, rna_recall_scores_overall = boot.model_test_main(CL, OUTCOME, FEATURES_RNA_TRAIN,LABELS_TRAIN,
                                                        FEATURES_RNA_TEST,LABELS_TEST, classes=classes, f1_scores_per_class = f1_scores_per_class_rna,
                                                        f1_scores_overall = f1_scores_overall_rna, pap_scores_per_class = pap_scores_per_class_rna,
                                                        precision_scores_per_class = precision_scores_per_class_rna, precision_scores_overall = precision_scores_overall_rna, 
                                                        recall_scores_per_class = recall_scores_per_class_rna, recall_scores_overall = recall_scores_overall_rna,
                                                        subset = False, detailed = False)
                ut.save_model(model_cl, f'{SUFFIX}_rna_{i}')
                
                # CLASSIFIER RNA + ATAC
                print("RNA + ATAC")
                model_cl, y_pred_test, comb_pap_scores_per_class, comb_f1_scores_per_class, comb_f1_scores_overall, comb_precision_scores_per_class, comb_precision_scores_overall, comb_recall_scores_per_class, comb_recall_scores_overall = boot.model_test_main(CL, OUTCOME, FEATURES_COMB_TRAIN,LABELS_TRAIN,
                                                        FEATURES_COMB_TEST,LABELS_TEST, classes=classes, f1_scores_per_class=f1_scores_per_class,
                                                        f1_scores_overall = f1_scores_overall, pap_scores_per_class = pap_scores_per_class,
                                                        precision_scores_per_class = precision_scores_per_class, precision_scores_overall = precision_scores_overall, 
                                                        recall_scores_per_class = recall_scores_per_class, recall_scores_overall = recall_scores_overall,
                                                        subset = False, detailed = False)
                ut.save_model(model_cl, f'{SUFFIX}_comb_{i}')
            
                # Save pap_scores_per_class and f1_scores_per_class from each model
                scores_vars = [comb_pap_scores_per_class, comb_f1_scores_per_class, comb_f1_scores_overall, rna_pap_scores_per_class, rna_f1_scores_per_class, rna_f1_scores_overall,
                            rna_precision_scores_per_class, rna_precision_scores_overall, rna_recall_scores_per_class, rna_recall_scores_overall,
                                comb_precision_scores_per_class, comb_precision_scores_overall, comb_recall_scores_per_class, comb_recall_scores_overall]
                prefixes = ['comb_pap', 'comb_f1', 'comb_f1_overall', 'rna_pap', 'rna_f1', 'rna_f1_overall',
                            'rna_precision', 'rna_precision_overall', 'rna_recall', 'rna_recall_overall',
                            'comb_precision', 'comb_precision_overall', 'comb_recall', 'comb_recall_overall']
                for scores_vars, prefixes in zip(scores_vars, prefixes):
                    with open(f'Data/{INPUT_ADDRESS.split("/")[0]}/{prefixes}_{SUFFIX}.pkl', 'wb') as f:
                        pickle.dump(scores_vars, f)
            
            boot_taken=(time.process_time() - boot_time)
            print(f'CPU time for boostrap ({SUFFIX}): {boot_taken} seconds or {boot_taken/60} mins or {boot_taken/(60*60)} hrs')


# %%

# %%
