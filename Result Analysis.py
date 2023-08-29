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
# LOAD PAP AND F1 SCORES
prefixes = ['comb_pap', 'comb_f1', 'comb_f1_overall', 'rna_pap', 'rna_f1', 'rna_f1_overall',
                'rna_precision', 'rna_precision_overall', 'rna_recall', 'rna_recall_overall',
                'comb_precision', 'comb_precision_overall', 'comb_recall', 'comb_recall_overall']
for GROUND_TRUTH in ['wnnL1', 'rna']: # ['wnnL2', 'wnnL1', 'rna']
    for EMBEDDING in ['scVI']: # ['PCA', 'scVI']
        for CL in [svm_model]: # [rf, svm_model, log_reg]
            results ={}
            # Get the classes
            SUFFIX = f'{DATA}_{CL.__class__.__name__}_{EMBEDDING}_{GROUND_TRUTH}_{CELL_TYPE}_{N_COMPONENTS}'
            for prefix in prefixes:
                with open(f'Data/{INPUT_ADDRESS.split("/")[0]}/{prefix}_{SUFFIX}.pkl', 'rb') as f:
                    results[prefix] = pickle.load(f)

            # PROCESS METRICS
            rna_results = boot.analyse_metrics(results, SUFFIX, rna=True, save = True)
            comb_results = boot.analyse_metrics(results, SUFFIX, rna=False, save = True)
            #print(f'{EMBEDDING} {CL.__class__.__name__} RNA: Precision: {rna_results.iloc[12]["mean Precision score"].round(2)} ({rna_results.iloc[12]["lower Precision CI"].round(2)} - {rna_results.iloc[12]["upper Precision CI"].round(2)}), Recall: {rna_results.iloc[12]["mean Recall score"].round(2)} ({rna_results.iloc[12]["lower Recall CI"].round(2)} - {rna_results.iloc[12]["upper Recall CI"].round(2)}), F1: {rna_results.iloc[12]["mean F1 score"].round(2)} ({rna_results.iloc[12]["lower F1 CI"].round(2)} - {rna_results.iloc[12]["upper F1 CI"].round(2)})')
            #print(f'{EMBEDDING} {CL.__class__.__name__} COMB: Precision: {comb_results.iloc[12]["mean Precision score"].round(2)} ({comb_results.iloc[12]["lower Precision CI"].round(2)} - {comb_results.iloc[12]["upper Precision CI"].round(2)}), Recall: {comb_results.iloc[12]["mean Recall score"].round(2)} ({comb_results.iloc[12]["lower Recall CI"].round(2)} - {comb_results.iloc[12]["upper Recall CI"].round(2)}), F1: {comb_results.iloc[12]["mean F1 score"].round(2)} ({comb_results.iloc[12]["lower F1 CI"].round(2)} - {comb_results.iloc[12]["upper F1 CI"].round(2)})')
# %% ----------------------------------------------------------------
# EMBEDDING COMPARISON

N = 10
rna_train_sil = {}
rna_test_sil = {}
comb_train_sil = {}
comb_test_sil = {}

for EMBEDDING in ['PCA','scVI']: # Iterate through embeddings

    rna_train_sil[EMBEDDING] = []
    rna_test_sil[EMBEDDING] = []
    comb_train_sil[EMBEDDING] = []
    comb_test_sil[EMBEDDING] = []

    for i in range(0,N):

        # Loading bootstrap sample
        FEATURES_RNA_TRAIN, FEATURES_RNA_TEST, FEATURES_COMB_TRAIN, FEATURES_COMB_TEST, LABELS_TRAIN, LABELS_TEST = boot.load_boot(i, N, INPUT_ADDRESS, EMBEDDING, GROUND_TRUTH, CELL_TYPE, DATA, N_COMPONENTS_TO_TEST, GROUND_TRUTH_SUFFIX)
        
        # Silhouette score for RNA feature set on training data
        silhouette_rna_train = silhouette_score(FEATURES_RNA_TRAIN, LABELS_TRAIN)

        # Silhouette score for RNA feature set on test data
        silhouette_rna_test = silhouette_score(FEATURES_RNA_TEST, LABELS_TEST)

        # Silhouette score for COMB feature set on training data
        silhouette_comb_train = silhouette_score(FEATURES_COMB_TRAIN, LABELS_TRAIN)

        # Silhouette score for COMB feature set on test data
        silhouette_comb_test = silhouette_score(FEATURES_COMB_TEST, LABELS_TEST)

        print("Silhouette Scores:")
        print(f"RNA Train: {silhouette_rna_train:.4f}")
        print(f"RNA Test: {silhouette_rna_test:.4f}")
        print(f"COMB Train: {silhouette_comb_train:.4f}")
        print(f"COMB Test: {silhouette_comb_test:.4f}")

        # Append bootstrap iteration silhouette score to lists
        rna_train_sil[EMBEDDING].append(silhouette_rna_train)
        rna_test_sil[EMBEDDING].append(silhouette_rna_test)
        comb_train_sil[EMBEDDING].append(silhouette_comb_train)
        comb_test_sil[EMBEDDING].append(silhouette_comb_test)

# %% ----------------------------------------------------------------
# VISUALISE A DATASET
# load data
x = pd.read_csv('X_y.csv')
y = x['2']
for i in range(0,3):
    x = x.drop(f'{i}', axis=1)

# %%
# Loading bootstrap sample
for EMBEDDING in ['PCA', 'scVI']: # Iterate through embeddings
    FEATURES_RNA_TRAIN, FEATURES_RNA_TEST, FEATURES_COMB_TRAIN, FEATURES_COMB_TEST, LABELS_TRAIN, LABELS_TEST = boot.load_boot(0, 0, INPUT_ADDRESS, EMBEDDING, GROUND_TRUTH, CELL_TYPE, DATA, N_COMPONENTS_TO_TEST, GROUND_TRUTH_SUFFIX)

    #embedding = ut.visualise_embeddings(FEATURES_RNA_TRAIN, LABELS_TRAIN)
    #embedding = ut.visualise_embeddings(FEATURES_COMB_TRAIN, LABELS_TRAIN)
    embedding = ut.visualise_embeddings(FEATURES_RNA_TEST, LABELS_TEST)
    embedding = ut.visualise_embeddings(FEATURES_COMB_TEST, LABELS_TEST)

# %% ----------------------------------------------------------------
# FIND MEAN AND STD OF NUMBER OF EACH CELL TYPE PER BOOTSTRAP FOR TRAIN AND TEST

# This will store mean and standard deviation for each column
results = {}

N = 10

# loop through train and test sets
for df_text in ['train', 'test']:
    # Loop over each column
    for col in ['0','1','2']:
        # List to store counts from each bootstrap for current column
        counts_list = []

        for i in range(0,N):
            print(f"Bootstrap sample {i}/{N-1}")

            # Load labels
            df=pd.read_pickle(f'Data/{INPUT_ADDRESS.split("/")[0]}/Bootstrap_y/y_{df_text}_{EMBEDDING}_{i}{GROUND_TRUTH_SUFFIX}.pkl')

            # Count occurrences for the current column in the current bootstrap sample
            counts = df[col].value_counts()
            counts_list.append(counts)
            print(counts)

        # Convert the list of counts into a DataFrame
        counts_df = pd.concat(counts_list, axis=1).fillna(0)

        # Calculate mean and standard deviation for each cell type
        mean = counts_df.mean(axis=1)
        std = counts_df.std(axis=1)
        
        results[col] = pd.DataFrame({'Mean': mean, 'Std': std})

    # Results now has a DataFrame for each column with mean and std for each cell type
    for col, df in results.items():
        df_sorted = df.sort_index()  # Sort rows based on the index (cell type)
        print(f"Column: {col}")
        print(df_sorted)
        print()

# %% ----------------------------------------------------------------
# SHAP ANALYSIS
# Load model
model_cl = pickle.load(open('Supervised Models\Saved Models\Model_pbmc_SVC_scVI_wnnL2_All_35_comb_0.pickle', 'rb'))
X_train = pd.read_pickle('Data\PBMC 10k multiomic\Bootstrap_X\X_train_scVI_0.pkl')
X_test = pd.read_pickle('Data\PBMC 10k multiomic\Bootstrap_X\X_test_scVI_0.pkl')
rna_sums, atac_sums, shap_vals = ut.feature_importance(model_cl, X_train = X_train[:100], X_test=X_test[:500])

# %% ----------------------------------------------------------------
# EXTRACT FEATURE IMPORTANCE FOR EACH CELL TYPE

class_imp = {}
for cell_idx, cell in enumerate(model_cl.classes_): # for each cell type
    class_imp[cell] = {}
    for gene_idx, gene in enumerate(mdata_train.mod['rna'].var_names): # look at each gene
        class_imp[cell][gene] = feat_imp[gene_idx][cell_idx] # extract the gene importance for that cell type
# %% ----------------------------------------------------------------
# PLOT LOADING COEFFICIENTS FOR EACH PC

ut.plot_loading_coefficients(shap_values, mdata_train)

# %% ----------------------------------------------------------------
# PEAK-GENE IMPUTATION
import pyranges as pr

# Loading QC data
mdata=mu.read_h5mu(f'Data/PBMC 10k multiomic/QC-pbmc10k.h5mu')
atac_peaks = mdata.mod['atac'].var['gene_ids'].values
genes = mdata.get_node_attrs('gene')
peak_to_gene = {}
for i, peak in enumerate(atac_peaks):
    closest_gene = pr.nearest(peak, genes, mode='extend')[0]
    peak_to_gene[peak] = closest_gene
mdata.obs['peak_to_gene'] = peak_to_gene
print(mdata.obs['peak_to_gene'][1000])

# %% ----------------------------------------------------------------
# GENERATE CSV FOR R LOG REG

# Load the pickled pandas DataFrame
X = pd.read_pickle('Data\PBMC 10k multiomic\Bootstrap_X\X_train_scVI_0.pkl')
# Load the pickled numpy array
y = np.load('Data\PBMC 10k multiomic\Bootstrap_y\y_train_scVI_0.pkl', allow_pickle=True)  # Assuming the array was saved using numpy's save function
# Convert the numpy array to a pandas DataFrame
y = pd.DataFrame(y)
# If you want to join/concatenate the numpy array DataFrame to the original DataFrame
df_combined = pd.concat([X, y], axis=1)  # Assuming you want to concatenate columns-wise
# Save the combined DataFrame to a CSV file
df_combined.to_csv('Data\PBMC 10k multiomic\Interpretation\X_y.csv', index=False)

# %%
