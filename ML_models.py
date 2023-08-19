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
import seaborn as sns
import Utils as ut
from importlib import reload

# %% ----------------------------------------------------------------
# CHOOSE FEATURE/LABEL SET
#for cells in ['B cells', 'T cells', 'Monoblast-Derived', 'All']:
EMBEDDING = 'PCA MINOR' # Choose from: PCA Minor, CCA, scVI
GROUND_TRUTH = 'wnnL2' # Choose from: wnnL2, wnnL1, rna
CELL_TYPE = 'All' # Choose from: B cells, T cells, Monoblast-Derived, All
N_COMPONENTS = 35   # Choose from: 10, 35
# %% ----------------------------------------------------------------
# LOAD PROCESSED DATA

features_comb_train, features_comb_test, labels_train, labels_test = ut.choose_feature_set(EMBEDDING, GROUND_TRUTH, n_components=N_COMPONENTS, resample = False)

# %% ----------------------------------------------------------------
# FILTER OUT CELL TYPES

FEATURES_COMB_TRAIN, FEATURES_COMB_TEST, LABELS_TRAIN, LABELS_TEST = ut.remove_cells(GROUND_TRUTH, CELL_TYPE, features_comb_train, features_comb_test, labels_train, labels_test)
# %% ----------------------------------------------------------------
#ML Models

xgb=XGBClassifier(random_state=42)
rf=RandomForestClassifier(random_state=42, class_weight='balanced')
svm_model=svm.SVC(random_state=42, class_weight='balanced')
log_reg=LogisticRegression(random_state=42, class_weight='balanced')

# %% ----------------------------------------------------------------
# RNA ONLY FEATURE SET

FEATURES_RNA_TRAIN = FEATURES_COMB_TRAIN.filter(like='RNA')
FEATURES_RNA_TEST = FEATURES_COMB_TEST.filter(like='RNA')

# %% ----------------------------------------------------------------
# SVM RNA ONLY

model_cl, y_pred_test, metrics, f1_df, pap_df = ut.model_test_main(svm_model,FEATURES_RNA_TRAIN,LABELS_TRAIN,
                                        FEATURES_RNA_TEST,LABELS_TEST, 
                                        subset = False)
ut.save_model(model_cl,f'{EMBEDDING}\SVM_RNA_only_{GROUND_TRUTH}_{CELL_TYPE}_{N_COMPONENTS}', y_pred_test, LABELS_TEST, f1_df, pap_df)
# %% ----------------------------------------------------------------
# SVM RNA + ATAC

model_cl, y_pred_test, metrics, f1_df, pap_df = ut.model_test_main(svm_model,FEATURES_COMB_TRAIN,LABELS_TRAIN,
                                        FEATURES_COMB_TEST,LABELS_TEST, 
                                        subset = False)
ut.save_model(model_cl,f'{EMBEDDING}\SVM_{GROUND_TRUTH}_{CELL_TYPE}_{N_COMPONENTS}', y_pred_test, LABELS_TEST, f1_df, pap_df)

# %% ----------------------------------------------------------------
# RANDOM FOREST RNA ONLY

model_cl, y_pred_test, metrics, f1_df, pap_df = ut.model_test_main(rf,FEATURES_RNA_TRAIN,LABELS_TRAIN,
                                           FEATURES_RNA_TEST,LABELS_TEST, 
                                           subset = False)
ut.save_model(model_cl,f'{EMBEDDING}\RF_RNA_only_{GROUND_TRUTH}_{CELL_TYPE}_{N_COMPONENTS}', y_pred_test, LABELS_TEST, f1_df, pap_df)
# %% ----------------------------------------------------------------
# RANDOM FOREST RNA + ATAC

model_cl, y_pred_test, metrics, f1_df, pap_df = ut.model_test_main(rf,FEATURES_COMB_TRAIN,LABELS_TRAIN,
                                           FEATURES_COMB_TEST,LABELS_TEST, 
                                           subset = False)
ut.save_model(model_cl,f'{EMBEDDING}\RF_{GROUND_TRUTH}_{CELL_TYPE}_{N_COMPONENTS}', y_pred_test, LABELS_TEST, f1_df, pap_df)

# %% ----------------------------------------------------------------
# Log Reg RNA ONLY

model_cl, y_pred_test, metrics, f1_df, pap_df = ut.model_test_main(log_reg,FEATURES_RNA_TRAIN,LABELS_TRAIN,
                                           FEATURES_RNA_TEST,LABELS_TEST, 
                                           subset = False)
ut.save_model(model_cl,f'{EMBEDDING}\LOGREG_RNA_only_{GROUND_TRUTH}_{CELL_TYPE}_{N_COMPONENTS}', y_pred_test, LABELS_TEST, f1_df, pap_df)
# %% ----------------------------------------------------------------
# Log Reg RNA + ATAC

model_cl, y_pred_test, metrics, f1_df, pap_df = ut.model_test_main(log_reg,FEATURES_COMB_TRAIN,LABELS_TRAIN,
                                           FEATURES_COMB_TEST,LABELS_TEST, 
                                           subset = False)
ut.save_model(model_cl,f'{EMBEDDING}\LOGREG_{GROUND_TRUTH}_{CELL_TYPE}_{N_COMPONENTS}', y_pred_test, LABELS_TEST, f1_df, pap_df)

# %% ----------------------------------------------------------------
# VISUALIZE EMBEDDINGS

embedding = ut.visualise_embeddings(FEATURES_COMB_TRAIN, LABELS_TRAIN)

# %% ----------------------------------------------------------------
# FEATURE IMPORTANCE 

# Load model
# open a file, where you stored the pickled data
file = open('Supervised Models\PCA\RF_RNA_only.pickle', 'rb')
# dump information to that file
model_cl = pickle.load(file)
# LOAD PROCESSED DATA
mdata_train = mu.read_h5mu("Data/PBMC 10k multiomic/processed_data/Dim_Red/unann_mdata_train.h5mu")
mdata_test = mu.read_h5mu("Data/PBMC 10k multiomic/processed_data/Dim_Red/unann_mdata_test.h5mu")
# %% ----------------------------------------------------------------
shap_values, feat_imp = ut.feature_importance(model_cl, FEATURES_COMB_TEST, mdata_train)
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
# PLOT COMPUTED FEATURE IMPORTANCE FOR EACH GENE

# Convert the dictionary to a DataFrame
df = pd.DataFrame(class_imp)

# For each cell type
for cell_type in df.columns:

    df[cell_type] = (df[cell_type] / df[cell_type].sum())*100 # Normalize gene importances for each cell type
    # Sort genes by importance for the current cell type
    sorted_genes = df[cell_type].sort_values(ascending=False)
    
    # Take the top 20
    top_20_genes = sorted_genes.head(20)
    
    # Plot
    plt.figure(figsize=(10,8))
    top_20_genes.plot(kind='barh')
    plt.title(f'Top 20 genes for {cell_type}')
    plt.xlabel('% Importance')
    plt.ylabel('Gene')
    plt.gca().invert_yaxis()  # To have the gene with the highest importance at the top
    plt.show()

# %% ----------------------------------------------------------------
# REMOVING OUTLIERS

filtered_df = ut.remove_outliers('LOF',FEATURES_COMB_TRAIN, LABELS_TRAIN, FEATURES_COMB_TEST, LABELS_TEST, 95)

# %%

ut.visualise_embeddings(filtered_df.drop(['label'], axis =1), filtered_df['label'])
# %% ----------------------------------------------------------------
# ADDITIONAL PLOTS

sns.scatterplot(x=FEATURES_COMB_TRAIN['ATAC PC1'], y=FEATURES_COMB_TRAIN['ATAC PC2'], hue=LABELS_TRAIN)
plt.show()
# %%
# CORRELATION MATRIX OF PCS
corr_matrix = FEATURES_COMB_TRAIN.corr()
sns.heatmap(corr_matrix)
plt.show()
