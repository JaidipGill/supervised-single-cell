# %% ----------------------------------------------------------------
import scanpy as sc
import muon as mu
import anndata as ad
import h5py as h5
import pandas as pd
from muon import atac as ac
import numpy as np
import scvi
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LearningCurveDisplay, train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.utils import resample
from xgboost import XGBClassifier
import time
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, make_scorer
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import pickle
import umap
import seaborn as sns
import shap
from imblearn.over_sampling import SMOTE
import plotly.express as px
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.metrics import classification_report
from collections import defaultdict
# %% ----------------------------------------------------------------
# FUNCTIONS

def train_test_split_mdata(mdata_obj):

    '''
    Split MuData object into train and test sets
    '''

    # Get the indices of cells
    indices = np.arange(mdata_obj['rna'].shape[0])

    # Split indices into train and test
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    # Create train and test MuData objects
    mdata_train = mu.MuData({mod: mdata_obj[mod][train_idx] for mod in mdata_obj.mod.keys()}) 
    mdata_test = mu.MuData({mod: mdata_obj[mod][test_idx] for mod in mdata_obj.mod.keys()})

    # Convert views to AnnData objects
    mdata_train = mdata_train.copy()
    mdata_test = mdata_test.copy()

    return mdata_train, mdata_test

def pre_process_train(adata):

    '''
    Pre-process MuData TRAIN object for either RNA or ATAC modality
    '''

    # Filter out low-frequency features
    mu.pp.filter_var(adata, 'n_cells_by_counts', lambda x: x >= 10) 

    # Saving raw counts
    adata.layers["counts"] = adata.X.copy() # Store unscaled counts in .layers["counts"] attribute

    # Normalizing peaks/genes
    print(f"Total peaks/genes in random cell before normalization: {adata.X[1,:].sum()}")
    sc.pp.normalize_total(adata, target_sum=1e4) # Normalize counts per cell
    print(f"Total peaks/genes in random cell (Should be 10000): {adata.X[1,:].sum()}") # Sanity check for normalization - should be 1e4
    sc.pp.log1p(adata) # Logarithmize + 1
    print(f"Total peaks/genes in random cell: {adata.X[1,:].sum()}") 
    print(f"Total peaks/genes in layers['counts']: {adata.layers['counts'][1,:].sum()}") # Sanity check for non-normalized - should be same as before normalization

    # Filtering features
    sc.pp.highly_variable_genes(adata) # Select highly variable genes
    sc.pl.highly_variable_genes(adata) # Plot highly variable genes
    mu.pp.filter_var(adata, 'highly_variable', lambda x: x == True) # Filter out non-highly variable genes
    print(f"Number of highly variable peaks/genes: {np.sum(adata.var.highly_variable)}")
    print(f"Number of peaks/genes after filtering: {adata.shape[1]}")

    # Scaling
    adata.raw = adata # Store unscaled counts in .raw attribute
    sc.pp.scale(adata) # Scale to unit variance and shift to zero mean
    print(f"Min Scaled peaks/genes value:{adata.X.min()}") 
    print(f"Min unscaled peaks/genes value - Should not be negative: {adata.raw.X.min()}") # Sanity check for scaling - should not be negative

    return adata

def pre_process_test(adata, adata_train):

    '''
    Pre-process MuData TEST object for either RNA or ATAC modality using 
    parameters from pre-processing of TRAIN object
    '''

    # Filtering features
    mu.pp.filter_var(adata, adata_train.var.index) # Filter out non-highly variable genes
    print(f"Number of peaks/genes after filtering: {adata.shape[1]}")

    # Saving raw counts
    adata.layers["counts"] = adata.X.copy() # Store unscaled counts in .layers["counts"] attribute

    # Normalizing peaks/genes
    print(f"Total peaks/genes in random cell before normalization: {adata.X[1,:].sum()}")
    sc.pp.normalize_total(adata, target_sum=1e4) # Normalize counts per cell
    print(f"Total peaks/genes in random cell (Should be 10000): {adata.X[1,:].sum()}") # Sanity check for normalization - should be 1e4
    sc.pp.log1p(adata) # Logarithmize + 1
    print(f"Total peaks/genes in random cell: {adata.X[1,:].sum()}") 
    print(f"Total peaks/genes in layers['counts']: {adata.layers['counts'][1,:].sum()}") # Sanity check for non-normalized - should be same as before normalization

    # Scaling
    adata.raw = adata # Store unscaled counts in .raw attribute
    adata.X = adata.X.toarray()
    adata.X = (adata.X - adata_train.var['mean'].values[np.newaxis,:])/adata_train.var['std'].values[np.newaxis,:]
    print(f"Min Scaled peaks/genes value:{adata.X.min()}") 
    print(f"Min unscaled peaks/genes value: {adata.raw.X.min()}") # Sanity check for scaling - should not be negative

    return adata

def wnn_cluster(mdata):
    '''
    Generat ground truth labels for MuData object using Weighted Nearest Neighbours
    (Seurat V4) which incporates both modalities in clustering
    '''
    # Pre-process entire dataset
    mdata.mod['rna'] = pre_process_train(mdata['rna'])
    mdata.mod['atac'] = pre_process_train(mdata['atac'])
    
    # PCA
    sc.tl.pca(mdata.mod['rna'])
    sc.tl.pca(mdata.mod['atac'])
    
    # Calculate weighted nearest neighbors
    sc.pp.neighbors(mdata['rna'])
    sc.pp.neighbors(mdata['atac'])
    mu.pp.neighbors(mdata, key_added='wnn', add_weights_to_modalities = True)

    # PLot WNN UMAP
    mdata.uns['wnn']['params']['use_rep']
    mu.tl.umap(mdata, neighbors_key='wnn', random_state=10)
    mu.pl.umap(mdata, color=['rna:mod_weight', 'atac:mod_weight'], cmap='RdBu')
    
    # Clustering WNN
    sc.tl.leiden(mdata, resolution=1.0, neighbors_key='wnn', key_added='leiden_wnn')
    sc.pl.umap(mdata, color='leiden_wnn', legend_loc='on data')
    sc.pl.violin(mdata, groupby='leiden_wnn', keys='atac:mod_weight')

def annotate_clusters(mdata, level):    
    '''
    Annotate clusters based on marker genes
    level = 1: Majory cell subtypes
    level = 2: Minor cell subtypes
    level = 3: Not complete due to lack of meaning in cell subtypes
    '''
    marker_genes_l1 = {
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
    }
    marker_genes_l2 = {
        'B intermediate': ['MS4A1', 'TNFRSF13B', 'IGHM', 'IGHD', 'AIM2', 'CD79A', 'LINC01857', 'RALGPS2', 'BANK1', 'CD79B'],
        'B memory': ['MS4A1', 'COCH', 'AIM2', 'BANK1', 'SSPN', 'CD79A', 'TEX9', 'RALGPS2', 'TNFRSF13C', 'LINC01781'],
        'B naive': ['IGHM', 'IGHD', 'CD79A', 'IL4R', 'MS4A1', 'CXCR4', 'BTG1', 'TCL1A', 'CD79B', 'YBX3'],
        'Plasmablast': ['IGHA2', 'MZB1', 'TNFRSF17', 'DERL3', 'TXNDC5', 'TNFRSF13B', 'POU2AF1', 'CPNE5', 'HRASLS2', 'NT5DC2'],
        'CD4 CTL': ['GZMH', 'CD4', 'FGFBP2', 'ITGB1', 'GZMA', 'CST7', 'GNLY', 'B2M', 'IL32', 'NKG7'],
        'CD4 Naive': ['TCF7', 'CD4', 'CCR7', 'IL7R', 'FHIT', 'LEF1', 'MAL', 'NOSIP', 'LDHB', 'PIK3IP1'],
        'CD4 Proliferating': ['MKI67', 'TOP2A', 'PCLAF', 'CENPF', 'TYMS', 'NUSAP1', 'ASPM', 'PTTG1', 'TPX2', 'RRM2'],
        'CD4 TCM': ['IL7R', 'TMSB10', 'CD4', 'ITGB1', 'LTB', 'TRAC', 'AQP3', 'LDHB', 'IL32', 'MAL'],
        'CD4 TEM': ['IL7R', 'CCL5', 'FYB1', 'GZMK', 'IL32', 'GZMA', 'KLRB1', 'TRAC', 'LTB', 'AQP3'],
        'Treg': ['RTKN2', 'FOXP3', 'AC133644.2', 'CD4', 'IL2RA', 'TIGIT', 'CTLA4', 'FCRL3', 'LAIR2', 'IKZF2'],
        'CD8 Naive': ['CD8B', 'S100B', 'CCR7', 'RGS10', 'NOSIP', 'LINC02446', 'LEF1', 'CRTAM', 'CD8A', 'OXNAD1'],
        'CD8 Proliferating': ['MKI67', 'CD8B', 'TYMS', 'TRAC', 'PCLAF', 'CD3D', 'CLSPN', 'CD3G', 'TK1', 'RRM2'],
        'CD8 TCM': ['CD8B', 'ANXA1', 'CD8A', 'KRT1', 'LINC02446', 'YBX3', 'IL7R', 'TRAC', 'NELL2', 'LDHB'],
        'CD8 TEM': ['CCL5', 'GZMH', 'CD8A', 'TRAC', 'KLRD1', 'NKG7', 'GZMK', 'CST7', 'CD8B', 'TRGC2'],
        'ASDC': ['PPP1R14A', 'LILRA4', 'AXL', 'IL3RA', 'SCT', 'SCN9A', 'LGMN', 'DNASE1L3', 'CLEC4C', 'GAS6'],
        'cDC1': ['CLEC9A', 'DNASE1L3', 'C1orf54', 'IDO1', 'CLNK', 'CADM1', 'FLT3', 'ENPP1', 'XCR1', 'NDRG2'],
        'cDC2': ['FCER1A', 'HLA-DQA1', 'CLEC10A', 'CD1C', 'ENHO', 'PLD4', 'GSN', 'SLC38A1', 'NDRG2', 'AFF3'],
        'pDC': ['ITM2C', 'PLD4', 'SERPINF1', 'LILRA4', 'IL3RA', 'TPM2', 'MZB1', 'SPIB', 'IRF4', 'SMPD3'],
        'CD14 Mono': ['S100A9', 'CTSS', 'S100A8', 'LYZ', 'VCAN', 'S100A12', 'IL1B', 'CD14', 'G0S2', 'FCN1'],
        'CD16 Mono': ['CDKN1C', 'FCGR3A', 'PTPRC', 'LST1', 'IER5', 'MS4A7', 'RHOC', 'IFITM3', 'AIF1', 'HES4'],
        'NK': ['GNLY', 'TYROBP', 'NKG7', 'FCER1G', 'GZMB', 'TRDC', 'PRF1', 'FGFBP2', 'SPON2', 'KLRF1'],
        'NK Proliferating': ['MKI67', 'KLRF1', 'TYMS', 'TRDC', 'TOP2A', 'FCER1G', 'PCLAF', 'CD247', 'CLSPN', 'ASPM'],
        'NK_CD56bright': ['XCL2', 'FCER1G', 'SPINK2', 'TRDC', 'KLRC1', 'XCL1', 'SPTSSB', 'PPP1R9A', 'NCAM1', 'TNFRSF11A'],
        'Eryth': ['HBD', 'HBM', 'AHSP', 'ALAS2', 'CA1', 'SLC4A1', 'IFIT1B', 'TRIM58', 'SELENBP1', 'TMCC2'],
        'HSPC':['SPINK2', 'PRSS57', 'CYTL1', 'EGFL7', 'GATA2', 'CD34', 'SMIM24', 'AVP', 'MYB', 'LAPTM4B'],
        'ILC': ['KIT', 'TRDC', 'TTLL10', 'LINC01229', 'SOX4', 'KLRB1', 'TNFRSF18', 'TNFRSF4', 'IL1R1', 'HPGDS'],
        'Platelet': ['PPBP', 'PF4', 'NRGN', 'GNG11', 'CAVIN2', 'TUBB1', 'CLU', 'HIST1H2AC', 'RGS18', 'GP9'],
        'dnT': ['PTPN3', 'MIR4422HG', 'NUCB2', 'CAV1', 'DTHD1', 'GZMA', 'MYB', 'FXYD2', 'GZMK', 'AC004585.1'],
        'gdT': ['TRDC', 'TRGC1', 'TRGC2', 'KLRC1', 'NKG7', 'TRDV2', 'CD7', 'TRGV9', 'KLRD1', 'KLRG1'],
        'MAIT': ['KLRB1', 'NKG7', 'GZMK', 'IL7R', 'SLC4A10', 'GZMA', 'CXCR6', 'PRSS35', 'RBM24', 'NCR3']
        }
    marker_genes_l3 = {
        'ASDC_mDC': ['AXL', 'LILRA4', 'SCN9A', 'CLEC4C', 'LTK', 'PPP1R14A', 'LGMN', 'SCT', 'IL3RA', 'GAS6'],
        'ASDC_pDC' : ['LILRA4', 'CLEC4C', 'SCT', 'EPHB1', 'AXL', 'PROC', 'LRRC26', 'SCN9A', 'LTK', 'DNASE1L3'],
        'B intermediate lambda' : ['MS4A1', 'IGLC2', 'IGHM', 'CD79A', 'IGLC3', 'IGHD', 'BANK1', 'TNFRSF13C', 'CD22', 'TNFRSF13B'],
        'B memory kappa' : ['BANK1', 'IGKC', 'LINC01781', 'MS4A1', 'SSPN', 'CD79A', 'RALGPS2', 'TNFRSF13C', 'LINC00926'],
        'B memory lambda' : ['BANK1', 'IGLC2', 'MS4A1', 'IGLC3', 'COCH', 'TNFRSF13C', 'IGHA2', 'BLK', 'TNFRSF13B', 'LINC01781'],
        'B naive kappa' : ['IGHM', 'TCL1A', 'IGHD', 'IGHG3', 'CD79A', 'IL4R', 'CD37', 'MS4A1', 'IGKC'],
        'CD14 Mono' : ['S100A9', 'CTSS', 'LYZ', 'CTSD', 'S100A8', 'VCAN', 'CD14', 'FCN1', 'S100A12', 'MS4A6A'],
        'CD16 Mono' : ['LST1', 'YBX1', 'AIF1', 'FCGR3A', 'NAP1L1', 'MS4A7', 'FCER1G', 'TCF7L2', 'COTL1', 'CDKN1C'],
        'CD4 CTL' : ['GZMH', 'CD4', 'GNLY', 'FGFBP2', 'IL7R', 'S100A4', 'GZMA', 'CST7', 'IL32', 'CCL5'],
        'CD4 Naive' : ['TCF7', 'CD4', 'NUCB2', 'LDHB', 'TRAT1', 'SARAF', 'FHIT', 'LEF1', 'CCR7', 'IL7R'],
        'CD4 Proliferating' : ['MKI67', 'TYMS', 'PCLAF', 'TOP2A', 'CENPF', 'NUSAP1', 'CENPM', 'BIRC5', 'ZWINT', 'TPX2'],
        'CD4 TCM_1' : ['LTB', 'CD4', 'FYB1', 'IL7R', 'LIMS1', 'MAL', 'TMSB4X', 'TSHZ2', 'AP3M2', 'TRAC'],
        'CD4 TCM_2' : [],
        '' : [],
        '' : [],
        '' : [],
        '' : []
        }
    
    if level == 1:
        MARKER_GENES = marker_genes_l1
        CLUSTER_7 = 'CD4+ Central Memory T'
        FILE = 'L1'
    elif level == 2:
        MARKER_GENES = marker_genes_l2
        CLUSTER_7 = 'CD4 TCM'
        FILE = 'L2'
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

    # Marker genes from Azimuth: https://azimuth.hubmapconsortium.org/references/#Human%20-%20PBMC
    markers_df = sc.tl.marker_gene_overlap(mdata.mod['rna'], MARKER_GENES,method='jaccard')
    print(markers_df)
    max_index_dict = markers_df.idxmax().to_dict() # Get the column name of the max value per row
    print(f'Cluster identities: {max_index_dict}')

    # Dictionary to store columns with max value occurring more than once
    multi_max_cols = {}
    # Iterate over columns
    for col in markers_df.columns:
        # Get the max value in the column
        max_val = markers_df[col].max()
        # Find rows with max value
        max_rows = markers_df[markers_df[col] == max_val]
        # If max value occurs more than once
        if len(max_rows) > 1:
            # Store column name and indices of rows with max value
            multi_max_cols[col] = max_rows.index.tolist()
    print(f'Clusters with more than 1 maximum: {multi_max_cols}')

    # Manual Relabel
    new_cluster_names = max_index_dict.copy()
    new_cluster_names['7'] = CLUSTER_7

    mdata.mod['rna'].obs['cell_type'] = mdata.mod['rna'].obs.leiden_wnn.astype("str").values
    mdata.mod['rna'].obs.cell_type = mdata.mod['rna'].obs.cell_type.replace(new_cluster_names)
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

    # Generate alternate annotation txt file
    # Extract index and a column
    df_to_save = mdata.obs.reset_index()[['index', 'cell_type']]
    # Save to a txt file, separator can be specified, here it's a space
    df_to_save.to_csv(f'Data\PBMC 10k multiomic\WNN{FILE}-PBMC-10K-celltype.csv', index=True, header=True, sep='\t')
    return

def perform_pca(mdata_train, mdata_test, raw=False, components=20, random_state=42):
    '''
    Perform PCA on RNA and ATAC modalities of given mdata_train and mdata_test
    Raw = True: Perform PCA on raw counts using TruncatedSVD
    '''
    pca ={}
    for mod in ['rna', 'atac']:
        st=time.process_time()
        if raw == False:
            pca[mod] = PCA(n_components=components, random_state=random_state).fit(mdata_train.mod[mod].X)
        else:
            pca[mod] = TruncatedSVD(n_components=components, random_state=random_state).fit(mdata_train.mod[mod].layers['counts'])

        # Transform count matrix using pca and store in mdata object
        mdata_train.mod[mod].obsm['X_pca'] = pca[mod].transform(mdata_train.mod[mod].X)
        mdata_test.mod[mod].obsm['X_pca'] = pca[mod].transform(mdata_test.mod[mod].X)
        et=time.process_time()
        mdata_train.mod[mod].varm['PCs'] = np.transpose(pca[mod].components_)
        print(f"PCA {mod} took {et-st} seconds")

        # Scree plot
        if raw == False:
            PC_values = np.arange(pca[mod].n_components_) + 1
        else:
            PC_values = np.arange(pca[mod].explained_variance_ratio_.shape[0]) + 1
        plt.plot(PC_values, pca[mod].explained_variance_ratio_, 'ro-', linewidth=2)
        plt.title('Scree Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Proportion of Variance Explained')
        plt.show()
    
        # Ask user input for desired number of PCs and compute cumulative variance explained
        #num_pcs = int(input(f"Enter the number of PCs you want to keep for {mod}: "))
        #print(f"PCA {mod} with {num_pcs} PCs explains {np.cumsum(pca[mod].explained_variance_ratio_[0:num_pcs])[-1]*100}% of variance")

    return mdata_train, mdata_test, pca

def perform_cca(mdata_train, mdata_test, n_components=50):
    '''
    Performs CCA on the data and adds the CCA components to the mdata object.
    Also generates scree plot of correlation between each CCA component.
    '''
    # Perform CCA
    cca = CCA(n_components=n_components)
    st=time.process_time()
    cca.fit(mdata_train.mod['rna'].X, mdata_train.mod['atac'].X)
    rna_train, atac_train = cca.transform(mdata_train.mod['rna'].X, mdata_train.mod['atac'].X)
    rna_test, atac_test = cca.transform(mdata_test.mod['rna'].X, mdata_test.mod['atac'].X)
    et=time.process_time()
    print('CCA took {} seconds'.format(et-st))
    
    # Add CCA components to mdata
    mdata_train.mod['rna'].obsm['cca'] = rna_train
    mdata_train.mod['atac'].obsm['cca'] = atac_train
    mdata_test.mod['rna'].obsm['cca'] = rna_test
    mdata_test.mod['atac'].obsm['cca'] = atac_test

    # Scree plot
    # sklearn CCA doesn't directly provide the canonical correlations,
    # so we compute them as follows:
    correlations = [np.corrcoef(rna_train[:,i], atac_train[:,i])[0,1] for i in range(rna_train.shape[1])]

    # The canonical correlations
    plt.plot(range(1, len(correlations) + 1), correlations, 'ro-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlabel('Canonical Variate')
    plt.ylabel('Canonical Correlation')
    plt.show()

    return mdata_train, mdata_test


def add_annon(mdata_train, mdata_test, wnn):
    '''
    Adds annotations to mdata_train and mdata_test based on WNN clustering
    WNN = 0: RNA labels, WNN = 1: WNN Level 1 labels, WNN = 2: WNN Level 2 labels
    '''
    for df, name in zip([mdata_train, mdata_test],['train', 'test']):
        # Loading annotations
        if wnn == 0:
            annotations = pd.read_csv('Data\PBMC 10k multiomic\PBMC-10K-celltype.txt', sep='\t', header=0, index_col=0)
        elif wnn == 1:
            annotations = pd.read_csv('Data\PBMC 10k multiomic\WNNL1-PBMC-10K-celltype.csv', sep='\t', header=0, index_col='index')
        elif wnn == 2:
            annotations = pd.read_csv('Data\PBMC 10k multiomic\WNNL2-PBMC-10K-celltype.csv', sep='\t', header=0, index_col='index')
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

def scvi_process(mdata_train, mdata_test, epochs):
    '''
    Dimensionality reduction using scVI autoencoder
    '''
    for mod in ['rna', 'atac']:
        # Setup the anndata object
        scvi.model.SCVI.setup_anndata(mdata_train.mod[mod], layer="counts")
        # Create a model
        vae = scvi.model.SCVI(mdata_train.mod[mod])
        # Train the model
        vae.train(max_epochs=epochs)

        # Extract the low-dimensional representations
        mdata_train.mod[mod].obsm["X_scVI"] = vae.get_latent_representation(mdata_train.mod[mod])

        # Transform the test data
        scvi.model.SCVI.setup_anndata(mdata_test.mod[mod], layer="counts")
        mdata_test.mod[mod].obsm["X_scVI"] = vae.get_latent_representation(mdata_test.mod[mod])

    return mdata_train, mdata_test

def generate_feature_matrix(mdata_train, mdata_test, y_train, y_test, embedding, n_components_rna, n_components_atac):
    '''
    Generates feature matrix and removes NAs for training and test set based on embedding and number of components
    Embedding options of PCA, CCA and scVI
    '''
    if embedding == 'PCA':
        obsm = 'X_pca'
    elif embedding == 'CCA':
        obsm = 'cca'
    elif embedding == 'scVI':
        obsm = 'X_scVI'
    # Generating feature matrix for training set
    X_train = np.concatenate((mdata_train.mod['rna'].obsm[obsm][:,:n_components_rna], mdata_train.mod['atac'].obsm[obsm][:,:n_components_atac]), axis=1)
    # Convert to dataframe
    X_train = pd.DataFrame(X_train, columns=[f"RNA Comp{i}" for i in range(1,n_components_rna+1)] + [f"ATAC Comp{i}" for i in range(1,n_components_atac+1)])
    print(X_train.head())

    # Generating feature matrix for test set
    X_test = np.concatenate((mdata_test.mod['rna'].obsm[obsm][:,:n_components_rna], mdata_test.mod['atac'].obsm[obsm][:,:n_components_atac]), axis=1)
    # Convert to dataframe
    X_test = pd.DataFrame(X_test, columns=[f"RNA Comp{i}" for i in range(1,n_components_rna+1)] + [f"ATAC Comp{i}" for i in range(1,n_components_atac+1)])
    print(X_test.head())

    # Standardization
    sclr = StandardScaler().fit(X_train)
    X_train = sclr.transform(X_train)
    X_test = sclr.transform(X_test)

    # Convert back to dataframe format with column names
    X_train = pd.DataFrame(X_train, columns=[f"RNA PC{i}" for i in range(1,n_components_rna+1)] + [f"ATAC PC{i}" for i in range(1,n_components_atac+1)])
    X_test = pd.DataFrame(X_test, columns=[f"RNA PC{i}" for i in range(1,n_components_rna+1)] + [f"ATAC PC{i}" for i in range(1,n_components_atac+1)])

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

    print(X_train.head())
    print(X_test.head())
    print(set(y_train))
    print(set(y_test))

        
    return X_train, X_test, y_train, y_test

def save_data(labels, embedding, Xpca_train, Xpca_test, y_train, y_test, XscVI_train, XscVI_test, Xcca_train, Xcca_test):
    if embedding == 'PCA' and labels == 'rna':
        Xpca_train.to_pickle("Data/PBMC 10k multiomic/processed_data/X_Matrices/Xpca_train_35_RAW.pkl")
        Xpca_test.to_pickle("Data/PBMC 10k multiomic/processed_data/X_Matrices/Xpca_test_35_RAW.pkl")
    elif embedding == 'PCA' and (labels == 'wnn' or labels == 'wnnL1' or labels == 'wnnL2'):
        Xpca_train.to_pickle("Data/PBMC 10k multiomic/processed_data/X_Matrices/Xpca_train_35_wnn.pkl")
        Xpca_test.to_pickle("Data/PBMC 10k multiomic/processed_data/X_Matrices/Xpca_test_35_wnn.pkl")
    elif embedding == 'scVI' and labels == 'rna':
        XscVI_train.to_pickle("Data/PBMC 10k multiomic/processed_data/X_Matrices/XscVI_train_35.pkl")
        XscVI_test.to_pickle("Data/PBMC 10k multiomic/processed_data/X_Matrices/XscVI_test_35.pkl")
    elif embedding == 'scVI' and (labels == 'wnn' or labels == 'wnnL1' or labels == 'wnnL2'):
        XscVI_train.to_pickle("Data/PBMC 10k multiomic/processed_data/X_Matrices/XscVI_train_35_wnn.pkl")
        XscVI_test.to_pickle("Data/PBMC 10k multiomic/processed_data/X_Matrices/XscVI_test_35_wnn.pkl")
    elif embedding == 'CCA' and labels == 'rna':
        Xcca_train.to_pickle("Data/PBMC 10k multiomic/processed_data/X_Matrices/Xcca_train_35.pkl")
        Xcca_test.to_pickle("Data/PBMC 10k multiomic/processed_data/X_Matrices/Xcca_test_35.pkl")
    elif embedding == 'CCA' and (labels == 'wnn' or labels == 'wnnL1' or labels == 'wnnL2'):
        Xcca_train.to_pickle("Data/PBMC 10k multiomic/processed_data/X_Matrices/Xcca_train_35_wnn.pkl")
        Xcca_test.to_pickle("Data/PBMC 10k multiomic/processed_data/X_Matrices/Xcca_test_35_wnn.pkl")
    if labels == 'wnnL1':
        np.save("Data/PBMC 10k multiomic/y_train_wnnL1.npy", y_train)
        np.save("Data/PBMC 10k multiomic/y_test_wnnL1.npy", y_test)
    elif labels == 'wnnL2':
        np.save("Data/PBMC 10k multiomic/y_train_wnnL2.npy", y_train)
        np.save("Data/PBMC 10k multiomic/y_test_wnnL2.npy", y_test)
    elif labels == 'rna':
        np.save("Data/PBMC 10k multiomic/y_train.npy", y_train)
        np.save("Data/PBMC 10k multiomic/y_test.npy", y_test)

def choose_feature_set(feature_set, labels, n_components, resample):
    '''
    Function to choose feature set
    feature_set
        PCA Major = PCA with major cell types
        PCA Minor = PCA with minor cell types
        scVI = scVI (auto encoder) latent space
    labels
        rna = Ground truth from RNA
        wnnL1 = Ground truth from WNN major cell types
        wnnL2 = Ground truth from WNN minor cell types
    n_components = number of components to use from embedding for each modality
    Resample = True/False for SMOTE oversampling
    '''
    if feature_set == 'PCA MINOR':
        FEAT_PREFIX = 'Xpca'
    elif feature_set == 'scVI':
        FEAT_PREFIX = 'XscVI'
    elif feature_set == 'CCA':
        FEAT_PREFIX = 'Xcca'
    if labels == 'rna':
        LABEL_SUFFIX = ''
        FEAT_SUFFIX = ''
    elif labels == 'wnnL1':
        LABEL_SUFFIX = '_wnnL1'
        FEAT_SUFFIX = '_wnn'
    elif labels == 'wnnL2':
        LABEL_SUFFIX = '_wnnL2'
        FEAT_SUFFIX = '_wnn'
    if feature_set == 'PCA MAJOR':
        FEATURES_COMB_TRAIN = pd.read_pickle('Data/PBMC 10k multiomic/processed_data/X_Matrices/Xpca_train_mjr.pkl')
        FEATURES_COMB_TEST = pd.read_pickle('Data/PBMC 10k multiomic/processed_data/X_Matrices/Xpca_test_mjr.pkl')
    FEATURES_COMB_TRAIN = pd.read_pickle(f'Data/PBMC 10k multiomic/processed_data/X_Matrices/{FEAT_PREFIX}_train_{n_components}{FEAT_SUFFIX}.pkl')
    FEATURES_COMB_TEST = pd.read_pickle(f'Data/PBMC 10k multiomic/processed_data/X_Matrices/{FEAT_PREFIX}_test_{n_components}{FEAT_SUFFIX}.pkl') 
    LABELS_TRAIN = np.load(f'Data/PBMC 10k multiomic/y_train{LABEL_SUFFIX}.npy', allow_pickle=True)
    LABELS_TEST = np.load(f'Data/PBMC 10k multiomic/y_test{LABEL_SUFFIX}.npy', allow_pickle=True)  
    if resample == True:
        smote = SMOTE(random_state=42)
        FEATURES_COMB_TRAIN, LABELS_TRAIN = smote.fit_resample(FEATURES_COMB_TRAIN, LABELS_TRAIN)
    return FEATURES_COMB_TRAIN, FEATURES_COMB_TEST, LABELS_TRAIN, LABELS_TEST

def remove_cells(GROUND_TRUTH, CELL_TYPE, X_train, X_test, y_train, y_test):
    '''
    Remove low frequency cell types from the training and test set
    '''
    cells = []
    if CELL_TYPE == 'B cells':
        target_strings = ['B','NK']
    elif CELL_TYPE == 'T cells':
        target_strings = ['CD4', 'CD8']
    elif CELL_TYPE == 'Monoblast-Derived':
        target_strings = ['Mono', 'DC']
    elif CELL_TYPE == 'All':
        target_strings = set(y_train)
    cells = [s for s in set(y_train) if not any(target_string in s for target_string in target_strings)]
    if GROUND_TRUTH == 'wnnL2':
        cells.append('Plasmablast') # list of low freq cells
        cells.append('dnT')
    print(cells)
    # Print initial shapes
    print(f"Initial shapes of X: {X_train.shape}, {X_test.shape}; and y: {y_train.shape}, {y_test.shape}")

    # Create DataFrame from X_train and X_test
    train_data = X_train.copy()
    test_data = X_test.copy()

    # Add y_train and y_test as columns
    train_data['label'] = y_train
    test_data['label'] = y_test

    # Removing rows with specfic cell types
    train_data = train_data.loc[~train_data['label'].isin(cells)]
    test_data = test_data.loc[~test_data['label'].isin(cells)]

    # Separate X_train, y_train, X_test, and y_test from the updated DataFrame
    X_train = train_data.iloc[:, :-1]
    y_train = train_data['label'].to_numpy()

    X_test = test_data.iloc[:, :-1]
    y_test = test_data['label'].to_numpy()

    # Print new shapes
    print(f"New shapes of X: {X_train.shape}, {X_test.shape}; and y: {y_train.shape}, {y_test.shape}")

    return X_train, X_test, y_train, y_test

def bootstrap_confidence_interval(model, X, y, n_bootstrap=1000):
    """
    Function to estimate a 95% confidence interval for a model's F1 score and PAP score using bootstrapping.
    """

    # Get the classes
    classes = np.unique(y)
    f1_scores_per_class = defaultdict(list)
    f1_scores_overall = []
    pap_scores_per_class = defaultdict(list)

    for _ in range(n_bootstrap):
        X_resample, y_resample = resample(X, y, n_samples=len(X) // 2)
        y_pred = model.predict(X_resample)

        # Compute F1 scores for each class
        f1_scores = f1_score(y_resample, y_pred, average=None)

        # Compute overall F1 score
        f1_score_overall = f1_score(y_resample, y_pred, average='macro')
        f1_scores_overall.append(f1_score_overall)

        # Compute prediction probabilities
        proba = model.predict_proba(X_resample)
        # Compute PAP for each class
        for j in range(proba.shape[1]):
            # Create an ECDF for this class's prediction probabilities
            ecdf = ECDF(proba[:, j])

            # Compute PAP score
            cdf_x1 = ecdf(0.1)
            cdf_x2 = ecdf(0.9)
            pap_score = cdf_x2 - cdf_x1

            # Append PAP score for this class
            pap_scores_per_class[classes[j]].append(pap_score)

        # Append F1 scores for each class
        for class_, f1_score_ in zip(classes, f1_scores):
            f1_scores_per_class[class_].append(f1_score_)

    for class_, scores in f1_scores_per_class.items():
        print(f"Class: {class_}, number of scores: {len(scores)}")

    # Generate dataframes of bootstrap distributions
    df_f1_bootstrap = pd.DataFrame.from_dict(f1_scores_per_class)
    df_pap_bootstrap = pd.DataFrame.from_dict(pap_scores_per_class)

    # Initialize lists for DataFrame
    class_list = []
    lower_f1_list = []
    upper_f1_list = []
    mean_f1_list = []
    lower_pap_list = []
    upper_pap_list = []
    mean_pap_list = []
    f1_scores_list = []
    class_scores_list = []
    pap_scores_list = []
    class_pap_scores_list = []

    # Compute and print confidence intervals per class
    for class_ in classes:

        # Compute confidence intervals for metrics
        lower_f1 = np.percentile(f1_scores_per_class[class_], 2.5)
        upper_f1 = np.percentile(f1_scores_per_class[class_], 97.5)
        mean_f1 = np.mean(f1_scores_per_class[class_])

        lower_pap = np.percentile(pap_scores_per_class[class_], 2.5)
        upper_pap = np.percentile(pap_scores_per_class[class_], 97.5)
        mean_pap = np.mean(pap_scores_per_class[class_])

        # Add data to lists
        class_list.append(class_)
        lower_f1_list.append(lower_f1)
        upper_f1_list.append(upper_f1)
        mean_f1_list.append(mean_f1)
        lower_pap_list.append(lower_pap)
        upper_pap_list.append(upper_pap)
        mean_pap_list.append(mean_pap)
        # Add F1 scores to list
        f1_scores_list += f1_scores_per_class[class_] # Add F1 scores for this class
        class_scores_list += [class_] * len(f1_scores_per_class[class_]) # Repeat class name for each F1 score
        # Add PAP scores to list
        pap_scores_list += pap_scores_per_class[class_] # Add PAP scores for this class
        class_pap_scores_list += [class_] * len(pap_scores_per_class[class_]) # Repeat class name for each PAP score

    # Compute and print confidence intervals for overall F1 score
    lower_f1_overall = np.percentile(f1_scores_overall, 2.5)
    upper_f1_overall = np.percentile(f1_scores_overall, 97.5)
    mean_f1_overall = np.mean(f1_scores_overall)

    # Add overall data to lists
    class_list.append('Overall')
    lower_f1_list.append(lower_f1_overall)
    upper_f1_list.append(upper_f1_overall)
    mean_f1_list.append(mean_f1_overall)
    lower_pap_list.append(None)
    upper_pap_list.append(None)
    mean_pap_list.append(None)
    # Add overall F1 scores to list
    f1_scores_list += f1_scores_overall
    class_scores_list += ['Overall'] * len(f1_scores_overall)

    # Create DataFrame
    df = pd.DataFrame({
        'class': class_list,
        'mean F1 score': mean_f1_list,
        'lower F1 CI': lower_f1_list,
        'upper F1 CI': upper_f1_list,
        'mean PAP score': mean_pap_list,
        'lower PAP CI': lower_pap_list,
        'upper PAP CI': upper_pap_list
    })
    print(df_f1_bootstrap.head())
    print(df_pap_bootstrap.head())
    print(df)
    
    # Create DataFrame for visualization
    df_viz = pd.DataFrame({
        'F1 score': f1_scores_list,
        'class': class_scores_list
    })

    # Plot histogram of F1 scores for each class in a single facet plot
    g = sns.FacetGrid(df_viz, col="class", col_wrap=5, sharex=False, sharey=True)
    g.map(plt.hist, "F1 score")
    g.set_titles("{col_name}")
    plt.suptitle('Histogram of F1 scores for each class', y=1.02) # Adding a main title above all the subplots
    plt.show()

    # Create DataFrame for PAP visualization
    df_viz_pap = pd.DataFrame({
        'PAP score': pap_scores_list,
        'class': class_pap_scores_list
    })

    # Plot histogram of PAP scores for each class in a single facet plot
    g_pap = sns.FacetGrid(df_viz_pap, col="class", col_wrap=5, sharex=True, sharey=True)
    g_pap.map(plt.hist, "PAP score")
    g_pap.set_titles("{col_name}")
    plt.suptitle('Histogram of PAP scores for each class', y=1.02) # Adding a main title above all the subplots
    plt.show()

    return df, df_f1_bootstrap, df_pap_bootstrap

def model_test_main(model,x_train,y_train,x_test,y_test, subset, table_one_only=False):
    '''
    Function to test a model on a train and test set
    Subset = True: Use only 500 samples from train set for quick testing
    '''
    if subset == True:
        x_train = x_train.iloc[:500,:]
        y_train = y_train[:500]
        cv_splits = 1
    else:
        cv_splits = 5
    start_time = time.process_time()
    #Hyperparameter dictionaries
    if  isinstance(model, XGBClassifier):
        param_grid={'n_estimators'     : np.array([4**2, 4**3, 4**4]),
                    'learning_rate'    : np.array([1e-2, 1e-1, 1]),
                    'max_depth'        : np.array([2**2, 2**3, 2**4])}
        #Encoding labels for xgb compatability
        encoder = LabelEncoder()
        y_train = encoder.fit_transform(y_train)
        y_test = encoder.transform(y_test)
    elif  isinstance(model, RandomForestClassifier):
        param_grid={'n_estimators'   : np.array([20, 40, 60]),
                    'min_samples_leaf'   : np.array([5, 10]),
                    'max_features' : np.array(['sqrt']),
                    'max_depth':np.array([2**2, 2**3, 2**4]),
                    'min_samples_split': np.array([3, 5, 10])}
    elif isinstance(model, svm.SVC):
        param_grid={'C':[2**0, 2**1, 2**2, 2**3, 2**4, 2**5],
                    'kernel':['poly','rbf', 'sigmoid']}
    elif isinstance(model, LogisticRegression):
        param_grid={'penalty':['elasticnet','none'],
                    'l1_ratio':[0,0.2,0.4,0.6,0.8,1],
                    'C':[2**0, 2**1, 2**2, 2**3, 2**4, 2**5],
                    'solver':['saga']}
    inner=ShuffleSplit(n_splits=cv_splits,test_size=0.3,random_state=0)

    #Inner CV for hyperparameter tuning
    f1 = make_scorer(f1_score , average='macro')
    search_results=GridSearchCV(model,param_grid,cv=inner,n_jobs=-1, scoring=f1,return_train_score = True, verbose=3).fit(x_train,y_train)
    #Selecting optimal hyperparameters
    optimal_params=search_results.best_params_
    print(f'Best Params:{optimal_params}, with score of {search_results.best_score_}')
    #Re-instantiating models with optimal hyperparameters
    if  isinstance(model, XGBClassifier):
        model=XGBClassifier(**optimal_params, random_state=42, scale_pos_weight=weights_dict)
    elif  isinstance(model, RandomForestClassifier):
        model =RandomForestClassifier(**optimal_params, random_state=42, class_weight='balanced')    
    elif isinstance(model, svm.SVC):
        model=svm.SVC(**optimal_params, random_state=42, probability=True, class_weight='balanced')
    elif isinstance(model, LogisticRegression):
        model=LogisticRegression(**optimal_params, random_state=42, class_weight='balanced')
    #Learning curve
    if subset == False:
        LearningCurveDisplay.from_estimator(model, x_train, y_train, random_state=42, 
                                            score_type = 'both', n_jobs = -1, scoring = f1, verbose = 4)
        plt.show()
    #Fitting model with optimised parameters to training data
    model.fit(x_train,y_train)

    #Predicting using fitted model on train set
    y_pred_train = model.predict(x_train)
    #Predicting using fitted model on test set
    y_pred_test = model.predict(x_test)
    time_taken=(time.process_time() - start_time)
    print(f'CPU time for training and testing: {time_taken} seconds or {time_taken/60} mins or {time_taken/(60*60)} hrs')
    
    # Get results for train and test sets
    df_list = []
    # Create a list to hold the PAP scores
    pap_list = []
    for i, (predictions, observations, features) in enumerate(zip([y_pred_train, y_pred_test], [y_train, y_test], [x_train, x_test])):
        # De-encoding labels for xgb
        if isinstance(model, XGBClassifier):
            predictions = encoder.inverse_transform(predictions)
            observations = encoder.inverse_transform(observations)
        # Results Visualisation
        print(f'{["Train", "Test"][i]} Set Results:')
        print(classification_report(observations, predictions))
        proba = model.predict_proba(features)
        print(f"AUC: {roc_auc_score(observations, proba, multi_class='ovr')}")
        plt.rcParams.update({'font.size': 8})
        cmatrix = ConfusionMatrixDisplay.from_predictions(observations, predictions, xticks_rotation='vertical')
        plt.show(cmatrix)

        # PAP - Percentage Arbitrary Preditions metric
        # Compute prediction probabilities and create a DataFrame
        x1 = 0.1
        x2 = 0.9
        for j in range(proba.shape[1]):
            df = pd.DataFrame()
            df['Probability'] = proba[:, j]
            df['Class'] = f'Class {model.classes_[j]}'
            df['Set'] = ["Train", "Test"][i]
            # Compute the PAC score
            ecdf = ECDF(proba[:, j])
            # computing the 10th and 90th percentiles of your data
            #p10 = np.percentile(proba[:, j], x1)
            #p90 = np.percentile(proba[:, j], x2)

            # evaluating the ECDF at the 10th and 90th percentiles
            cdf_x1 = ecdf(x1)
            cdf_x2 = ecdf(x2)

            # computing the PAP score
            pap_score = cdf_x2 - cdf_x1
            # Store the PAP score in the list
            pap_list.append({'Class': f'{model.classes_[j]}', 'Set': ["Train", "Test"][i], 'PAP': pap_score})
            df_list.append(df)
        

    # Convert the list of dictionaries to a DataFrame
    pap_df = pd.DataFrame(pap_list)
    # Concatenate all DataFrames
    df_total = pd.concat(df_list, ignore_index=True)

    # Plot the ECDFs
    fig = px.ecdf(df_total, x="Probability", color="Class", facet_row="Set")
    # Set the width and height of the figure
    fig.update_layout(autosize=False, width=800, height=600)
    fig.show()

    # Estimate confidence interval for F1 score
    start_time = time.process_time()
    metrics, f1_df, pap_df = bootstrap_confidence_interval(model, x_test, y_test)
    time_taken=(time.process_time() - start_time)
    print(f"95% confidence interval for F1 score: ({metrics[metrics['class'] == 'Overall']['lower F1 CI'].values[0]:.3f}, {metrics[metrics['class'] == 'Overall']['upper F1 CI'].values[0]:.3f}, mean: {metrics[metrics['class'] == 'Overall']['mean F1 score'].values[0]:.3f})")
    print(f'CPU time for boostrap: {time_taken} seconds or {time_taken/60} mins or {time_taken/(60*60)} hrs')

    return(model, y_pred_test, metrics, f1_df, pap_df)

def save_model(model_cl, location, y_pred_test, y_test, f1_df, pap_df):
    '''
    Save predictions for model as a csv and change name depending on the model
    '''
    if isinstance(model_cl,RandomForestClassifier):
        model_name='rf'
    elif isinstance(model_cl,XGBClassifier):
        model_name='xgb'
    elif isinstance(model_cl,svm.SVC):
        model_name='svm'
    #Save model
    pickle.dump(model_cl, open(f'Supervised Models\{location}.pickle', 'wb'))
    #Save predictions and observations in a pickle file
    df = pd.DataFrame(
        {"Observed" : y_pred_test,
        "Predictions" : y_test})
    df.to_pickle(f'Supervised Models\{location}_Predictions.pickle')
    f1_df.to_csv(f'Supervised Models\{location}_F1_df.csv')
    pap_df.to_csv(f'Supervised Models\{location}_PAP_df.csv')


def visualise_embeddings(features, labels):
    '''
    Visualise embeddings using UMAP
    '''
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(features)
    '''
    Plot using Plotly
    df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
    df['labels'] = labels

    fig = px.scatter(df, x='UMAP1', y='UMAP2', color='labels', title="UMAP Visualization")
     # Customize marker appearance
    fig.update_traces(marker=dict(opacity=0.8, 
                                  line=dict(width=0.2,
                                            color='White')))
    # Set the width and height of the figure
    fig.update_layout(autosize=False, width=800, height=600)
    fig.show()
    '''
    # Plot using Seaborn
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hue=labels,  # Provide your label array here
        palette=sns.color_palette("Paired", len(np.unique(labels))),  # Choose a color palette
        legend="full",
        alpha=0.8
    )
    plt.title("UMAP Visualization")
    plt.show()
    return embedding


def feature_importance(model, X_test, mdata_train):

    '''
    Explain feature importance for components using SHAP + loading coefficients
    using a custom metric
    '''
    if isinstance(model, RandomForestClassifier):

        # Create a tree explainer
        explainer = shap.TreeExplainer(model)
    elif isinstance(model, LogisticRegression):
    
        # Create a linear explainer
        explainer = shap.LinearExplainer(model, X_test)
    
    shap_values = explainer.shap_values(X_test)

    # Visualize the training set predictions
    for cell in range(0, len(shap_values)):
        shap.summary_plot(shap_values[cell], X_test, title=f"Cell {cell} SHAP Summary Plot")

    # Create a dictionary to store feature importances per gene per cell
    feat_imp = {}
    for gene in range(0, mdata_train.mod['rna'].varm['PCs'].shape[0]): 
        feat_imp[gene] = {}
        for cell in range(0,(len(shap_values))):
            # Computing feature importance as a function of gene loading and PC SHAP
            pc_loadings_gene = mdata_train.mod['rna'].varm['PCs'][gene]
            gene_imp = []
            for pc in range(0,(shap_values[0].shape[1])):
                pc_loadings_total = np.sum(np.abs(mdata_train.mod['rna'].varm['PCs'][:,pc]))
                # Normalize PCA loadings for each PC
                pc_loadings_norm = np.abs(pc_loadings_gene[pc]) / pc_loadings_total
                # Normalize SHAP values for a class
                shap_values_pc_norm = np.sum(np.abs(shap_values[cell][pc])) / np.sum(np.abs(shap_values[0]))
                # Compute feature contributions
                feature_contributions = pc_loadings_norm * shap_values_pc_norm
                # Compute feature importances
                feature_importances = np.sum(feature_contributions)
                gene_imp.append(feature_importances)
            feat_imp[gene][cell]=sum(gene_imp)
        
    return shap_values, feat_imp


def plot_loading_coefficients(shap_values, mdata_train):
    '''
    Function to plot loading coefficients for each PC
    '''
    # Extract the PCs
    pcs = mdata_train.mod['rna'].varm['PCs']

    # Convert to a DataFrame for easier handling
    pcs_df = pd.DataFrame(pcs, index=mdata_train.mod['rna'].var_names)

    for pc_idx in range(0,(shap_values[0].shape[1])):

        # Get the loadings for the first PC
        pc_loadings = pcs_df.iloc[:, pc_idx] 

        # Sort the loadings
        sorted_pc_loadings = pc_loadings.sort_values(ascending=False)

        # Create a bar plot of the loadings
        plt.figure(figsize=(10,5))
        sorted_pc_loadings[:20].plot(kind='bar') # you can change the number to show more or fewer genes
        plt.ylabel('Loading coefficient')
        plt.title(f'Top contributing genes to PC {pc_idx+1}')
        plt.xticks(rotation=45)
        plt.show()

def remove_outliers(method, train_features, train_labels, test_features, test_labels, threshold):

    '''
    (Not currently used) Function to remove outliers from training and test set
    '''
    # let's assume df is your DataFrame and 'class' is your class column
    train_features['label'] = train_labels.tolist()
    test_features['label'] = test_labels.tolist()
    df = pd.concat([train_features, test_features], axis=0)
    print(df.head())
    # Perform UMAP
    umap_components = ut.visualise_embeddings(df.drop(['label'],axis=1), df['label'])

    if method == 'LOF':
        
        # List to hold outlier indices
        outlier_indices = []

        # Get unique class labels
        classes = df['label'].unique()

        # Apply LOF per class
        for class_label in classes:
            # Subset df for the current class
            df_class = df[df['label'] == class_label]

            # Fit the model
            lof = LocalOutlierFactor(n_neighbors=20)  # Adjust parameters as needed
            y_pred = lof.fit_predict(df_class.drop(['label'], axis=1))

            # Negative scores represent outliers
            outlier_scores = -lof.negative_outlier_factor_

            # Consider data points as outliers based on a certain threshold on the LOF scores
            outliers = outlier_scores > np.percentile(outlier_scores, threshold)  # Here we consider top 5% as outliers

            # Collect indices of outliers
            outlier_indices.extend(df_class[outliers].index.tolist())

        # Now outlier_indices contains the indices of all within-class outliers
        # You can use this to filter your DataFrame
        filtered_df = df.drop(outlier_indices)

    elif method == 'UMAP':

        # Combine UMAP components and labels into a single DataFrame
        umap_df = pd.DataFrame(umap_components, columns=["UMAP1", "UMAP2"])
        df = pd.concat([umap_df, df['label'].reset_index(drop=True)], axis=1)

        # Calculate the centroid for each class
        centroids = df.groupby("label").mean()

        # Initialize an empty list to store the distances
        distances = []

        # Loop over each class
        for label in df["label"].unique():
            # Subset the DataFrame to include only data points from the current class
            df_class = df[df["label"] == label]
            
            # Calculate the distance from each point in this class to the centroid of the class
            for idx, row in df_class.iterrows():
                dist = distance.euclidean(row[["UMAP1", "UMAP2"]], centroids.loc[label])
                distances.append(dist)

        # Add distances to the DataFrame
        df["distance_to_centroid"] = distances

        # Now, you can consider points with a distance to the centroid above a certain threshold as outliers
        threshold = df["distance_to_centroid"].quantile(0.95)  # 95th percentile, adjust as needed
        outliers = df[df["distance_to_centroid"] > threshold]
        # Create an index of non-outliers
        non_outliers = df[df["distance_to_centroid"] <= threshold].index

        # Create the filtered dataframe
        filtered_df = df.loc[non_outliers]

    # Perform new UMAP
    ut.visualise_embeddings(filtered_df.drop(['label'],axis=1), filtered_df['label'])

    return filtered_df
    

# %%
