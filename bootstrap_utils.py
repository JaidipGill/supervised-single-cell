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
from sklearn.metrics import precision_score, recall_score

import Utils as ut
from importlib import reload
from sklearn.utils import resample

'''
This module contains variants of the functions in Utils.py compatible with bootstrapping
'''
def train_test_split_mdata(mdata_obj, seed):
    '''
    Split MuData object into train and test sets
    '''
    # Get the indices of cells
    indices = np.arange(mdata_obj['rna'].shape[0])

    # Perform bootstrap sampling to get the training indices
    train_idx = resample(indices, replace=True,random_state=seed)

    # Identify the out-of-bag samples for the test set
    test_idx = np.setdiff1d(indices, train_idx)

    # Create train and test MuData objects
    mdata_train = mu.MuData({mod: mdata_obj[mod][train_idx] for mod in mdata_obj.mod.keys()})
    mdata_test = mu.MuData({mod: mdata_obj[mod][test_idx] for mod in mdata_obj.mod.keys()})

    # Ensure unique obs_names
    for mod in mdata_train.mod.keys():
        mdata_train[mod].obs_names_make_unique(join='_')
        mdata_test[mod].obs_names_make_unique(join='_')

    # Convert views to AnnData objects
    mdata_train = mdata_train.copy()
    mdata_test = mdata_test.copy()

    return mdata_train, mdata_test

def add_annon(mdata, subset, data, GROUND_TRUTH):
    '''
    Adds annotations to mdata_train and mdata_test based on WNN clustering
    WNN = 0: RNA labels, WNN = 1: WNN Level 1 labels, WNN = 2: WNN Level 2 labels
    '''
    if subset==True:
        mdata = mdata[:1000]

    if data == 'cancer':
        if GROUND_TRUTH == 'rna':
            annotations = pd.read_csv('Data/B cell lymphoma/Original Cell Types.csv', sep=',', index_col=0)
        elif GROUND_TRUTH == 'wnnL1':
            annotations= pd.read_csv('Data/B cell lymphoma/wnn Cell Types 1.csv',  sep='\t', header=0, index_col=1)
        elif GROUND_TRUTH == 'T cells':
            annotations= pd.read_csv('Data/B cell lymphoma/T-cell Subtypes.csv',  sep=',', header=0, index_col=0)
        elif GROUND_TRUTH == 'wnnL2':
            annotations= pd.read_csv('Data/B cell lymphoma/Complete Cell Types.csv',  sep=',', index_col=0) 
        ann_list= [annotations]
    elif data == 'pbmc':
        annotations_rna = pd.read_csv('Data/PBMC 10k multiomic/PBMC-10K-celltype.txt', sep='\t', header=0, index_col=0)
        annotations_wnn1 = pd.read_csv('Data/PBMC 10k multiomic/WNNL1-PBMC-10K-celltype.csv', sep='\t', header=0, index_col='index')
        annotations_wnn2 = pd.read_csv('Data/PBMC 10k multiomic/WNNL2-PBMC-10K-celltype.csv', sep='\t', header=0, index_col='index')
        ann_list= [annotations_rna, annotations_wnn1, annotations_wnn2]
    elif data == 'AD':
        annotations_wnn1 = pd.read_csv('Data/Alz multiomic/GSE214979_cell_metadata.csv')
        # Set the index to be the cell barcode
        annotations_wnn1.set_index('Unnamed: 0', inplace=True)
        annotations_wnn1.index.rename('index', inplace=True)
        # Drop all columns except cell type and index
        annotations_wnn1 = annotations_wnn1[['predicted.id']]
        annotations_wnn1 = annotations_wnn1.rename(columns={'predicted.id': 'x'})
        ann_list = [annotations_wnn1]
    for idx, annotations in enumerate(ann_list):
        # Take intersection of cell barcodes in annotations and mdata
        print(annotations)
        if 'cell_type' in annotations.columns:
            annotations = annotations.rename(columns={'cell_type': 'x'})
        elif 'T-cell Subtypes' in annotations.columns:
            annotations = annotations.rename(columns={'T-cell Subtypes': 'x'})
        common_barcodes = annotations.index.intersection(mdata.obs_names)
        print(common_barcodes)
        # Filter annotations and mdata to keep only common barcodes
        annotations = annotations.loc[common_barcodes]
        print(annotations)
        # Add the annotations to the .obs DataFrame
        mdata.obs = pd.concat([mdata.obs, annotations], axis=1)
        print(mdata.obs.columns)
        mdata.obs.rename(columns={'x': f'cell_type{idx}'}, inplace=True)
        #print(mdata.obs[f'cell_type{idx}'])
        mdata.mod['rna'].obs[f'cell_type{idx}'] = mdata.obs[f'cell_type{idx}']
        mdata.mod['atac'].obs[f'cell_type{idx}'] = mdata.obs[f'cell_type{idx}']

        # Count number of NAs in cell_type column
        print(f"Cell_type {idx} NAs: {mdata.obs[f'cell_type{idx}'].isna().sum()}")
        mu.pp.filter_obs(mdata['rna'], f'cell_type{idx}', lambda x: pd.notna(x))
        print(f"Filtered: Cell_type {idx} NAs: {mdata['rna'].obs[f'cell_type{idx}'].isna().sum()}")
    '''
    if data == 'cancer':
        # Get the cell type annotations
        cell_types = mdata.obs['cell_type0'].values
        # Create a Boolean mask to select the cells with the desired cell type
        mask = (cell_types == 'B cell')|(cell_types == 'Tumour B cell')
        # Subset the mudata object using the mask
        subset_mdata = mdata[mask]

    print(subset_mdata)
    '''
    return mdata

def load_boot(i, N, INPUT_ADDRESS, EMBEDDING, GROUND_TRUTH, CELL_TYPE, DATA, N_COMPONENTS_TO_TEST, GROUND_TRUTH_SUFFIX):
    '''
    Load Bootstrap and OOB samples into dataframe for testing
    '''
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
        #noise = np.random.normal(loc=0, scale=0.5, size=X_train.shape)
        #X_train = X_train + noise
        #noise = np.random.normal(loc=0, scale=0.5, size=X_test.shape)
        #X_test = X_test + noise
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

    return FEATURES_RNA_TRAIN, FEATURES_RNA_TEST, FEATURES_COMB_TRAIN, FEATURES_COMB_TEST, LABELS_TRAIN, LABELS_TEST

def model_test_main(model, outcome, x_train,y_train,x_test,y_test, subset, classes, f1_scores_per_class, f1_scores_overall, pap_scores_per_class, 
                    precision_scores_per_class, precision_scores_overall, recall_scores_per_class, recall_scores_overall, detailed):
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
    print(x_train.shape)
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
        param_grid={'C': [0.01, 0.1, 1, 2],
                    'kernel':['poly'], 
                    'gamma': [0.01, 0.1, 1]}
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
    if detailed == True:
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
    #df_list = []
    # Create a list to hold the PAP scores
    #pap_list = []
    for i, (predictions, observations, features) in enumerate(zip([y_pred_train, y_pred_test], [y_train, y_test], [x_train, x_test])):
        # De-encoding labels for xgb
        if isinstance(model, XGBClassifier):
            predictions = encoder.inverse_transform(predictions)
            observations = encoder.inverse_transform(observations)
        # Results Visualisation
        print(f'{["Train", "Test"][i]} Set Results:')
        print(classification_report(observations, predictions))
        proba = model.predict_proba(features)
        if outcome == 'multi':
            print(f"AUC: {roc_auc_score(observations, proba, multi_class='ovr')}") # AUC for multi-class
        if outcome == 'binary':
            print(f"AUC: {roc_auc_score(observations, proba[:,1])}") # AUC for binary
        plt.rcParams.update({'font.size': 8})
        cmatrix = ConfusionMatrixDisplay.from_predictions(observations, predictions, xticks_rotation='vertical')
        plt.show(cmatrix)

    # Metric calculations

    # Compute F1 scores for each class
    f1_scores = f1_score(y_test, y_pred_test, average=None)
    # Compute overall F1 score
    f1_score_overall = f1_score(y_test, y_pred_test, average='macro')
    f1_scores_overall.append(f1_score_overall)

    # Compute Precision scores for each class
    precision_scores = precision_score(y_test, y_pred_test, average=None)
    # Compute overall Precision score
    precision_score_overall = precision_score(y_test, y_pred_test, average='macro')
    precision_scores_overall.append(precision_score_overall)

    # Compute Recall scores for each class
    recall_scores = recall_score(y_test, y_pred_test, average=None)
    # Compute overall Recall score
    recall_score_overall = recall_score(y_test, y_pred_test, average='macro')
    recall_scores_overall.append(recall_score_overall)

    # Compute prediction probabilities
    proba = model.predict_proba(x_test)
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

    # Append F1, Precision, and Recall scores for each class
    for class_, f1, prec, rec in zip(classes, f1_scores, precision_scores, recall_scores):
        f1_scores_per_class[class_].append(f1)
        precision_scores_per_class[class_].append(prec)
        recall_scores_per_class[class_].append(rec)
    
    '''
    if detailed == True:
        # Convert the list of dictionaries to a DataFrame
        pap_df = pd.DataFrame(pap_list)
        # Concatenate all DataFrames
        df_total = pd.concat(df_list, ignore_index=True)

        # Plot the ECDFs
        fig = px.ecdf(df_total, x="Probability", color="Class", facet_row="Set")
        # Set the width and height of the figure
        fig.update_layout(autosize=False, width=800, height=600)
        fig.show()
    '''
    # Estimate confidence interval for F1 score
    #start_time = time.process_time()
    #metrics, f1_df, pap_df = bootstrap_confidence_interval(model, x_test, y_test)
    time_taken=(time.process_time() - start_time)
    #print(f"95% confidence interval for F1 score: ({metrics[metrics['class'] == 'Overall']['lower F1 CI'].values[0]:.3f}, {metrics[metrics['class'] == 'Overall']['upper F1 CI'].values[0]:.3f}, mean: {metrics[metrics['class'] == 'Overall']['mean F1 score'].values[0]:.3f})")
    #print(f'CPU time for boostrap: {time_taken} seconds or {time_taken/60} mins or {time_taken/(60*60)} hrs')
    print(f1_scores_per_class)

    return(model, y_pred_test, pap_scores_per_class, f1_scores_per_class, f1_scores_overall, precision_scores_per_class, precision_scores_overall, recall_scores_per_class, recall_scores_overall)

def analyse_metrics(results, suffix, rna, save):
    '''
    Analyse metrics from bootstrapping
    - Derives confidence intervals of F1 and PAP scores for each class using
      percentile method
    - Plots histograms of F1 and PAP scores for each class
    '''       
    if rna == True:
        f1_scores_per_class = results['rna_f1']
        f1_scores_overall = results['rna_f1_overall']
        pap_scores_per_class = results['rna_pap']
        precision_scores_per_class = results['rna_precision']
        precision_scores_overall = results['rna_precision_overall']
        recall_scores_per_class = results['rna_recall']
        recall_scores_overall = results['rna_recall_overall']
    elif rna == False:
        f1_scores_per_class = results['comb_f1']
        f1_scores_overall = results['comb_f1_overall']
        pap_scores_per_class = results['comb_pap']
        precision_scores_per_class = results['comb_precision']
        precision_scores_overall = results['comb_precision_overall']
        recall_scores_per_class = results['comb_recall']
        recall_scores_overall = results['comb_recall_overall']

    # Get list of classes
    classes = list(f1_scores_per_class.keys())
    for class_, scores in f1_scores_per_class.items():
        print(f"Class: {class_}, number of scores: {len(scores)}")
        
    # Generate dataframes of bootstrap distributions
    df_f1_bootstrap = pd.DataFrame.from_dict(f1_scores_per_class)
    df_pap_bootstrap = pd.DataFrame.from_dict(pap_scores_per_class)

    df_f1_overall = pd.DataFrame.from_dict(f1_scores_overall)
    df_precision_overall = pd.DataFrame.from_dict(precision_scores_overall)
    df_recall_overall = pd.DataFrame.from_dict(recall_scores_overall)

    df_precision_class = pd.DataFrame.from_dict(precision_scores_per_class)
    df_recall_class = pd.DataFrame.from_dict(recall_scores_per_class)

    if save == True:
        if rna == True:
            suf = '_rna'
        elif rna == False:
            suf = ''
        #df_f1_bootstrap.to_csv(f'Supervised Models/Results_{suffix}_F1_df{suf}.csv')
        #df_pap_bootstrap.to_csv(f'Supervised Models/Results_{suffix}_PAP_df{suf}.csv')
        df_f1_overall.to_csv(f'Supervised Models/Macro Metrics/Results_{suffix}_F1_overall_df{suf}.csv')
        df_precision_overall.to_csv(f'Supervised Models/Macro Metrics/Results_{suffix}_Precision_overall_df{suf}.csv')
        df_recall_overall.to_csv(f'Supervised Models/Macro Metrics/Results_{suffix}_Recall_overall_df{suf}.csv')
        #df_precision_class.to_csv(f'Supervised Models/Results_{suffix}_Precision_class_df{suf}.csv')
        #df_recall_class.to_csv(f'Supervised Models/Results_{suffix}_Recall_class_df{suf}.csv')

    # Initialize lists for DataFrame
    class_list = []

    lower_f1_list = []
    upper_f1_list = []
    mean_f1_list = []
    f1_scores_list = []
    class_scores_list = []

    lower_pap_list = []
    upper_pap_list = []
    mean_pap_list = []
    pap_scores_list = []
    class_pap_scores_list = []

    lower_precision_list = []
    upper_precision_list = []
    mean_precision_list = []

    lower_recall_list = []
    upper_recall_list = []
    mean_recall_list = []


    # Compute and print confidence intervals per class
    for class_ in classes:

        # Compute confidence intervals for metrics
        lower_f1 = np.percentile(f1_scores_per_class[class_], 2.5)
        upper_f1 = np.percentile(f1_scores_per_class[class_], 97.5)
        mean_f1 = np.mean(f1_scores_per_class[class_])
        lower_pap = np.percentile(pap_scores_per_class[class_], 2.5)
        upper_pap = np.percentile(pap_scores_per_class[class_], 97.5)
        mean_pap = np.mean(pap_scores_per_class[class_])
        lower_precision = np.percentile(precision_scores_per_class[class_], 2.5)
        upper_precision = np.percentile(precision_scores_per_class[class_], 97.5)
        mean_precision = np.mean(precision_scores_per_class[class_])
        lower_recall = np.percentile(recall_scores_per_class[class_], 2.5)
        upper_recall = np.percentile(recall_scores_per_class[class_], 97.5)
        mean_recall = np.mean(recall_scores_per_class[class_])

        # Add data to lists
        class_list.append(class_)
        lower_f1_list.append(lower_f1)
        upper_f1_list.append(upper_f1)
        mean_f1_list.append(mean_f1)
        lower_pap_list.append(lower_pap)
        upper_pap_list.append(upper_pap)
        mean_pap_list.append(mean_pap)
        lower_precision_list.append(lower_precision)
        upper_precision_list.append(upper_precision)
        mean_precision_list.append(mean_precision)
        lower_recall_list.append(lower_recall)
        upper_recall_list.append(upper_recall)
        mean_recall_list.append(mean_recall)

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
    lower_precision_overall = np.percentile(precision_scores_overall, 2.5)
    upper_precision_overall = np.percentile(precision_scores_overall, 97.5)
    mean_precision_overall = np.mean(precision_scores_overall)
    lower_recall_overall = np.percentile(recall_scores_overall, 2.5)
    upper_recall_overall = np.percentile(recall_scores_overall, 97.5)
    mean_recall_overall = np.mean(recall_scores_overall)

    # Add overall data to lists
    class_list.append('Overall')
    lower_f1_list.append(lower_f1_overall)
    upper_f1_list.append(upper_f1_overall)
    mean_f1_list.append(mean_f1_overall)
    lower_precision_list.append(lower_precision_overall)
    upper_precision_list.append(upper_precision_overall)
    mean_precision_list.append(mean_precision_overall)
    lower_recall_list.append(lower_recall_overall)
    upper_recall_list.append(upper_recall_overall)
    mean_recall_list.append(mean_recall_overall)
    lower_pap_list.append(None)
    upper_pap_list.append(None)
    mean_pap_list.append(None)
    # Add overall F1 scores to list
    f1_scores_list += f1_scores_overall
    class_scores_list += ['Overall'] * len(f1_scores_overall)

    df = pd.DataFrame({
        'class': class_list,
        'mean F1 score': mean_f1_list,
        'lower F1 CI': lower_f1_list,
        'upper F1 CI': upper_f1_list,
        'mean PAP score': mean_pap_list,
        'lower PAP CI': lower_pap_list,
        'upper PAP CI': upper_pap_list,
        'mean Precision score': mean_precision_list,  
        'lower Precision CI': lower_precision_list,
        'upper Precision CI': upper_precision_list,
        'mean Recall score': mean_recall_list,
        'lower Recall CI': lower_recall_list,
        'upper Recall CI': upper_recall_list
    })

    #print(df_f1_bootstrap.head())
    #print(df_pap_bootstrap.head())
    #print(df)

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
    
    return df
