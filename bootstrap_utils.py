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

def add_annon(mdata, subset, data):
    '''
    Adds annotations to mdata_train and mdata_test based on WNN clustering
    WNN = 0: RNA labels, WNN = 1: WNN Level 1 labels, WNN = 2: WNN Level 2 labels
    '''
    if subset==True:
        mdata = mdata[:1000]

    if data == 'cancer':
        annotations_rna = pd.read_csv('Data/B cell lymphoma/Cell Types.csv', header=0, index_col=0)
        ann_list= [annotations_rna]
    elif data == 'pbmc':
        annotations_rna = pd.read_csv('Data/PBMC 10k multiomic/PBMC-10K-celltype.txt', sep='\t', header=0, index_col=0)
        annotations_wnn1 = pd.read_csv('Data/PBMC 10k multiomic/WNNL1-PBMC-10K-celltype.csv', sep='\t', header=0, index_col='index')
        annotations_wnn2 = pd.read_csv('Data/PBMC 10k multiomic/WNNL2-PBMC-10K-celltype.csv', sep='\t', header=0, index_col='index')
        ann_list= [annotations_rna, annotations_wnn1, annotations_wnn2]
    for idx, annotations in enumerate(ann_list):
        # Take intersection of cell barcodes in annotations and mdata
        #print(annotations)
        common_barcodes = annotations.index.intersection(mdata.obs_names)
        #print(annotations.index)
        # Filter annotations and mdata to keep only common barcodes
        annotations = annotations.loc[common_barcodes]
        #print(annotations)
        # Add the annotations to the .obs DataFrame
        mdata.obs = pd.concat([mdata.obs, annotations], axis=1)
        print(mdata.obs.columns)
        mdata.obs.rename(columns={'x': f'cell_type{idx}'}, inplace=True)
        print(mdata.obs[f'cell_type{idx}'])
        mdata.mod['rna'].obs[f'cell_type{idx}'] = mdata.obs[f'cell_type{idx}']
        mdata.mod['atac'].obs[f'cell_type{idx}'] = mdata.obs[f'cell_type{idx}']

        # Count number of NAs in cell_type column
        print(f"Cell_type {idx} NAs: {mdata.obs[f'cell_type{idx}'].isna().sum()}")
    return mdata

def model_test_main(model,x_train,y_train,x_test,y_test, subset, classes, f1_scores_per_class, f1_scores_overall, pap_scores_per_class):
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
        param_grid={'C': [0.01, 0.1, 1, 2],
                    'kernel':['poly'], # RBF only for scVI
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
    #if subset == False:
    #    LearningCurveDisplay.from_estimator(model, x_train, y_train, random_state=42, 
    #                                        score_type = 'both', n_jobs = -1, scoring = f1, verbose = 4)
    #    plt.show()
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
        print(f"AUC: {roc_auc_score(observations, proba, multi_class='ovr')}")
        plt.rcParams.update({'font.size': 8})
        cmatrix = ConfusionMatrixDisplay.from_predictions(observations, predictions, xticks_rotation='vertical')
        plt.show(cmatrix)

    # Metric calculations

    # Compute F1 scores for each class
    f1_scores = f1_score(y_test, y_pred_test, average=None)

    # Compute overall F1 score
    f1_score_overall = f1_score(y_test, y_pred_test, average='macro')
    f1_scores_overall.append(f1_score_overall)

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

    # Append F1 scores for each class
    for class_, f1_score_ in zip(classes, f1_scores):
        f1_scores_per_class[class_].append(f1_score_)
    '''
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
    start_time = time.process_time()
    #metrics, f1_df, pap_df = bootstrap_confidence_interval(model, x_test, y_test)
    time_taken=(time.process_time() - start_time)
    #print(f"95% confidence interval for F1 score: ({metrics[metrics['class'] == 'Overall']['lower F1 CI'].values[0]:.3f}, {metrics[metrics['class'] == 'Overall']['upper F1 CI'].values[0]:.3f}, mean: {metrics[metrics['class'] == 'Overall']['mean F1 score'].values[0]:.3f})")
    #print(f'CPU time for boostrap: {time_taken} seconds or {time_taken/60} mins or {time_taken/(60*60)} hrs')
    print(f1_scores_per_class)

    return(model, y_pred_test, pap_scores_per_class, f1_scores_per_class, f1_scores_overall)

def analyse_metrics(f1_scores_per_class, pap_scores_per_class, f1_scores_overall, suffix, rna):
    '''
    Analyse metrics from bootstrapping
    - Derives confidence intervals of F1 and PAP scores for each class using
      percentile method
    - Plots histograms of F1 and PAP scores for each class
    '''
    classes = list(f1_scores_per_class.keys())
    for class_, scores in f1_scores_per_class.items():
        print(f"Class: {class_}, number of scores: {len(scores)}")

    # Generate dataframes of bootstrap distributions
    df_f1_bootstrap = pd.DataFrame.from_dict(f1_scores_per_class)
    df_pap_bootstrap = pd.DataFrame.from_dict(pap_scores_per_class)
    if rna == True:
        df_f1_bootstrap.to_csv(f'Supervised Models/Results_{suffix}_F1_df_rna.csv')
        df_pap_bootstrap.to_csv(f'Supervised Models/Results_{suffix}_PAP_df_rna.csv')
    else:
        df_f1_bootstrap.to_csv(f'Supervised Models/Results_{suffix}_F1_df.csv')
        df_pap_bootstrap.to_csv(f'Supervised Models/Results_{suffix}_PAP_df.csv')

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
        print(pap_scores_per_class[class_])
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

    return df
