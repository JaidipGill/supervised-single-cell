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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from xgboost import XGBClassifier
import time
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, make_scorer
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import pickle
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

def pre_process(adata):

    '''
    Pre-process MuData object for either RNA or ATAC modality
    '''

    # Filter out low-frequency features
    mu.pp.filter_var(adata, 'n_cells_by_counts', lambda x: x >= 10) 

    # Saving raw counts
    adata.layers["counts"] = adata.X # Store unscaled counts in .layers["counts"] attribute

    # Normalizing peaks/genes
    print(f"Total peaks/genes in random cell: {adata.X[1,:].sum()}")
    sc.pp.normalize_total(adata, target_sum=1e4) # Normalize counts per cell
    print(f"Total peaks/genes in random cell: {adata.X[1,:].sum()}")
    sc.pp.log1p(adata) # Logarithmize + 1
    print(f"Total peaks/genes in random cell: {adata.X[1,:].sum()}") # Sanity check for normalization - should be 1e4

    # Filtering features
    sc.pp.highly_variable_genes(adata) # Select highly variable genes
    sc.pl.highly_variable_genes(adata) # Plot highly variable genes
    print(f"Number of highly variable peaks/genes: {np.sum(adata.var.highly_variable)}")
    print(f"Number of peaks/genes before filtering: {adata.n_vars}")
    #adata = adata[:, adata.var.highly_variable] # Filter out non-highly variable genes
    print(f"Number of peaks/genes after filtering: {adata.n_vars}")

    # Scaling
    adata.raw = adata # Store unscaled counts in .raw attribute
    sc.pp.scale(adata) # Scale to unit variance and shift to zero mean
    print(f"Min Scaled peaks/genes value:{adata.X.min()}") 
    print(f"Min unscaled peaks/genes value: {adata.raw.X.min()}") # Sanity check for scaling - should not be negative

    return adata

def model_test_main(model,x_train,y_train,x_test,y_test, subset, table_one_only=False):
    '''
    Function to test a model on a train and test set
    '''
    if subset == True:
        x_train = x_train.iloc[:500,:]
        y_train = y_train[:500]
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
                    'min_samples_leaf'   : np.array([1, 2, 5]),
                    'max_features' : np.array(['sqrt']),
                    'max_depth':np.array([2**2, 2**3, 2**4]),
                    'min_samples_split': np.array([3, 5, 10])}
    elif isinstance(model, svm.SVC):
        param_grid={'C':[2**0, 2**1, 2**2, 2**3, 2**4, 2**5],
                    'kernel':['poly','rbf', 'sigmoid']}
    inner=ShuffleSplit(n_splits=1,test_size=0.3,random_state=0)

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
    #Fitting model with optimised parameters to training data
    model.fit(x_train,y_train)

    #Predicting using fitted model on train set
    y_pred_train = model.predict(x_train)
    #Predicting using fitted model on test set
    y_pred_test = model.predict(x_test)

    #Get results for train and test sets
    for predictions, observations, features in zip([y_pred_train, y_pred_test],[y_train, y_test],[x_train,x_test]):
        #De-encoding labels for xgb
        if isinstance(model,XGBClassifier):
            predictions = encoder.inverse_transform(predictions)
            observations = encoder.inverse_transform(observations)
        #Results Visualisation
        print(f'{predictions} Set Results:')
        print(classification_report(observations,predictions))
        print(f"AUC: {roc_auc_score(observations, model.predict_proba(features), multi_class='ovr')}")
        plt.rcParams.update({'font.size': 8}) 
        cmatrix=ConfusionMatrixDisplay.from_predictions(observations, predictions, xticks_rotation='vertical')
        plt.show(cmatrix)
    time_taken=(time.process_time() - start_time)
    print(f'CPU time for training and testing: {time_taken} seconds or {time_taken/60} mins or {time_taken/(60*60)} hrs')
    return(model, y_pred_test)

def save_model(model_cl, location, y_pred_test, y_test):
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
    df.to_pickle(f'Supervised Models\{location}.pickle')