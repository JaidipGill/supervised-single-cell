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

def perform_pca(mdata_train, mdata_test, raw=False, components=20, random_state=42):
    '''
    Perform PCA on RNA and ATAC modalities of given mdata_train and mdata_test
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
        num_pcs = int(input(f"Enter the number of PCs you want to keep for {mod}: "))
        print(f"PCA {mod} with {num_pcs} PCs explains {np.cumsum(pca[mod].explained_variance_ratio_[0:num_pcs])[-1]*100}% of variance")

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

def bootstrap_confidence_interval(model, X, y, n_bootstrap=1000):
    """
    Function to estimate a 95% confidence interval for a model's F1 score using bootstrapping.
    """
    f1_scores = []
    for _ in range(n_bootstrap):
        X_resample, y_resample = resample(X, y, n_samples=len(X) // 2)
        y_pred = model.predict(X_resample)
        f1_scores.append(f1_score(y_resample, y_pred, average='macro'))
    lower = np.percentile(f1_scores, 2.5)
    upper = np.percentile(f1_scores, 97.5)
    mean = np.mean(f1_scores)
    median = np.median(f1_scores)
    # Plot histogram of F1 scores
    plt.hist(f1_scores)
    plt.title("Histogram of F1 scores")
    plt.xlabel("F1 score")
    plt.ylabel("Frequency")
    plt.show()
    return lower, upper, mean, median

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
    elif isinstance(model, LogisticRegression):
        model=LogisticRegression(**optimal_params, random_state=42, class_weight='balanced')
    LearningCurveDisplay.from_estimator(model, x_train, y_train, random_state=42)
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
    # Create a list to hold the PAC scores
    pac_list = []
    for i, (predictions, observations, features) in enumerate(zip([y_pred_train, y_pred_test], [y_train, y_test], [x_train, x_test])):
        # De-encoding labels for xgb
        if isinstance(model, XGBClassifier):
            predictions = encoder.inverse_transform(predictions)
            observations = encoder.inverse_transform(observations)
        # Results Visualisation
        print(f'{predictions} Set Results:')
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
            pac_score = cdf_x2 - cdf_x1
            # Store the PAP score in the list
            pac_list.append({'Class': f'{model.classes_[j]}', 'Set': ["Train", "Test"][i], 'PAP': pac_score})
            df_list.append(df)
        

    # Convert the list of dictionaries to a DataFrame
    pac_df = pd.DataFrame(pac_list)
    # Concatenate all DataFrames
    df_total = pd.concat(df_list, ignore_index=True)

    # Plot the ECDFs
    fig = px.ecdf(df_total, x="Probability", color="Class", facet_row="Set")
    fig.show()
    print(pac_df)

    # Estimate confidence interval for F1 score
    start_time = time.process_time()
    lower, upper, mean, median = bootstrap_confidence_interval(model, x_test, y_test)
    time_taken=(time.process_time() - start_time)
    print(f"95% confidence interval for F1 score: ({lower:.3f}, {upper:.3f}, mean: {mean:.3f}, median: {median:.3f})")
    print(f'CPU time for boostrap: {time_taken} seconds or {time_taken/60} mins or {time_taken/(60*60)} hrs')

    return(model, y_pred_test, df_list, pac_df)

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
    df.to_pickle(f'Supervised Models\{location}_Predictions.pickle')


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
