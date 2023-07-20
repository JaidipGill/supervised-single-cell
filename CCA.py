import muon as mu
import numpy as np
from sklearn.model_selection import train_test_split
import time
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt

# LOAD PROCESSED DATA

mdata_train = mu.read_h5mu("Data/PBMC 10k multiomic/processed_data/pre-proc_mdata_train.h5mu")
mdata_test = mu.read_h5mu("Data/PBMC 10k multiomic/processed_data/pre-proc_mdata_test.h5mu")

def perform_cca(mdata_train, mdata_test, n_components=50):
    '''
    Performs CCA on the data and adds the CCA components to the mdata object.
    Also generates scree plot of correlation between each CCA component.
    '''
    # Perform CCA
    cca = CCA(n_components=n_components, scale=False)
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

# DIMENSIONALITY REDUCTION - CCA

sample_train = mu.pp.sample_obs(mdata_train,0.1)[:,0:30000]
sample_test = mu.pp.sample_obs(mdata_test,0.1)[:,0:30000]

sample_train, sample_test = perform_cca(sample_train, sample_test, n_components=50)

# SAVE PROCESSED DATA

mdata_train.write_h5mu("CCA_mdata_train.h5mu")
mdata_test.write_h5mu("CCA_mdata_test.h5mu")

