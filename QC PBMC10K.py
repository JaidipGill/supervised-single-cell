# %% ----------------------------------------------------------------
import mudata as md
import scanpy as sc
import muon as mu
import anndata as ad
import h5py as h5
import pandas as pd
from muon import atac as ac
'''
#CONVERTING MATRIX DIRECTORY TO MUDATA FORMAT
# Load the .mtx file
matrix = scipy.io.mmread('Data/PBMC 10k multiomic/filtered_feature_bc_matrix/matrix.mtx.gz')
# Load barcodes file
barcodes = pd.read_csv('Data/PBMC 10k multiomic/filtered_feature_bc_matrix/barcodes.tsv.gz', header=None, sep='\t')
# Load genes file
features = pd.read_csv('Data/PBMC 10k multiomic/filtered_feature_bc_matrix/features.tsv.gz', header=None, sep='\t')

# Split the features into genes and peaks based on the type column
genes = features[features[2] == 'Gene Expression']
peaks = features[features[2] == 'Peaks']

# Split the matrix into two based on the features
matrix = matrix.tocsr()  # Convert to CSR format for efficient row slicing
rna_matrix = matrix[genes.index, :]
atac_matrix = matrix[peaks.index, :]
# Transpose the matrices
rna_matrix = rna_matrix.transpose()
atac_matrix = atac_matrix.transpose()

# Create two separate AnnData objects
rna = ad.AnnData(X=rna_matrix, obs=barcodes, var=genes)
atac = ad.AnnData(X=atac_matrix, obs=barcodes, var=peaks)
# Combine the AnnData objects into a MuData object
mdata = mu.MuData({'rna': rna, 'atac': atac})
#mdata['atac'].var_names = [f"Peak_{i:d}" for i in range(mdata['atac'].n_vars)]
#mdata['rna'].var_names = [f"Gene_{i:d}" for i in range(mdata['rna'].n_vars)]
mdata['rna'].var_names = mdata['rna'].var[1]
mdata['atac'].var_names = mdata['atac'].var[1]
print(mdata.obs_names[:10])
'''
# %% ----------------------------------------------------------------
#LOADING DATA

def quality_control(input_file, output_file, lower_unique_peaks, upper_unique_peaks, lower_total_peaks, upper_total_peaks):
    '''
    Quality control of RNA and ATAC components of an mdata object
    '''
    mdata = mu.read_10x_h5(input_file)
    mdata.var_names_make_unique()

    #EDA - GENERAL
    print(f'''The number of (observations, features) for RNA and ATAC is 
    {[ad.shape for ad in mdata.mod.values()]}''')
    sc.pl.highest_expr_genes(mdata['rna'], n_top=20)
    sc.pl.highest_expr_genes(mdata['atac'], n_top=20)

    #EDA - RNA
    mdata['rna'].var['mt'] = mdata['rna'].var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'

    #Visualising RNA data
    sc.pp.calculate_qc_metrics(mdata['rna'], qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    sc.pl.violin(mdata['rna'], ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
                jitter=0.4, multi_panel=True)
    sc.pl.scatter(mdata['rna'], x='total_counts', y='pct_counts_mt')
    sc.pl.scatter(mdata['rna'], x='total_counts', y='n_genes_by_counts')

    #FILTERING - RNA
    #Filtering out cells 
    print(mdata['rna'].n_obs)
    mu.pp.filter_obs(mdata['rna'], 'n_genes_by_counts', lambda x: (x >= 500) & (x < 5000))
    print(mdata['rna'].n_obs)
    #Filtering out cells with more than 15000 counts (e.g. doublets)
    mu.pp.filter_obs(mdata['rna'], 'total_counts', lambda x: x < 15000)
    print(mdata['rna'].n_obs)
    #Filtering out cells with more than 20% mitochondrial genes
    mu.pp.filter_obs(mdata['rna'], 'pct_counts_mt', lambda x: x < 20)
    print(mdata['rna'].n_obs)

    #Visualising effects of RNA filtering
    sc.pp.calculate_qc_metrics(mdata['rna'], qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    sc.pl.violin(mdata['rna'], ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
                jitter=0.4, multi_panel=True)
    sc.pl.scatter(mdata['rna'], x='total_counts', y='pct_counts_mt')
    sc.pl.scatter(mdata['rna'], x='total_counts', y='n_genes_by_counts')

    # EDA - ATAC
    sc.pp.calculate_qc_metrics(mdata['atac'], percent_top=None, log1p=False, inplace=True)
    sc.pl.violin(mdata['atac'], ['total_counts', 'n_genes_by_counts'], jitter=0.4, multi_panel=True)
    mu.pl.histogram(mdata['atac'], ['n_genes_by_counts', 'total_counts'])

    #FILTERING - ATAC
    print(mdata['atac'].n_obs)
    # Filtering out cells with extreme numbers of unique peaks
    mu.pp.filter_obs(mdata['atac'], 'n_genes_by_counts', lambda x: (x >= lower_unique_peaks) & (x <= upper_unique_peaks))
    print(mdata['atac'].n_obs)
    # Filtering out cells with extreme values of total peaks
    mu.pp.filter_obs(mdata['atac'], 'total_counts', lambda x: (x >= lower_total_peaks) & (x <= upper_total_peaks))
    print(mdata['atac'].n_obs)

    #Effects of filtering
    sc.pl.violin(mdata['atac'], ['n_genes_by_counts', 'total_counts'], jitter=0.4, multi_panel=True)

    # Retaining cells that passed QC for RNA and ATAC
    mu.pp.intersect_obs(mdata)
    mdata.write(output_file)
# %% ----------------------------------------------------------------
# QC PBMC10K

quality_control(input_file = "Data/PBMC 10k multiomic/pbmc_filtered.h5", output_file = "Data/PBMC 10k multiomic/QC-pbmc10k.h5mu", upper_unique_peaks = 15000, lower_unique_peaks=2000, lower_total_peaks=4000, upper_total_peaks=40000)
# %% ----------------------------------------------------------------
# QC B CELL LYMPHOMA

quality_control(input_file = "Data\B cell lymphoma\lymph_node_lymphoma_14k_filtered_feature_bc_matrix.h5", output_file = "Data/B cell lymphoma/QC-bcell.h5mu", upper_unique_peaks = 25000, lower_unique_peaks=500, lower_total_peaks=1500, upper_total_peaks=50000)
# %%
