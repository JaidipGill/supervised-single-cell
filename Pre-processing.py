# %%
import mudata as md
import scanpy as sc
import muon as mu
import anndata as ad

#Raw Data
#mdata = md.read("PBMC 10k multiomic/pbmc10k.h5mu")

#Original Data
#adata = ad.read_h5ad("Data/PBMC 10k multiomic/pbmc_raw.h5")
# Convert the AnnData object to MuData
mdata = mu.MuData({'rna': adata})
# Save as .h5mu
mdata.write('your_file.h5mu')
# %%
#EDA
print(f'''The number of (observations, features) for RNA and ATAC is 
{[ad.shape for ad in mdata.mod.values()]}''')
sc.pl.highest_expr_genes(mdata['rna'], n_top=20)
sc.pl.highest_expr_genes(mdata['atac'], n_top=20)

# %%
#EDA - RNA
mdata['rna'].var['mt'] = mdata['rna'].var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'

#Visualising RNA data
sc.pp.calculate_qc_metrics(mdata['rna'], qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
sc.pl.violin(mdata['rna'], ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             jitter=0.4, multi_panel=True)
sc.pl.scatter(mdata['rna'], x='total_counts', y='pct_counts_mt')
sc.pl.scatter(mdata['rna'], x='total_counts', y='n_genes_by_counts')
# %%
#Filtering - RNA
#Filtering out cells with less than 2500 genes
mu.pp.filter_obs(mdata['rna'], 'n_genes_by_counts', lambda x: (x >= 500) & (x < 5000))
#Filtering out cells with more than 15000 counts (e.g. doublets)
mu.pp.filter_obs(mdata['rna'], 'total_counts', lambda x: x < 65)
#Filtering out cells with more than 5% mitochondrial genes
mu.pp.filter_obs(mdata['rna'], 'pct_counts_mt', lambda x: x < 5)

#Visualising RNA data
sc.pp.calculate_qc_metrics(mdata['rna'], qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
sc.pl.violin(mdata['rna'], ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             jitter=0.4, multi_panel=True)
sc.pl.scatter(mdata['rna'], x='total_counts', y='pct_counts_mt')
sc.pl.scatter(mdata['rna'], x='total_counts', y='n_genes_by_counts')

# %%
