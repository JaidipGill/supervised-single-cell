rm(list=ls())

library(lme4)
library(sgPLS)
library(utils)
library(pheatmap)
library(RColorBrewer)
library(rhdf5)
library(mixOmics)
library(Matrix)

# Load the file
h5file <- "Data/PBMC 10k multiomic/QC-pbmc10k.h5mu"
h5ls(h5file)
hdf5_data <- h5read(h5file, "/")

# Loading Data ------------------------------------------------------------

# Read the components of the sparse matrix
data <- h5read(h5file, "/mod/rna/X/data")
indices <- h5read(h5file, "/mod/rna/X/indices")
indptr <- h5read(h5file, "/mod/rna/X/indptr")
# Convert data, indices, and indptr from the h5mu object to R's format
n_cols <- length(indptr) - 1
n_rows <- max(indices) + 1

# Generate the column (j) indices
j <- rep(1:n_cols, times = diff(indptr))

# Increment indices to account for R's 1-based indexing
i <- indices + 1

# Construct the matrix
sparse_matrix <- sparseMatrix(i = i, j = j, x = data, dims = c(n_rows, n_cols))

# Check the matrix
head(sparse_matrix)

rna_features <- h5read(h5file, "/mod/rna/var/gene_ids")
splsda.result <- splsda(rna_data, rna_features, keepX = c(50,30)) # run the method
plotIndiv(splsda.result) # plot the samples
plotVar(splsda.result) # plot the variables