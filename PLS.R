rm(list=ls())

library(lme4)
library(sgPLS)
library(utils)
library(pheatmap)
library(RColorBrewer)
library(rhdf5)

# Load the file
h5file <- "Data/PBMC 10k multiomic/QC-pbmc10k.h5mu"
h5ls(h5file)
hdf5_data <- h5read(h5file, "/")

# Loading Data ------------------------------------------------------------

rna_data <- h5read(h5file, "/mod/rna/X")
