# Load necessary package
library(tidyverse)
library(ggplot2)
library(glue)
library(ggpubr)
library(dplyr)
library(stringr)
library(RColorBrewer)
library(ggh4x)
library(purrr)
library(dunn.test)

# Interpretation Modelling ------------------------------------------------

data <- read.csv("Data/PBMC 10k multiomic/Interpretation/X_y.csv")
predictors <- names(data)[1:70]
formula_string <- paste("X2 ~", paste(predictors, collapse = " + "))
formula_obj <- as.formula(formula_string)
model <- glm(formula_obj, data = data, family = "binomial")
summary(model)


# GROUND TRUTH (FIGURE 4) -------------------------------------------------

EMBEDDINGS = 'scVI'
CLASSIFIERS = 'SVC'
DATA = 'pbmc'
CELL_TYPE = 'All'
N_COMPONENTS = 35

df_combined <- data.frame()

for (GROUND_TRUTH in c('wnnL2','wnnL1')) {  
  SUFFIX = glue('{DATA}_{CLASSIFIERS}_{EMBEDDINGS}_{GROUND_TRUTH}_{CELL_TYPE}_{N_COMPONENTS}')
  
  # Read the CSV files
  df_A_metric1 <- read_csv(glue("Supervised Models/Macro Metrics/Results_{SUFFIX}_F1_overall_df.csv"))
  df_A_metric2 <- read_csv(glue("Supervised Models/Macro Metrics/Results_{SUFFIX}_Precision_overall_df.csv"))
  df_A_metric3 <- read_csv(glue("Supervised Models/Macro Metrics/Results_{SUFFIX}_Recall_overall_df.csv"))
  
  df_B_metric1 <- read_csv(glue("Supervised Models/Macro Metrics/Results_{SUFFIX}_F1_overall_df_rna.csv"))
  df_B_metric2 <- read_csv(glue("Supervised Models/Macro Metrics/Results_{SUFFIX}_Precision_overall_df_rna.csv"))
  df_B_metric3 <- read_csv(glue("Supervised Models/Macro Metrics/Results_{SUFFIX}_Recall_overall_df_rna.csv"))
  
  # Add identifying columns
  df_A_metric1$Model <- 'RNA + ATAC'
  df_A_metric1$Metric <- 'F1 Scores'
  df_A_metric2$Model <- 'RNA + ATAC'
  df_A_metric2$Metric <- 'Precision'
  df_A_metric3$Model <- 'RNA + ATAC'
  df_A_metric3$Metric <- 'Recall'
  
  df_B_metric1$Model <- 'RNA'
  df_B_metric1$Metric <- 'F1 Scores'
  df_B_metric2$Model <- 'RNA'
  df_B_metric2$Metric <- 'Precision'
  df_B_metric3$Model <- 'RNA'
  df_B_metric3$Metric <- 'Recall'
  
  # Combine datasets
  df_all <- bind_rows(df_A_metric1, df_A_metric2, df_A_metric3, df_B_metric1, df_B_metric2, df_B_metric3) %>%
    mutate(GroundTruth = GROUND_TRUTH)
  df_all <- subset(df_all, select = -c(...1))
  
  df_combined <- bind_rows(df_combined, df_all)
}

# Pivot the data
df_long <- df_combined %>%
  pivot_longer(cols = -c(Model, Metric, GroundTruth),
               names_to = "Metric Value",
               values_to = "Value")

# Plot the data
p <- ggplot(df_long, aes(x = GroundTruth, y = Value, color = Model)) +
  geom_boxplot(position = "dodge") +
  facet_wrap(~ Metric) +
  labs(x = "Ground Truth", y = "Value", fill = "Model") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_color_brewer(palette="Set1")

print(p)

# Generate summary dataframe
summary_df <- df_long %>%
  group_by(Model, Metric, GroundTruth) %>%
  summarise(
    Median = median(Value, na.rm = TRUE),
    Lower_Quartile = quantile(Value, 0.25, na.rm = TRUE),
    Upper_Quartile = quantile(Value, 0.75, na.rm = TRUE)
  ) %>%
  ungroup()

# Print the summary dataframe
print(summary_df)

# Modified function to perform Wilcoxon tests for a given subset of data
perform_test_groundtruth <- function(sub_data) {
  group1 <- sub_data[sub_data$Model == "RNA", ]$Value
  group2 <- sub_data[sub_data$Model == "RNA + ATAC", ]$Value
  
  # Check if both groups have more than one data point
  if (length(group1) <= 1 | length(group2) <= 1) {
    return(list(statistic = NA, p.value = NA))
  }
  
  test_result <- wilcox.test(group1, group2, paired = TRUE)
  return(list(statistic = test_result$statistic, p.value = test_result$p.value))
}

# Group by Ground Truth and Metric, then apply the Wilcoxon test
results_groundtruth <- df_long %>%
  group_by(GroundTruth, Metric) %>%
  do(test = perform_test_groundtruth(.))

# Extract p-values
results_groundtruth$p_value <- sapply(results_groundtruth$test, function(x) x$p.value)

# Adjust p-values for multiple testing
results_groundtruth$adjusted_p_value <- p.adjust(results_groundtruth$p_value, method = "bonferroni")

# Add a column to the results dataframe to show the number of asterisks for significance level
results_groundtruth$Significance <- case_when(
  results_groundtruth$adjusted_p_value < 0.001 ~ '***',
  results_groundtruth$adjusted_p_value < 0.01  ~ '**',
  results_groundtruth$adjusted_p_value < 0.05  ~ '*',
  TRUE                                          ~ ''
)

# Print the results
print(results_groundtruth[, c("GroundTruth", "Metric", "p_value", "adjusted_p_value", "Significance")])


# CELL-TYPES (FIGURE 3) -------------------------------------------------------


EMBEDDINGS = 'scVI'
CLASSIFIERS = 'SVC'
GROUND_TRUTHS = 'wnnL2'
DATA = 'pbmc'
CELL_TYPE = 'All'
N_COMPONENTS = 35

df_combined <- data.frame()

for (GROUND_TRUTH in GROUND_TRUTHS) {
  for (EMBEDDING in EMBEDDINGS) {
    for (CL in CLASSIFIERS) {
      SUFFIX = glue('{DATA}_{CL}_{EMBEDDING}_{GROUND_TRUTH}_{CELL_TYPE}_{N_COMPONENTS}')
      
      # Read the CSV files
      df_A_metric1 <- read_csv(glue("Supervised Models/Results_{SUFFIX}_F1_df.csv"))
      df_A_metric2 <- read_csv(glue("Supervised Models/Results_{SUFFIX}_Precision_class_df.csv"))
      df_A_metric3 <- read_csv(glue("Supervised Models/Results_{SUFFIX}_Recall_class_df.csv"))
      
      df_B_metric1 <- read_csv(glue("Supervised Models/Results_{SUFFIX}_F1_df_rna.csv"))
      df_B_metric2 <- read_csv(glue("Supervised Models/Results_{SUFFIX}_Precision_class_df_rna.csv"))
      df_B_metric3 <- read_csv(glue("Supervised Models/Results_{SUFFIX}_Recall_class_df_rna.csv"))
      # Add identifying columns
      df_A_metric1$Model <- 'RNA + ATAC'
      df_A_metric1$Metric <- 'F1 Scores'
      df_A_metric2$Model <- 'RNA + ATAC'
      df_A_metric2$Metric <- 'Precision'
      df_A_metric3$Model <- 'RNA + ATAC'
      df_A_metric3$Metric <- 'Recall'
      
      df_B_metric1$Model <- 'RNA'
      df_B_metric1$Metric <- 'F1 Scores'
      df_B_metric2$Model <- 'RNA'
      df_B_metric2$Metric <- 'Precision'
      df_B_metric3$Model <- 'RNA'
      df_B_metric3$Metric <- 'Recall'
      
      # Read the PAP CSV files
      df_A_PAP <- read_csv(glue("Supervised Models/Results_{SUFFIX}_PAP_df.csv"))
      df_B_PAP <- read_csv(glue("Supervised Models/Results_{SUFFIX}_PAP_df_rna.csv"))
      
      # Add identifying columns for PAP
      df_A_PAP$Model <- 'RNA + ATAC'
      df_A_PAP$Metric <- 'PAP'
      df_B_PAP$Model <- 'RNA'
      df_B_PAP$Metric <- 'PAP'
      
      # Combine datasets including PAP
      df_all <- bind_rows(df_A_metric1, df_A_metric2, df_A_metric3, df_B_metric1, df_B_metric2, df_B_metric3, df_A_PAP, df_B_PAP) %>%
        mutate(Combination = paste(EMBEDDING, CL, sep = "_"))
      df_all <- df_all %>% select(-...1)
      df_all <- df_all %>%
        mutate(Combination = str_replace_all(Combination, "_", " "))
      df_all <- df_all %>%
        mutate(Combination = str_replace_all(Combination, "RandomForestClassifier", "RF"))
      df_all <- df_all %>%
        mutate(Combination = str_replace_all(Combination, "LogisticRegression", "LR"))
      
      df_combined <- bind_rows(df_combined, df_all)
    }
  }
}

# Pivot the data
df_long <- df_combined %>%
  pivot_longer(cols = c(-Model, -Metric, -Combination),
               names_to = "Cell Type",
               values_to = "Value")

# Plot the data excluding PAP
df_long_plot <- df_long %>% filter(Metric != "PAP")

p <- ggplot(df_long_plot, aes(x = `Cell Type`, y = Value, color = Model)) +
  geom_boxplot(position = "dodge") +
  facet_wrap(~ Metric) +
  labs(x = NULL, y = "Value", fill = "Model") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_color_brewer(palette="Set1")

print(p)


# Generate summary dataframe
summary_df <- df_long %>%
  group_by(Model, Metric, `Cell Type`) %>%
  summarise(
    Median = median(Value, na.rm = TRUE),
    Lower_Quartile = quantile(Value, 0.25, na.rm = TRUE),
    Upper_Quartile = quantile(Value, 0.75, na.rm = TRUE)
  ) %>%
  ungroup()

# Print the summary dataframe
print(summary_df)

# Modified function to perform Wilcoxon tests for a given subset of data
perform_test_celltype <- function(sub_data) {
  group1 <- sub_data[sub_data$Model == "RNA", ]$Value
  group2 <- sub_data[sub_data$Model == "RNA + ATAC", ]$Value
  
  # Check if both groups have more than one data point
  if (length(group1) <= 1 | length(group2) <= 1) {
    return(list(statistic = NA, p.value = NA))
  }
  
  test_result <- wilcox.test(group1, group2, paired = TRUE)
  return(list(statistic = test_result$statistic, p.value = test_result$p.value))
}

# Group by Cell Type and Metric, then apply the Wilcoxon test
results_celltype <- df_long %>%
  group_by(`Cell Type`, Metric) %>%
  do(test = perform_test_celltype(.))

# Extract p-values
results_celltype$p_value <- sapply(results_celltype$test, function(x) x$p.value)

# Adjust p-values for multiple testing
results_celltype$adjusted_p_value <- p.adjust(results_celltype$p_value, method = "BH")

# Add a column to the results dataframe to show the number of asterisks for significance level
results_celltype$Significance <- case_when(
  results_celltype$adjusted_p_value < 0.001 ~ '***',
  results_celltype$adjusted_p_value < 0.01  ~ '**',
  results_celltype$adjusted_p_value < 0.05  ~ '*',
  TRUE                                       ~ ''
)

# Print the results
print(results_celltype[, c("Cell Type", "Metric", "p_value", "adjusted_p_value", "Significance")])


# PLOT PAP SCORES (FIGURE 3) --------------------------------------------------------------

df_combined <- data.frame()

for (GROUND_TRUTH in GROUND_TRUTHS) {
  for (EMBEDDING in EMBEDDINGS) {
    for (CL in CLASSIFIERS) {
      SUFFIX = glue('{DATA}_{CL}_{EMBEDDING}_{GROUND_TRUTH}_{CELL_TYPE}_{N_COMPONENTS}')
      
      # Read the CSV files
      df_A <- read_csv(glue("Supervised Models/Results_{SUFFIX}_PAP_df.csv"))
      df_B <- read_csv(glue("Supervised Models/Results_{SUFFIX}_PAP_df_rna.csv"))
      
      # Add identifying columns
      df_A$Model <- 'RNA + ATAC'
      df_A$Metric <- 'PAP'
      
      df_B$Model <- 'RNA'
      df_B$Metric <- 'PAP'
      
      # Combine datasets
      df_all <- bind_rows(df_A, df_B) %>%
        mutate(Combination = paste(EMBEDDING, CL, sep = "_"))
      df_all <- df_all %>% select(-...1)
      df_all <- df_all %>%
        mutate(Combination = str_replace_all(Combination, "_", " "))
      df_all <- df_all %>%
        mutate(Combination = str_replace_all(Combination, "RandomForestClassifier", "RF"))
      df_all <- df_all %>%
        mutate(Combination = str_replace_all(Combination, "LogisticRegression", "LR"))
      
      df_combined <- bind_rows(df_combined, df_all)
    }
  }
}

# Pivot the data
df_long <- df_combined %>%
  pivot_longer(cols = c(-Model, -Metric, -Combination),
               names_to = "Cell Type",
               values_to = "Value")

# Plot the data
p <- ggplot(df_long, aes(x = `Cell Type`, y = Value, color = Model)) +
  geom_boxplot(position = "dodge") +
  labs(x = "Cell Type", y = "PAP Score", fill = "Model") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_color_brewer(palette="Set1") +
  theme(legend.position = "none")

print(p)

# ALGORITHMS (FIGURE 2) --------------------------------------------


EMBEDDINGS = c('PCA', 'scVI')
GROUND_TRUTHS = c('wnnL2')
CLASSIFIERS = c('RandomForestClassifier', 'SVC', 'LogisticRegression')
DATA = 'pbmc'
CELL_TYPE = 'All'
N_COMPONENTS = 35

df_combined <- data.frame()

for (GROUND_TRUTH in GROUND_TRUTHS) {
  for (EMBEDDING in EMBEDDINGS) {
    for (CL in CLASSIFIERS) {
      SUFFIX = glue('{DATA}_{CL}_{EMBEDDING}_{GROUND_TRUTH}_{CELL_TYPE}_{N_COMPONENTS}')
      
      # Read the CSV files
      df_A_metric1 <- read_csv(glue("Supervised Models/Macro Metrics/Results_{SUFFIX}_F1_overall_df.csv"))
      df_A_metric2 <- read_csv(glue("Supervised Models/Macro Metrics/Results_{SUFFIX}_Precision_overall_df.csv"))
      df_A_metric3 <- read_csv(glue("Supervised Models/Macro Metrics/Results_{SUFFIX}_Recall_overall_df.csv"))
      
      df_B_metric1 <- read_csv(glue("Supervised Models/Macro Metrics/Results_{SUFFIX}_F1_overall_df_rna.csv"))
      df_B_metric2 <- read_csv(glue("Supervised Models/Macro Metrics/Results_{SUFFIX}_Precision_overall_df_rna.csv"))
      df_B_metric3 <- read_csv(glue("Supervised Models/Macro Metrics/Results_{SUFFIX}_Recall_overall_df_rna.csv"))
      
      # Add identifying columns
      df_A_metric1$Model <- 'RNA + ATAC'
      df_A_metric1$Metric <- 'F1 Scores'
      df_A_metric2$Model <- 'RNA + ATAC'
      df_A_metric2$Metric <- 'Precision'
      df_A_metric3$Model <- 'RNA + ATAC'
      df_A_metric3$Metric <- 'Recall'
      
      df_B_metric1$Model <- 'RNA'
      df_B_metric1$Metric <- 'F1 Scores'
      df_B_metric2$Model <- 'RNA'
      df_B_metric2$Metric <- 'Precision'
      df_B_metric3$Model <- 'RNA'
      df_B_metric3$Metric <- 'Recall'
      
      # Combine datasets
      # Combine datasets
      df_all <- bind_rows(df_A_metric1, df_A_metric2, df_A_metric3, df_B_metric1, df_B_metric2, df_B_metric3) %>%
        mutate(Combination = paste(EMBEDDING, CL, sep = "_"))
      
      # Correct extraction of Embedding and Classifier from Combination
      df_all$Embedding <- gsub("_.*", "", df_all$Combination)
      df_all$Classifier <- gsub(".*_", "", df_all$Combination)
      df_all$Classifier <- str_replace_all(df_all$Classifier, c(
        "RandomForestClassifier" = "RF",
        "LogisticRegression" = "LR"
      ))
      
      df_combined <- bind_rows(df_combined, df_all)
    }
  }
}

# Boxplot with nested facet
#df_combined$Embedding <- gsub("\n.*", "", df_combined$Combination)
#df_combined$Classifier <- gsub(".*\n", "", df_combined$Combination)
# Nested facet plot using facet_nested from ggh4x
p <- ggplot(df_combined, aes(x = Classifier, y = `0`, color = Model)) +
  geom_boxplot(position = "dodge") +
  facet_nested(.~Embedding + Metric, scales="free") + 
  labs(x = "Classifier", y = "Score", fill = "Model") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_color_brewer(palette="Set1")

print(p)

# REPEATED WILCOXON
# Function to perform Wilcoxon tests for given subset of data
perform_test <- function(sub_data) {
  group1 <- sub_data[sub_data$Model == "RNA", ]$`0`
  group2 <- sub_data[sub_data$Model == "RNA + ATAC", ]$`0`
  
  # Check if both groups have more than one data point
  if (length(group1) <= 1 | length(group2) <= 1) {
    return(list(statistic = NA, p.value = NA))
  }
  
  test_result <- wilcox.test(group1, group2, paired = TRUE)
  return(list(statistic = test_result$statistic, p.value = test_result$p.value))
}

# Group by Classifier, Embedding, and Metric, then apply the Wilcoxon test
results <- df_combined %>%
  group_by(Classifier, Embedding, Metric) %>%
  do(test = perform_test(.))

# Extract p-values
results$p_value <- sapply(results$test, function(x) x$p.value)

# Adjust p-values for multiple testing
results$adjusted_p_value <- p.adjust(results$p_value, method = "bonferroni")
# Add a column to the results dataframe to show the number of asterisks
results$Significance <- case_when(
  results$adjusted_p_value < 0.001 ~ '***',
  results$adjusted_p_value < 0.01  ~ '**',
  results$adjusted_p_value < 0.05  ~ '*',
  TRUE                           ~ ''
)

print(results[, c("Classifier", "Embedding", "Metric", "p_value", "adjusted_p_value")])

# Generating the summary dataframe
summary_df <- df_combined %>%
  group_by(Model, Metric, Combination) %>%
  summarise(
    Median = median(`0`, na.rm = TRUE),
    Lower_Quartile = quantile(`0`, 0.25, na.rm = TRUE),
    Upper_Quartile = quantile(`0`, 0.75, na.rm = TRUE)
  )
print(summary_df)

# SILHOUETTE SCORES (FIGURE 1) -------------------------------------------------------

# Load Data
pca <- read.csv('Data/PBMC 10k multiomic/Silhouette Scores/silhouette_scores_PCA.csv')
scvi <- read.csv('Data/PBMC 10k multiomic/Silhouette Scores/silhouette_scores_scVI.csv')

# Add an embedding column to each dataframe
pca$Embedding <- "PCA"
scvi$Embedding <- "scVI"

# Combine the dataframes
df_combined <- rbind(pca, scvi)

# Split DataSet column into Model and DataSet.Split columns
df_combined$Model <- ifelse(grepl("RNA", df_combined$DataSet), "RNA", "RNA + ATAC")
df_combined$DataSet.Split <- ifelse(grepl("Train", df_combined$DataSet), "Train", "Test")

# Reorder the factor levels for the desired order in the x-axis
df_combined$DataSet.Split <- factor(df_combined$DataSet.Split, levels = c("Train", "Test"))

# Reorder levels for the color palette to keep RNA in red and RNA + ATAC in blue
df_combined$Model <- factor(df_combined$Model, levels = c("RNA", "RNA + ATAC"))

# Plot with ggplot2
p <- ggplot(df_combined, aes(x = DataSet.Split, y = Silhouette.Score, color = Model)) +
  geom_boxplot(position = position_dodge(width=0.75), width=0.6) +
  facet_wrap(~ Embedding) + 
  labs(x = "Data Split", y = "Silhouette Score", fill = "Model") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_color_manual(values = c("red", "blue"))

print(p)

# Calculate the median and IQR
summary_df <- df_combined %>%
  group_by(Embedding, DataSet.Split, Model) %>%
  summarise(
    Median = median(Silhouette.Score),
    IQR_Lower = quantile(Silhouette.Score, 0.25),
    IQR_Upper = quantile(Silhouette.Score, 0.75)
  )

# Display the summary dataframe
print(summary_df)

# Statistical Tests

# Function to perform the Wilcoxon signed-rank test for each permutation
perform_test <- function(data) {
  group1 <- data[data$Model == "RNA",]$Silhouette.Score
  group2 <- data[data$Model == "RNA + ATAC",]$Silhouette.Score
  
  # Check if both groups have more than one data point
  if (length(group1) <= 1 | length(group2) <= 1) {
    return(list(statistic = NA, p.value = NA))
  }
  
  test_result <- wilcox.test(group1, group2, paired = TRUE)
  return(list(statistic = test_result$statistic, p.value = test_result$p.value))
}

# Group by Embedding and DataSet.Split, then apply the function
results <- df_combined %>%
  group_by(Embedding, DataSet.Split) %>%
  nest() %>%
  mutate(TestResults = map(data, perform_test))

# Extract p-values and adjust for multiple testing
p_values <- sapply(results$TestResults, function(x) x$p.value)
adjusted_p_values <- p.adjust(p_values, method = "bonferroni")

# Add adjusted p-values to results dataframe
results$AdjustedPValue <- adjusted_p_values
# Add a column to the results dataframe to show the number of asterisks
results$Significance <- case_when(
  results$AdjustedPValue < 0.001 ~ '***',
  results$AdjustedPValue < 0.01  ~ '**',
  results$AdjustedPValue < 0.05  ~ '*',
  TRUE                           ~ ''
)



# Violin Plots ------------------------------------------------------------

EMBEDDING = 'PCA' # Choose from: PCA, scVI (raw PCA?)
GROUND_TRUTH = 'wnnL2' # Choose from: wnnL2, wnnL1, rna [rna only for cancer data]
CELL_TYPE = 'All' # Choose from: B cells, T cells, Monoblast-Derived, All
N_COMPONENTS = 35   # Choose from: 10, 35
CL = 'RandomForestClassifier' # Choose from: RandomForestClassifier, SVC, LogisticRegression
DATA = 'pbmc' # Choose from: pbmc, cancer

for(GROUND_TRUTH in list('wnnL2')){
  for (CL in list('RandomForestClassifier', 'SVC', 'LogisticReg')){
    SUFFIX = glue('{DATA}_{CL}_{EMBEDDING}_{GROUND_TRUTH}_{CELL_TYPE}_{N_COMPONENTS}')
    # Read the CSV files
    df_A_metric1 <- read_csv(glue("Supervised Models/Results_{SUFFIX}_F1_df.csv"))
    df_A_metric2 <- read_csv(glue("Supervised Models/Results_{SUFFIX}_PAP_df.csv"))
    
    df_B_metric1 <- read_csv(glue("Supervised Models/Results_{SUFFIX}_F1_df_rna.csv"))
    df_B_metric2 <- read_csv(glue("Supervised Models/Results_{SUFFIX}_PAP_df_rna.csv"))
    
    # Add identifying columns to each dataframe
    df_A_metric1$Model <- 'RNA + ATAC'
    df_A_metric1$Metric <- 'F1 Scores'
    df_A_metric2$Model <- 'RNA + ATAC'
    df_A_metric2$Metric <- 'PAP Scores'
    
    df_B_metric1$Model <- 'RNA'
    df_B_metric1$Metric <- 'F1 Scores'
    df_B_metric2$Model <- 'RNA'
    df_B_metric2$Metric <- 'PAP Scores'
    # 
    # if(CELL_TYPE == 'B Cells') {
    #   col_names <-  c('B intermediate','B naive', 'NK')
    # } else if(CELL_TYPE == 'T Cells') {
    #   col_names <-  c('CD4 Naive', 'CD4 TCM', 'CD4 TEM', 'CD8 Naive', 'CD8 TEM')
    # } else if(CELL_TYPE == 'Monoblast-Derived') {
    #   col_names <-  c('CD14 Mono', 'CD16 Mono', 'cDC2', 'pDC')
    # }
    
    # Plot Results (unnanotated) ----------------------------------------------
    
    # Combine all dataframes
    df_all <- bind_rows(df_A_metric1, df_A_metric2, df_B_metric1, df_B_metric2)
    
    # Assuming 'df' is your data frame
    second_column_index <- 1
    last_two_columns_index <- ncol(df_all) - 1
    # Extract column names
    selected_column_names <- colnames(df_all)[(second_column_index + 1):(last_two_columns_index - 1)]
    # Print the selected column names
    print(selected_column_names)
    
    # Pivot the dataframe to a long format
    df_long <- df_all %>%
      pivot_longer(cols = selected_column_names,
                   names_to = "Cell Type",
                   values_to = "Value")
    
    # Separate plots for each metric
    for (metric in unique(df_long$Metric)) {
      # Filter data for the current metric
      df_metric <- df_long %>% filter(Metric == metric)
      
      # Plot
      p <- ggplot(df_metric, aes(x = Model, y = Value, fill = Model)) +
        geom_violin(position = "dodge") +
        geom_boxplot(width = 0.1, position = position_dodge(0.9)) +
        facet_wrap(~ `Cell Type`) +
        labs(x = "Model", y = "Value", fill = "Model", title = glue("{metric} {GROUND_TRUTH} {EMBEDDING} {CL}")) +
        theme_minimal()
      
      print(p)
    }
  }
}


# LEGACY CODE -------------------------------------------------------------

# Kurskal wallis figure 2 (ALGS) ------------------------------------------
# Statistical Tests
# # Adjust the Combination column to include the Model information
# df_combined$Combination <- paste(df_combined$Combination, df_combined$Model, sep = "_")
# 
# # Now run the Kruskal-Wallis test followed by the Dunn's test
# 
# results <- list()
# 
# # For each metric
# for (metric in unique(df_combined$Metric)) {
#   # Subset data by metric
#   subset_data <- df_combined[df_combined$Metric == metric, ]
#   
#   # Kruskal Wallis test
#   kruskal_res <- kruskal.test(`0` ~ Combination, data = subset_data)
#   
#   # Post hoc Dunn's test
#   print(metric)
#   dunn_res <- dunn.test(subset_data$`0`, subset_data$Combination, method = "bonferroni")
#   
#   # Store results
#   results[[metric]] <- list(KruskalWallis = kruskal_res, DunnTest = dunn_res)
# }
# 
# # Helper function to perform paired Wilcoxon test
# paired_wilcoxon_test <- function(data1, data2) {
#   test_result <- wilcox.test(data1, data2, paired = TRUE)
#   return(test_result$p.value)
# }
# 
# results <- list()
# 
# # For each metric
# for (metric in unique(df_combined$Metric)) {
#   # Subset data by metric
#   subset_data <- df_combined[df_combined$Metric == metric, ]
#   
#   # List to store p-values for each pairwise combination
#   p_values <- list()
#   
#   # Unique combinations
#   combinations <- unique(subset_data$Combination)
#   
#   # Perform Wilcoxon signed-rank test for every combination
#   for (i in 1:(length(combinations) - 1)) {
#     for (j in (i + 1):length(combinations)) {
#       combo1 <- combinations[i]
#       combo2 <- combinations[j]
#       
#       data1 <- subset_data[subset_data$Combination == combo1, ]$`0`
#       data2 <- subset_data[subset_data$Combination == combo2, ]$`0`
#       
#       p_value <- paired_wilcoxon_test(data1, data2)
#       p_values[[paste(combo1, combo2, sep = " vs. ")]] <- p_value
#     }
#   }
#   
#   # Adjust p-values using Bonferroni correction
#   adjusted_p_values <- p.adjust(unlist(p_values), method = "bonferroni")
#   
#   # Store results for the current metric
#   results[[metric]] <- adjusted_p_values
# }
# 
# # Print the results
# print(results)
# 
# # Group by Model, Classifier, Embedding, and Metric, then calculate the summary statistics
# summary_df <- df_combined %>%
#   group_by(Model, Classifier, Embedding, Metric) %>%
#   summarise(
#     Median = median(`0`),
#     Q1 = quantile(`0`, 0.25),
#     Q3 = quantile(`0`, 0.75)
#   )
# 
# print(summary_df)

# Per-class Boxplot ----------------------------------------------------------

EMBEDDINGS = c('PCA', 'scVI')
GROUND_TRUTHS = 'wnnL2'
CLASSIFIERS = c('RandomForestClassifier', 'SVC', 'LogisticRegression')
DATA = 'pbmc'
CELL_TYPE = 'All'
N_COMPONENTS = 35

df_combined <- data.frame()

for (GROUND_TRUTH in GROUND_TRUTHS) {
  for (EMBEDDING in EMBEDDINGS) {
    for (CL in CLASSIFIERS) {
      SUFFIX = glue('{DATA}_{CL}_{EMBEDDING}_{GROUND_TRUTH}_{CELL_TYPE}_{N_COMPONENTS}')
      
      # Read the CSV files
      df_A_metric1 <- read_csv(glue("Supervised Models/Results_{SUFFIX}_F1_df.csv"))
      df_A_metric2 <- read_csv(glue("Supervised Models/Results_{SUFFIX}_PAP_df.csv"))
      
      df_B_metric1 <- read_csv(glue("Supervised Models/Results_{SUFFIX}_F1_df_rna.csv"))
      df_B_metric2 <- read_csv(glue("Supervised Models/Results_{SUFFIX}_PAP_df_rna.csv"))
      
      # Add identifying columns
      df_A_metric1$Model <- 'RNA + ATAC'
      df_A_metric1$Metric <- 'F1 Scores'
      df_A_metric2$Model <- 'RNA + ATAC'
      df_A_metric2$Metric <- 'PAP Scores'
      
      df_B_metric1$Model <- 'RNA'
      df_B_metric1$Metric <- 'F1 Scores'
      df_B_metric2$Model <- 'RNA'
      df_B_metric2$Metric <- 'PAP Scores'
      
      df_all <- bind_rows(df_A_metric1, df_A_metric2, df_B_metric1, df_B_metric2) %>%
        mutate(Combination = paste(EMBEDDING, CL, sep = "_"))
      df_all <- df_all %>% select(-...1)
      df_all <- df_all %>%
        mutate(Combination = str_replace_all(Combination, "_", " "))
      df_all <- df_all %>%
        mutate(Combination = str_replace_all(Combination, "RandomForestClassifier", "RF"))
      df_all <- df_all %>%
        mutate(Combination = str_replace_all(Combination, "LogisticRegression", "LR"))
      
      df_combined <- bind_rows(df_combined, df_all)
    }
  }
}

# Extract the column names for the cell types
selected_column_names <- colnames(df_combined)[3:(ncol(df_combined) - 2)]

numeric_columns <- select(df_combined, where(is.numeric)) %>% names()
print(numeric_columns)

df_long <- df_combined %>%
  pivot_longer(cols = all_of(numeric_columns),
               names_to = "Cell Type",
               values_to = "Value")

# Plot the data
for (metric in unique(df_long$Metric)) {
  # Filter data for the current metric
  df_metric <- df_long %>% filter(Metric == metric)
  
  # Plot
  p <- ggplot(df_metric, aes(x = Combination, y = Value, color = Model)) +
    geom_boxplot(position = "dodge") +
    facet_wrap(~ `Cell Type`) +
    labs(x = "Embedding + Classifier", y = metric, fill = "Model") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))+
    scale_color_brewer(palette="Set1")
  
  print(p)
}

