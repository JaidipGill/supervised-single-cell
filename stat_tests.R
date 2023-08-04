# Load necessary package
library(tidyverse)
library(ggplot2)
library(glue)
library(ggsignif)


# Loading Data ------------------------------------------------------------

EMBEDDING <- 'PCA MINOR' # Choose from: PCA, CCA, scVI
GROUND_TRUTH <- 'wnnL2' # Choose from: wnnL2, wnnL1, rna
CELL_TYPE <- 'ALL' # Choose from B Cells, T cells, Monoblast-Derived
N_COMPONENTS <- '35'   # Choose from: 10, 35
SUPERVISED_MODEL <- 'RF' # Choose from: SVM, LOGREG, RF

# Read the CSV files
df_A_metric1 <- read_csv(glue("Supervised Models/{EMBEDDING}/{SUPERVISED_MODEL}_RNA_only_{GROUND_TRUTH}_{CELL_TYPE}_{N_COMPONENTS}_F1_df.csv"))
df_A_metric2 <- read_csv(glue("Supervised Models/{EMBEDDING}/{SUPERVISED_MODEL}_RNA_only_{GROUND_TRUTH}_{CELL_TYPE}_{N_COMPONENTS}_PAP_df.csv"))
df_B_metric1 <- read_csv(glue("Supervised Models/{EMBEDDING}/{SUPERVISED_MODEL}_{GROUND_TRUTH}_{CELL_TYPE}_{N_COMPONENTS}_F1_df.csv"))
df_B_metric2 <- read_csv(glue("Supervised Models/{EMBEDDING}/{SUPERVISED_MODEL}_{GROUND_TRUTH}_{CELL_TYPE}_{N_COMPONENTS}_PAP_df.csv"))

# Add identifying columns to each dataframe
df_A_metric1$Model <- 'RNA'
df_A_metric1$Metric <- 'F1 Scores'
df_A_metric2$Model <- 'RNA'
df_A_metric2$Metric <- 'PAP Scores'
df_B_metric1$Model <- 'RNA + ATAC'
df_B_metric1$Metric <- 'F1 Scores'
df_B_metric2$Model <- 'RNA + ATAC'
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
    labs(x = "Model", y = "Value", fill = "Model", title = paste0("Violin plots for ", metric)) +
    theme_minimal()
  
  print(p)
}

# Statistical Tests -------------------------------------------------------

# for(df in list(df_A_metric1,df_A_metric2,df_B_metric1,df_B_metric2)){
#   # Convert dataframe from wide to long format
#   df_long <- df %>% pivot_longer(cols = all_of(col_names), 
#                                  names_to = "Column", 
#                                  values_to = "Value")
#   
#   # Create facet plot of histograms
#   p<-ggplot(df_long, aes(x = Value)) +
#     geom_histogram(binwidth = 10) +  # You can adjust binwidth as per your data
#     facet_wrap(~ Column, scales = "free_x") +  # if you want to have individual scales for each facet
#     theme_minimal() +
#     xlab("Value") +
#     ylab("Frequency") 
#   print(p)
# }

# Define function to perform paired t-test
perform_t_test <- function(df1, df2, column_name) {
  result<-t.test(df1[[column_name]], df2[[column_name]], paired = TRUE)
  print(result)
  return(result$p.value)
}

# Initialize list to store t-test results
t_test_results <- list()

# Perform t-tests
for (col_name in col_names) {
  print(col_name)
  t_test_results[[paste0(col_name, "_F1Score")]] <- perform_t_test(df_A_metric1, df_B_metric1, col_name)
  t_test_results[[paste0(col_name, "_PAPScore")]] <- perform_t_test(df_A_metric2, df_B_metric2, col_name)
}

# Check t-test results
print(t_test_results)

# Adjust p-values for multiple comparisons
adjusted_p_values<- p.adjust(t_test_results, method = "bonferroni")

# Print the results
print(adjusted_p_values)

# Plot Results ------------------------------------------------------------

# Combine all dataframes
df_all <- bind_rows(df_A_metric1, df_A_metric2, df_B_metric1, df_B_metric2)

# Pivot the dataframe to a long format
df_long <- df_all %>%
  pivot_longer(cols = col_names,
               names_to = "Cell Type",
               values_to = "Value")

# Function to get significance annotations
get_sig_annotation <- function(p_value) {
  if (is.na(p_value)) {
    return("NA")
  } else if (p_value < 0.001) {
    return("***")
  } else if (p_value < 0.01) {
    return("**")
  } else if (p_value < 0.05) {
    return("*")
  } else {
    return("NS")
  }
}

# Apply function to your adjusted p-values to get the annotations
annotations <- sapply(adjusted_p_values, get_sig_annotation)

# Convert annotations to a dataframe for easier access
df_annotations <- data.frame(Cell_Type = names(annotations), Annotation = annotations)

# Append Metric info to the `Cell Type` in df_metric
df_metric$`Cell Type` <- paste(df_metric$`Cell Type`, "_", df_metric$Metric, sep = "")

# Separate plots for each metric
for (metric in unique(df_long$Metric)) {
  # Filter data for the current metric
  df_metric <- df_long %>% filter(Metric == metric)
  
  # Calculate max value for each cell type
  max_values <- df_metric %>% group_by(`Cell Type`) %>% summarise(Max_Value = max(Value))
  
  # Plot
  p <- ggplot(df_metric, aes(x = Model, y = Value, fill = Model)) +
    geom_violin(position = "dodge") +
    geom_boxplot(width = 0.1, position = position_dodge(0.9)) +
    facet_wrap(~ `Cell Type`) +
    labs(x = "Model", y = "Value", fill = "Model", title = paste0("Violin plots for ", metric)) +
    theme_minimal()
  
  # Add significance annotations
  for (cell_type in unique(df_metric$`Cell Type`)) {
    annotation <- df_annotations %>% filter(Cell_Type == cell_type) %>% pull(Annotation)
    max_value <- max_values %>% filter(`Cell Type` == cell_type) %>% pull(Max_Value)
    
    # Add significance brackets
    p <- p + geom_signif(comparison = list(c("Model A", "Model B")), y_position = max_value, annotation = annotation, textsize = 4, vjust = -0.5)
  }
  
  print(p)
}

