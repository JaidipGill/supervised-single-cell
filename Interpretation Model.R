library(tidyverse) 
library(pheatmap)
library(glmnet)
library(sharp)
library(ggplot2)
library(nnet)
library(Hmisc)
library(corrplot)

X <- read.csv("Data/PBMC 10k multiomic/Interpretation/X_.csv")
y <- read.csv("Data/PBMC 10k multiomic/Interpretation/y_.csv")
y <- subset(y, select = c(X2))
X <- subset(X, select = -c(X))

# Identify the rows to drop
rows_to_drop <- which(y$X2 %in% c('Plasmablast', 'dnT'))
# Drop these rows from both X and y
X <- X[-rows_to_drop, ]
y <- y[-rows_to_drop, ]

# Descriptive stats
predictors <- names(X)

# Calculate correlation matrix and p-values
cor_test <- rcorr(as.matrix(X))
cor_matrix <- cor_test$r
p_values <- cor_test$P

# Correct p-values for multiple testing using the Bonferroni method
p_values_corrected <- p.adjust(p_values, method = "bonferroni")
p_values_corrected_matrix <- matrix(p_values_corrected, ncol=70)


# Determine significance based on corrected p-values, for instance at 0.05 level
significance_matrix <- ifelse(p_values_corrected < 0.05, "*", "")
p_values_corrected_matrix <- matrix(significance_matrix, ncol=70)
rownames(p_values_corrected_matrix) <- rownames(cor_matrix)
colnames(p_values_corrected_matrix) <- colnames(cor_matrix)
pheatmap(cor_matrix,
         main = "",
         color = colorRampPalette(c("blue", "white", "red"))(25), # Blue (negative) to white to red (positive)
         fontsize_row = 8,
         fontsize_col = 8,
         border_color = NA,
         display_numbers = TRUE) # This will remove the border around cells

# Find proportion of weak correlations
n <- 35  # Replace this with your actual value
m <- 35 # Replace this with your actual value

rna_atac_cor <- cor_matrix[1:n, (n+1):(n+m)]
logical_matrix <- rna_atac_cor > -0.3 & rna_atac_cor < 0.3
proportion_correlations <- sum(logical_matrix) / length(rna_atac_cor)

print(proportion_correlations)

# Extracting p-values for RNA-ATAC correlations
rna_atac_pvalues <- p_values_corrected[1:n, (n+1):(n+m)]

# Get logical matrix for significant correlations (e.g., at 0.05 level)
logical_significant <- rna_atac_pvalues < 0.05

# Calculate proportion
proportion_significant <- sum(logical_significant) / length(rna_atac_pvalues)

print(proportion_significant)



# Corrplot ----------------------------------------------------------------

# Calculate correlation matrix and p-values
cor_test <- cor.mtest(as.matrix(X))
cor_matrix <- cor(X)
p_values <- cor_test$p
p_values_corrected <- p.adjust(p_values, method = "bonferroni")
p_values_corrected_matrix <- matrix(p_values_corrected, ncol=70)

# Display correlation plot
corrplot(cor_matrix, 
         p.mat = p_values, 
         method = 'color', 
         diag = FALSE, 
         type = 'upper',
         sig.level = c(0.01, 0.05), 
         pch.cex = 0.6,   # Adjust the size of significance indicators (*)
         insig = 'label_sig', 
         pch.col = 'black', # Color of significance indicators
         order = 'AOE',
         tl.col = 'black', # Text label color
         tl.cex = 0.8,     # Text label size
         title = "Correlation Plot", 
         mar = c(0.5,0.5,0.5,0.5), # Margin around the plot
         cl.cex = 0.8)    # Color legend size
# Logistic Regression -----------------------------------------------------

# Convert y to a factor
y <- as.factor(y)
# Assuming you have your predictors in X and response in y
# Fit the model
df<-cbind(X, y)
model <- multinom(y ~ ., data = df)
# Print the summary
summary(model)



# Lasso -------------------------------------------------------------------

# Convert y to a factor
y <- as.factor(y)
# Fit multinomial logistic regression with lasso penalty
fit <- cv.glmnet(as.matrix(X), y, family = "multinomial", alpha=1)
# Display results
coef(fit, s = fit$lambda.1se)
plot(fit)

# Assuming you have already fitted your model as fit
coefs <- coef(fit, s = fit$lambda.1se)

# Extract non-zero coefficients for each class and bind them together
all_coefs <- lapply(names(coefs), function(class_name) {
  class_coef_matrix <- as.matrix(coefs[[class_name]])
  class_coef_df <- as.data.frame(class_coef_matrix)
  colnames(class_coef_df) <- c("Coefficient")
  
  # Add feature names as a separate column and the class name
  class_coef_df$Feature <- rownames(class_coef_matrix)
  class_coef_df$Class <- class_name
  
  # Filter out zero coefficients
  class_coef_df <- class_coef_df[!class_coef_df$Coefficient == 0, , drop = FALSE]
  
  class_coef_df
})

# Bind all data frames together
all_coefs_df <- bind_rows(all_coefs)

# Plot with faceting
ggplot(all_coefs_df, aes(x = Feature, y = Coefficient)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Non-zero coefficients for each class",
       x = "Features",
       y = "Coefficient Value") +
  theme_minimal() +
  facet_wrap(~Class, scales = "free", ncol = 2)


# ATAC Weights ------------------------------------------------------------

# Sum up coefficients for 'ATAC' and 'RNA' features for each class
sums <- all_coefs_df %>%
  group_by(Class, FeatureType = ifelse(grepl("ATAC", Feature), "ATAC", ifelse(grepl("RNA", Feature), "RNA", NA))) %>%
  summarise(Total = sum(abs(Coefficient), na.rm = TRUE)) %>%
  filter(!is.na(FeatureType)) %>%
  spread(FeatureType, Total, fill = 0)

# Compute ATAC weight for each class
sums <- sums %>%
  mutate(ATAC_weight = ATAC / (ATAC + RNA))

# Plot the ATAC weights for each class
ggplot(sums, aes(x = Class, y = ATAC_weight)) +
  geom_bar(stat = "identity") +
  labs(title = NULL,
       x = "Class",
       y = "ATAC Weight") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
