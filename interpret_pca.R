# Load libraries
suppressMessages(library(dplyr))
suppressMessages(library(ggplot2))

# Load data
rfm_data <- read.csv("data/customer_rfm.csv")

# Select numeric columns
rfm_vars <- rfm_data %>% select(Recency, Frequency, Monetary)

# Scale
rfm_scaled <- scale(rfm_vars)

# Compute PCA
pca <- prcomp(rfm_scaled)

# Print loadings (rotation)
cat("PCA Loadings (Rotation):\n")
print(pca$rotation)

# Print percent variance
cat("\nVariance Explained:\n")
print(summary(pca)$importance)
