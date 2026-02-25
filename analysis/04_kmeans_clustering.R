# CRM CA3 - K-Means Clustering Analysis
# Advanced CRM Technique: Customer Segmentation using K-Means

# Load required libraries
library(dplyr)
library(ggplot2)
library(cluster)
library(factoextra)

# Set working directory
setwd("/home/jd/crm-ca3")

# Create figures directory if not exists
if (!dir.exists("figures")) dir.create("figures")

# ============================================================
# 1. LOAD RFM DATA
# ============================================================
cat("Loading RFM data...\n")
rfm_data <- read.csv("data/customer_rfm.csv", stringsAsFactors = FALSE)
cat("Customers loaded:", nrow(rfm_data), "\n")

# Select RFM variables for clustering
rfm_vars <- rfm_data %>%
  dplyr::select(CustomerID, Recency, Frequency, Monetary)

cat("\nRFM Summary (before scaling):\n")
print(summary(rfm_vars[, c("Recency", "Frequency", "Monetary")]))

# ============================================================
# 2. DATA PREPARATION FOR K-MEANS
# ============================================================
cat("\n========================================\n")
cat("DATA PREPARATION FOR K-MEANS\n")
cat("========================================\n")

# Scale the data (standardization is mandatory for K-Means)
# As per Knowledge Base Lesson 8: "Scaling (z-scores) is mandatory"
rfm_scaled <- scale(rfm_vars[, c("Recency", "Frequency", "Monetary")])
cat("Data standardized using z-scores\n")

cat("\nScaled data summary:\n")
print(summary(rfm_scaled))

# ============================================================
# 3. DETERMINE OPTIMAL K (ELBOW METHOD)
# ============================================================
cat("\n========================================\n")
cat("DETERMINING OPTIMAL K (ELBOW METHOD)\n")
cat("========================================\n")

# Calculate Within Sum of Squares for k = 1 to 10
set.seed(42)  # For reproducibility
wss <- numeric(10)
for (k in 1:10) {
  km <- kmeans(rfm_scaled, centers = k, nstart = 25, iter.max = 100)
  wss[k] <- km$tot.withinss
}

# Create Elbow Plot
elbow_data <- data.frame(k = 1:10, WSS = wss)
cat("\nWithin Sum of Squares by K:\n")
print(elbow_data)

png("figures/13_elbow_plot.png", width = 800, height = 500, res = 120)
ggplot(elbow_data, aes(x = k, y = WSS)) +
  geom_line(color = "steelblue", size = 1) +
  geom_point(color = "steelblue", size = 3) +
  geom_vline(xintercept = 4, linetype = "dashed", color = "red") +
  annotate("text", x = 4.5, y = max(wss) * 0.9, label = "Optimal K = 4", color = "red") +
  labs(title = "Elbow Method for Optimal K",
       subtitle = "Within Sum of Squares (WSS) by Number of Clusters",
       x = "Number of Clusters (K)", y = "Within Sum of Squares") +
  scale_x_continuous(breaks = 1:10) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))
dev.off()
cat("Elbow plot saved: figures/13_elbow_plot.png\n")

# ============================================================
# 4. K-MEANS CLUSTERING (K=4)
# ============================================================
cat("\n========================================\n")
cat("K-MEANS CLUSTERING (K=4)\n")
cat("========================================\n")

# Apply K-Means with optimal K=4
set.seed(42)
kmeans_result <- kmeans(rfm_scaled, centers = 4, nstart = 25, iter.max = 100)

cat("\nK-Means Results:\n")
cat("  Cluster sizes:", kmeans_result$size, "\n")
cat("  Total WSS:", round(kmeans_result$tot.withinss, 2), "\n")
cat("  Between SS / Total SS:", round(kmeans_result$betweenss / kmeans_result$totss * 100, 2), "%\n")

# Add cluster assignment to data
rfm_vars$Cluster <- as.factor(kmeans_result$cluster)

# ============================================================
# 5. CLUSTER ANALYSIS AND PROFILING
# ============================================================
cat("\n========================================\n")
cat("CLUSTER PROFILING\n")
cat("========================================\n")

# Calculate cluster centroids (original scale)
cluster_profile <- rfm_vars %>%
  group_by(Cluster) %>%
  summarise(
    Count = n(),
    Percentage = round(n() / nrow(rfm_vars) * 100, 1),
    Avg_Recency = round(mean(Recency), 0),
    Avg_Frequency = round(mean(Frequency), 1),
    Avg_Monetary = round(mean(Monetary), 0),
    Total_Revenue = round(sum(Monetary), 0)
  ) %>%
  arrange(desc(Avg_Monetary))

cat("\nCluster Profiles:\n")
print(cluster_profile)

# Assign cluster labels based on characteristics
# Low Recency = Recent, High Frequency = Frequent, High Monetary = Valuable
cluster_labels <- cluster_profile %>%
  mutate(
    Label = case_when(
      Avg_Monetary > 100000 ~ "Elite VIP",
      Avg_Recency < 50 & Avg_Frequency > 5 & Avg_Monetary > 3000 ~ "VIP Customers",
      Avg_Recency < 80 & Avg_Frequency > 2 ~ "Active Customers",
      Avg_Recency > 150 ~ "Churned Customers",
      TRUE ~ "Occasional Customers"
    )
  )

cat("\nCluster Labels:\n")
print(cluster_labels %>% dplyr::select(Cluster, Label, Count, Percentage, Avg_Recency, Avg_Frequency, Avg_Monetary))

# Add labels to main data
label_map <- setNames(cluster_labels$Label, cluster_labels$Cluster)
rfm_vars$Cluster_Label <- label_map[as.character(rfm_vars$Cluster)]

# ============================================================
# 6. CLUSTER VISUALIZATIONS
# ============================================================
cat("\n========================================\n")
cat("CREATING VISUALIZATIONS\n")
cat("========================================\n")

# 6.1 Cluster Distribution
png("figures/14_kmeans_distribution.png", width = 900, height = 500, res = 120)
ggplot(cluster_labels, aes(x = reorder(Label, -Count), y = Count, fill = Label)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = paste0(Count, " (", Percentage, "%)")), vjust = -0.5) +
  labs(title = "K-Means Customer Segmentation",
       subtitle = "Distribution of Customers Across 4 Clusters",
       x = "Customer Segment", y = "Number of Customers") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        legend.position = "none",
        axis.text.x = element_text(angle = 15, hjust = 1)) +
  scale_fill_brewer(palette = "Set2")
dev.off()
cat("Distribution plot saved: figures/14_kmeans_distribution.png\n")

# 6.2 Cluster Scatter Plot (Frequency vs Monetary)
png("figures/15_kmeans_scatter.png", width = 1000, height = 700, res = 120)
ggplot(rfm_vars, aes(x = Frequency, y = Monetary/1000, color = Cluster_Label)) +
  geom_point(alpha = 0.6, size = 2) +
  labs(title = "K-Means Clustering: Frequency vs Monetary Value",
       x = "Purchase Frequency (Number of Orders)",
       y = "Monetary Value (£ thousands)",
       color = "Segment") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold")) +
  scale_color_brewer(palette = "Set1")
dev.off()
cat("Scatter plot saved: figures/15_kmeans_scatter.png\n")

# 6.3 Cluster Centroids Radar/Profile Plot
centroid_long <- cluster_labels %>%
  dplyr::select(Cluster, Label, Avg_Recency, Avg_Frequency, Avg_Monetary) %>%
  tidyr::pivot_longer(cols = c(Avg_Recency, Avg_Frequency, Avg_Monetary),
                      names_to = "Metric", values_to = "Value")

# Normalize for comparison
centroid_long <- centroid_long %>%
  group_by(Metric) %>%
  mutate(Normalized = (Value - min(Value)) / (max(Value) - min(Value))) %>%
  ungroup()

png("figures/16_cluster_profile.png", width = 1000, height = 600, res = 120)
ggplot(centroid_long, aes(x = Metric, y = Normalized, fill = Label)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Cluster Profiles: RFM Metrics Comparison",
       subtitle = "Normalized values (0-1 scale) for each metric",
       x = "RFM Metric", y = "Normalized Value",
       fill = "Segment") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold")) +
  scale_fill_brewer(palette = "Set2")
dev.off()
cat("Profile plot saved: figures/16_cluster_profile.png\n")

# 6.4 3D-like visualization using factoextra
png("figures/17_kmeans_clusters.png", width = 900, height = 600, res = 120)
fviz_cluster(kmeans_result, data = rfm_scaled,
             geom = "point",
             ellipse.type = "convex",
             palette = "Set2",
             ggtheme = theme_minimal()) +
  labs(title = "K-Means Clusters (PCA Projection)") +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))
dev.off()
cat("Cluster visualization saved: figures/17_kmeans_clusters.png\n")

# ============================================================
# 7. SILHOUETTE ANALYSIS
# ============================================================
cat("\n========================================\n")
cat("SILHOUETTE ANALYSIS\n")
cat("========================================\n")

# Calculate silhouette scores
sil <- silhouette(kmeans_result$cluster, dist(rfm_scaled))
avg_sil <- mean(sil[, 3])

cat("Average Silhouette Score:", round(avg_sil, 3), "\n")
cat("Interpretation: ")
if (avg_sil > 0.5) {
  cat("Strong structure (good clustering)\n")
} else if (avg_sil > 0.25) {
  cat("Reasonable structure\n")
} else {
  cat("Weak structure\n")
}

# Silhouette plot
png("figures/18_silhouette.png", width = 900, height = 600, res = 120)
fviz_silhouette(sil, palette = "Set2") +
  labs(title = paste("Silhouette Analysis (Avg Width =", round(avg_sil, 3), ")")) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))
dev.off()
cat("Silhouette plot saved: figures/18_silhouette.png\n")

# ============================================================
# 8. BUSINESS RECOMMENDATIONS PER CLUSTER
# ============================================================
cat("\n========================================\n")
cat("CLUSTER-SPECIFIC RECOMMENDATIONS\n")
cat("========================================\n")

recommendations <- data.frame(
  Cluster = cluster_labels$Label,
  Revenue_Share = paste0(round(cluster_labels$Total_Revenue / sum(cluster_labels$Total_Revenue) * 100, 1), "%"),
  Strategy = c(
    "Concierge Service: Dedicated account manager, custom deals, annual gifts",
    "VIP Program: Exclusive offers, early access, priority support",
    "Engagement: Regular newsletters, loyalty points, cross-sell campaigns",
    "Win-back: Reactivation emails, 'We miss you' discounts, feedback surveys",
    "Nurture: Welcome series, product education, second-purchase incentives"
  )[match(cluster_labels$Label, c("Elite VIP", "VIP Customers", "Active Customers", "Churned Customers", "Occasional Customers"))]
)

cat("\nCluster-Specific Marketing Strategies:\n")
print(recommendations)

# ============================================================
# 9. SAVE RESULTS
# ============================================================
write.csv(rfm_vars, "data/customer_kmeans.csv", row.names = FALSE)
cat("\nK-Means results saved to: data/customer_kmeans.csv\n")

# Save cluster summary
write.csv(cluster_labels, "data/cluster_summary.csv", row.names = FALSE)
cat("Cluster summary saved to: data/cluster_summary.csv\n")

cat("\n\nK-Means Clustering Analysis completed successfully!\n")
cat("New figures saved: 13-18\n")
