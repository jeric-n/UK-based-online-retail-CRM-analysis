# CRM CA3 - Regression Analysis (Corrected for Report Consistency)
# This script performs Linear Regression, Multiple Regression, and VIF analysis
# Modified to ensure outputs match the Full Model described in the report

# Load required libraries
library(dplyr)
library(ggplot2)
library(car)       # For VIF
library(MASS)      # For stepAIC

# Set working directory
setwd("/home/jd/crm-ca3")

# Create figures directory if not exists
if (!dir.exists("figures")) dir.create("figures")

# ============================================================
# 1. LOAD DATA
# ============================================================
cat("Loading customer data...\n")
customer_data <- read.csv("data/customer_data.csv", stringsAsFactors = FALSE)
cat("Customer dataset:", nrow(customer_data), "customers\n")

# Check for any NA values
cat("\nMissing values:\n")
print(colSums(is.na(customer_data)))

# Remove rows with NA values for regression
customer_data <- na.omit(customer_data)
cat("After removing NAs:", nrow(customer_data), "customers\n")

# ============================================================
# 2. CORRELATION ANALYSIS
# ============================================================
cat("\n========================================\n")
cat("CORRELATION ANALYSIS\n")
cat("========================================\n")

# Select numeric variables for correlation
numeric_vars <- customer_data %>%
  dplyr::select(TotalRevenue, TotalOrders, TotalItems, TotalProducts, 
                AvgOrderValue, AvgItemPrice, AvgQuantityPerOrder, 
                DaysSinceFirst, DaysSinceLast)

# Compute correlation matrix
cor_matrix <- cor(numeric_vars, use = "complete.obs")
cat("\nCorrelation matrix (with TotalRevenue):\n")
print(round(cor_matrix[, "TotalRevenue"], 3))

# Correlation with TotalRevenue
cor_with_revenue <- cor_matrix[, "TotalRevenue"]
cat("\nVariables correlated with TotalRevenue:\n")
print(sort(cor_with_revenue, decreasing = TRUE))

# ============================================================
# 3. SIMPLE LINEAR REGRESSION
# ============================================================
cat("\n========================================\n")
cat("SIMPLE LINEAR REGRESSION\n")
cat("========================================\n")

# Model: TotalRevenue ~ TotalOrders
cat("\n--- Model 1: TotalRevenue ~ TotalOrders ---\n")
model_simple <- lm(TotalRevenue ~ TotalOrders, data = customer_data)
print(summary(model_simple))

# Interpretation
cat("\nInterpretation:\n")
cat("  Intercept:", round(coef(model_simple)[1], 2), "\n")
cat("  Slope (TotalOrders):", round(coef(model_simple)[2], 2), "\n")
cat("  R-squared:", round(summary(model_simple)$r.squared, 4), "\n")
cat("  Adj. R-squared:", round(summary(model_simple)$adj.r.squared, 4), "\n")
cat("  F-statistic p-value:", format(pf(summary(model_simple)$fstatistic[1], 
    summary(model_simple)$fstatistic[2], summary(model_simple)$fstatistic[3], 
    lower.tail = FALSE), scientific = TRUE), "\n")

# Save linear regression plot
png("figures/08_linear_regression.png", width = 1000, height = 600, res = 120)
par(mfrow = c(1, 2))

# Scatter plot with regression line
plot(customer_data$TotalOrders, customer_data$TotalRevenue/1000,
     xlab = "Total Orders", ylab = "Total Revenue (£ thousands)",
     main = "Simple Linear Regression: Revenue vs Orders",
     pch = 16, col = rgb(0.2, 0.4, 0.6, 0.4), cex = 0.8)
abline(model_simple$coefficients[1]/1000, model_simple$coefficients[2]/1000, 
       col = "red", lwd = 2)
legend("topright", legend = paste("R² =", round(summary(model_simple)$r.squared, 3)),
       bty = "n", cex = 1.2)

# Residuals vs Fitted
plot(model_simple$fitted.values/1000, model_simple$residuals/1000,
     xlab = "Fitted Values (£ thousands)", ylab = "Residuals (£ thousands)",
     main = "Residuals vs Fitted Values",
     pch = 16, col = rgb(0.2, 0.4, 0.6, 0.4), cex = 0.8)
abline(h = 0, col = "red", lty = 2)

dev.off()
cat("\nPlot saved: figures/08_linear_regression.png\n")

# ============================================================
# 4. MULTIPLE LINEAR REGRESSION
# ============================================================
cat("\n========================================\n")
cat("MULTIPLE LINEAR REGRESSION\n")
cat("========================================\n")

# Full Model with multiple predictors
cat("\n--- Full Model: All Predictors ---\n")
model_full <- lm(TotalRevenue ~ TotalOrders + TotalItems + TotalProducts + 
                   AvgItemPrice + DaysSinceFirst + IsUK, 
                 data = customer_data)
print(summary(model_full))

# Check VIF for multicollinearity FIRST
cat("\n--- VIF Analysis (Full Model) ---\n")
vif_values <- vif(model_full)
print(vif_values)
cat("\nVIF Interpretation:\n")
cat("  VIF > 10: High multicollinearity (problematic)\n")
cat("  VIF > 5: Moderate multicollinearity (caution)\n")
cat("  VIF < 5: Low multicollinearity (acceptable)\n")

# Flag problematic variables
problematic <- names(vif_values[vif_values > 5])
if (length(problematic) > 0) {
  cat("\nVariables with high VIF (>5):", paste(problematic, collapse = ", "), "\n")
} else {
  cat("\nAll variables have acceptable VIF values (<5)\n")
}

# ============================================================
# 7. MODEL DIAGNOSTICS PLOTS (USING FULL MODEL)
# ============================================================
cat("\n========================================\n")
cat("MODEL DIAGNOSTICS (FULL MODEL)\n")
cat("========================================\n")

# Diagnostic plots for the FULL model (to match report R2=0.85)
png("figures/09_regression_diagnostics.png", width = 1200, height = 1000, res = 120)
par(mfrow = c(2, 2))
plot(model_full)
dev.off()
cat("Diagnostic plots saved: figures/09_regression_diagnostics.png\n")

# Additional visualization: Coefficient Plot
coef_data <- data.frame(
  Variable = names(coef(model_full))[-1],
  Coefficient = coef(model_full)[-1],
  StdError = summary(model_full)$coefficients[-1, 2]
)
coef_data$Lower <- coef_data$Coefficient - 1.96 * coef_data$StdError
coef_data$Upper <- coef_data$Coefficient + 1.96 * coef_data$StdError

png("figures/10_coefficient_plot.png", width = 900, height = 600, res = 120)
ggplot(coef_data, aes(x = reorder(Variable, Coefficient), y = Coefficient)) +
  geom_point(size = 3, color = "steelblue") +
  geom_errorbar(aes(ymin = Lower, ymax = Upper), width = 0.2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  coord_flip() +
  labs(title = "Multiple Regression Coefficients (Full Model)",
       x = "Variable", y = "Coefficient Value") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))
dev.off()
cat("Coefficient plot saved: figures/10_coefficient_plot.png\n")

# ============================================================
# 8. VIF VISUALIZATION (USING FULL MODEL)
# ============================================================
png("figures/11_vif_analysis.png", width = 900, height = 500, res = 120)
vif_df <- data.frame(Variable = names(vif_values), VIF = as.numeric(vif_values))
ggplot(vif_df, aes(x = reorder(Variable, VIF), y = VIF, fill = VIF > 5)) +
  geom_bar(stat = "identity") +
  geom_hline(yintercept = 5, linetype = "dashed", color = "orange", size = 1) +
  geom_hline(yintercept = 10, linetype = "dashed", color = "red", size = 1) +
  coord_flip() +
  labs(title = "Variance Inflation Factor (VIF) Analysis",
       subtitle = "VIF > 5 indicates moderate multicollinearity, > 10 indicates high",
       x = "Variable", y = "VIF Value") +
  scale_fill_manual(values = c("TRUE" = "coral", "FALSE" = "steelblue")) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
       legend.position = "none")
dev.off()
cat("VIF plot saved: figures/11_vif_analysis.png\n")

# ============================================================
# 9. PREDICTION ANALYSIS (USING FULL MODEL)
# ============================================================
cat("\n========================================\n")
cat("PREDICTION ANALYSIS (FULL MODEL)\n")
cat("========================================\n")

# Add predictions to customer data
customer_data$PredictedRevenue <- predict(model_full, customer_data)
customer_data$Residual <- customer_data$TotalRevenue - customer_data$PredictedRevenue

# Prediction accuracy metrics
mae <- mean(abs(customer_data$Residual))
rmse <- sqrt(mean(customer_data$Residual^2))
mape <- mean(abs(customer_data$Residual / customer_data$TotalRevenue)) * 100

cat("\nPrediction Metrics:\n")
cat("  Mean Absolute Error (MAE): £", round(mae, 2), "\n")
cat("  Root Mean Square Error (RMSE): £", round(rmse, 2), "\n")
cat("  Mean Absolute Percentage Error (MAPE):", round(mape, 2), "%\n")

# Actual vs Predicted plot
png("figures/12_actual_vs_predicted.png", width = 800, height = 600, res = 120)
plot(customer_data$TotalRevenue/1000, customer_data$PredictedRevenue/1000,
     xlab = "Actual Revenue (£ thousands)", 
     ylab = "Predicted Revenue (£ thousands)",
     main = "Actual vs Predicted Customer Revenue",
     pch = 16, col = rgb(0.2, 0.4, 0.6, 0.3), cex = 0.8)
abline(0, 1, col = "red", lwd = 2)
legend("topleft", 
       legend = c(paste("R² =", round(summary(model_full)$r.squared, 3)),
                  paste("MAE = £", round(mae, 0)),
                  paste("RMSE = £", round(rmse, 0))),
       bty = "n", cex = 1)
dev.off()
cat("Actual vs Predicted plot saved: figures/12_actual_vs_predicted.png\n")

# ============================================================
# 10. SUMMARY RESULTS
# ============================================================
cat("\n========================================\n")
cat("REGRESSION ANALYSIS SUMMARY\n")
cat("========================================\n")

cat("\n1. SIMPLE LINEAR REGRESSION (Revenue ~ Orders):\n")
cat("   R-squared:", round(summary(model_simple)$r.squared, 4), "\n")
cat("   Conclusion: TotalOrders explains", 
    round(summary(model_simple)$r.squared * 100, 1), "% of revenue variation\n")

cat("\n2. MULTIPLE REGRESSION (Full Model):\n")
cat("   R-squared:", round(summary(model_full)$r.squared, 4), "\n")
cat("   Adj. R-squared:", round(summary(model_full)$adj.r.squared, 4), "\n")
cat("   Significant predictors (p < 0.05):\n")
signif_vars <- summary(model_full)$coefficients[, 4] < 0.05
print(names(signif_vars[signif_vars == TRUE]))

cat("\n3. VIF ANALYSIS:\n")
cat("   All variables below threshold: ", all(vif_values < 5), "\n")
if (any(vif_values > 5)) {
  cat("   Variables with VIF > 5:", names(vif_values[vif_values > 5]), "\n")
}

cat("\n4. MODEL EQUATION:\n")
cat("   TotalRevenue = ", round(coef(model_full)[1], 2))
for (i in 2:length(coef(model_full))) {
  sign <- ifelse(coef(model_full)[i] >= 0, " + ", " - ")
  cat(sign, round(abs(coef(model_full)[i]), 2), "*", names(coef(model_full))[i])
}
cat("\n")

# Save regression results
regression_results <- list(
  simple_model = model_simple,
  full_model = model_full,
  vif_full = vif_values
)
saveRDS(regression_results, "data/regression_results.rds")
cat("\nRegression results saved to: data/regression_results.rds\n")

cat("\n\nRegression Analysis completed successfully!\n")
