# CRM CA3 - Exploratory Data Analysis and RFM Analysis
# This script performs EDA and creates customer RFM segmentation

# Load required libraries
library(dplyr)
library(ggplot2)
library(scales)
library(lubridate)

# Set working directory
setwd("/home/jd/crm-ca3")

# Create figures directory if not exists
if (!dir.exists("figures")) dir.create("figures")

# ============================================================
# 1. LOAD CLEANED DATA
# ============================================================
cat("Loading cleaned dataset...\n")
retail <- read.csv("data/online_retail_clean.csv", stringsAsFactors = FALSE)
retail$Date <- as.Date(retail$Date)
retail$InvoiceDate <- as.POSIXct(retail$InvoiceDate)

cat("Dataset loaded:", nrow(retail), "transactions\n")

# ============================================================
# 2. SUMMARY STATISTICS
# ============================================================
cat("\n========================================\n")
cat("SUMMARY STATISTICS\n")
cat("========================================\n")

cat("\nTotal Revenue: £", format(sum(retail$TotalRevenue), big.mark = ","), "\n")
cat("Total Transactions:", length(unique(retail$InvoiceNo)), "\n")
cat("Total Customers:", length(unique(retail$CustomerID)), "\n")
cat("Total Products:", length(unique(retail$StockCode)), "\n")
cat("Countries Served:", length(unique(retail$Country)), "\n")

# ============================================================
# 3. SALES BY COUNTRY
# ============================================================
cat("\n\nCreating country analysis...\n")

country_sales <- retail %>%
  group_by(Country) %>%
  summarise(
    TotalRevenue = sum(TotalRevenue),
    Transactions = n_distinct(InvoiceNo),
    Customers = n_distinct(CustomerID)
  ) %>%
  arrange(desc(TotalRevenue))

cat("\nTop 10 Countries by Revenue:\n")
print(head(country_sales, 10))

# Plot Top 10 Countries
top_countries <- head(country_sales, 10)
p1 <- ggplot(top_countries, aes(x = reorder(Country, TotalRevenue), y = TotalRevenue/1000)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 10 Countries by Revenue",
       x = "Country", y = "Revenue (£ thousands)") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))
ggsave("figures/01_top_countries_revenue.png", p1, width = 10, height = 6, dpi = 150)

# ============================================================
# 4. TIME SERIES ANALYSIS
# ============================================================
cat("\nCreating time series analysis...\n")

# Daily sales
daily_sales <- retail %>%
  group_by(Date) %>%
  summarise(
    Revenue = sum(TotalRevenue),
    Orders = n_distinct(InvoiceNo)
  )

p2 <- ggplot(daily_sales, aes(x = Date, y = Revenue/1000)) +
  geom_line(color = "steelblue", alpha = 0.7) +
  geom_smooth(method = "loess", color = "darkred", se = FALSE) +
  labs(title = "Daily Revenue Trend",
       x = "Date", y = "Revenue (£ thousands)") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))
ggsave("figures/02_daily_revenue_trend.png", p2, width = 12, height = 5, dpi = 150)

# Monthly sales
monthly_sales <- retail %>%
  mutate(YearMonth = format(Date, "%Y-%m")) %>%
  group_by(YearMonth) %>%
  summarise(
    Revenue = sum(TotalRevenue),
    Orders = n_distinct(InvoiceNo),
    Customers = n_distinct(CustomerID)
  )

p3 <- ggplot(monthly_sales, aes(x = YearMonth, y = Revenue/1000)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Monthly Revenue Distribution",
       x = "Month", y = "Revenue (£ thousands)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(hjust = 0.5, face = "bold"))
ggsave("figures/03_monthly_revenue.png", p3, width = 10, height = 6, dpi = 150)

# ============================================================
# 5. TOP PRODUCTS ANALYSIS
# ============================================================
cat("\nCreating product analysis...\n")

top_products <- retail %>%
  group_by(StockCode, Description) %>%
  summarise(
    TotalRevenue = sum(TotalRevenue),
    TotalQuantity = sum(Quantity),
    Transactions = n()
  ) %>%
  arrange(desc(TotalRevenue)) %>%
  head(15)

cat("\nTop 15 Products by Revenue:\n")
print(top_products)

p4 <- ggplot(head(top_products, 10), 
             aes(x = reorder(substr(Description, 1, 30), TotalRevenue), y = TotalRevenue/1000)) +
  geom_bar(stat = "identity", fill = "coral") +
  coord_flip() +
  labs(title = "Top 10 Products by Revenue",
       x = "Product", y = "Revenue (£ thousands)") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))
ggsave("figures/04_top_products.png", p4, width = 10, height = 6, dpi = 150)

# ============================================================
# 6. HOURLY SALES PATTERN
# ============================================================
cat("\nCreating hourly sales pattern...\n")

hourly_sales <- retail %>%
  group_by(Hour) %>%
  summarise(
    Revenue = sum(TotalRevenue),
    Orders = n_distinct(InvoiceNo)
  )

p5 <- ggplot(hourly_sales, aes(x = Hour, y = Orders)) +
  geom_bar(stat = "identity", fill = "darkgreen") +
  labs(title = "Orders by Hour of Day",
       x = "Hour", y = "Number of Orders") +
  scale_x_continuous(breaks = 0:23) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))
ggsave("figures/05_hourly_orders.png", p5, width = 10, height = 5, dpi = 150)

# ============================================================
# 7. RFM ANALYSIS
# ============================================================
cat("\n========================================\n")
cat("RFM ANALYSIS\n")
cat("========================================\n")

# Set reference date (day after last transaction)
reference_date <- max(retail$Date) + 1
cat("Reference date for Recency:", as.character(reference_date), "\n")

# Calculate RFM metrics per customer
rfm_data <- retail %>%
  group_by(CustomerID) %>%
  summarise(
    Recency = as.numeric(reference_date - max(Date)),  # Days since last purchase
    Frequency = n_distinct(InvoiceNo),                  # Number of orders
    Monetary = sum(TotalRevenue)                        # Total spend
  ) %>%
  ungroup()

cat("\nRFM Summary:\n")
print(summary(rfm_data[, c("Recency", "Frequency", "Monetary")]))

# Create RFM scores (1-5 scale, 5 is best)
rfm_data <- rfm_data %>%
  mutate(
    R_Score = ntile(-Recency, 5),   # Lower recency = higher score
    F_Score = ntile(Frequency, 5),
    M_Score = ntile(Monetary, 5)
  ) %>%
  mutate(
    RFM_Score = R_Score * 100 + F_Score * 10 + M_Score,
    RFM_Segment = paste0(R_Score, F_Score, M_Score)
  )

# Calculate average metrics per RFM score combination
rfm_summary <- rfm_data %>%
  group_by(R_Score, F_Score, M_Score) %>%
  summarise(
    Count = n(),
    Avg_Recency = round(mean(Recency), 1),
    Avg_Frequency = round(mean(Frequency), 1),
    Avg_Monetary = round(mean(Monetary), 2)
  ) %>%
  arrange(desc(R_Score), desc(F_Score), desc(M_Score))

cat("\nRFM Segments (Top 10 by Score):\n")
print(head(rfm_summary, 10))

# Create customer segments based on RFM
rfm_data <- rfm_data %>%
  mutate(
    Customer_Segment = case_when(
      R_Score >= 4 & F_Score >= 4 & M_Score >= 4 ~ "Champions",
      R_Score >= 3 & F_Score >= 3 & M_Score >= 3 ~ "Loyal Customers",
      R_Score >= 4 & F_Score <= 2 ~ "Promising",
      R_Score <= 2 & F_Score >= 4 ~ "At Risk",
      R_Score <= 2 & F_Score <= 2 ~ "Lost",
      TRUE ~ "Regular"
    )
  )

segment_summary <- rfm_data %>%
  group_by(Customer_Segment) %>%
  summarise(
    Count = n(),
    Percentage = round(n()/nrow(rfm_data)*100, 1),
    Avg_Recency = round(mean(Recency), 0),
    Avg_Frequency = round(mean(Frequency), 1),
    Avg_Monetary = round(mean(Monetary), 0),
    Total_Revenue = round(sum(Monetary), 0)
  ) %>%
  arrange(desc(Total_Revenue))

cat("\nCustomer Segments:\n")
print(segment_summary)

# RFM Visualization
p6 <- ggplot(segment_summary, aes(x = reorder(Customer_Segment, Count), y = Count, fill = Customer_Segment)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = paste0(Count, " (", Percentage, "%)")), hjust = -0.1) +
  coord_flip() +
  labs(title = "Customer Segmentation (RFM Analysis)",
       x = "Segment", y = "Number of Customers") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        legend.position = "none") +
  scale_fill_brewer(palette = "Set2")
ggsave("figures/06_rfm_segments.png", p6, width = 10, height = 6, dpi = 150)

# RFM Scatter plot: Frequency vs Monetary colored by Recency
p7 <- ggplot(rfm_data, aes(x = Frequency, y = Monetary/1000, color = Recency)) +
  geom_point(alpha = 0.5) +
  scale_color_gradient(low = "green", high = "red") +
  labs(title = "Customer Distribution: Frequency vs Monetary Value",
       x = "Frequency (Number of Orders)",
       y = "Monetary Value (£ thousands)",
       color = "Recency\n(Days)") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))
ggsave("figures/07_rfm_scatter.png", p7, width = 10, height = 7, dpi = 150)

# ============================================================
# 8. SAVE RFM DATA
# ============================================================
write.csv(rfm_data, "data/customer_rfm.csv", row.names = FALSE)
cat("\nRFM data saved to: data/customer_rfm.csv\n")

# ============================================================
# 9. CUSTOMER-LEVEL AGGREGATION FOR REGRESSION
# ============================================================
cat("\n========================================\n")
cat("CUSTOMER-LEVEL AGGREGATION\n")
cat("========================================\n")

# Create customer-level dataset for regression analysis
customer_data <- retail %>%
  group_by(CustomerID, Country) %>%
  summarise(
    TotalRevenue = sum(TotalRevenue),
    TotalOrders = n_distinct(InvoiceNo),
    TotalItems = sum(Quantity),
    TotalProducts = n_distinct(StockCode),
    AvgOrderValue = sum(TotalRevenue) / n_distinct(InvoiceNo),
    AvgItemPrice = mean(UnitPrice),
    AvgQuantityPerOrder = sum(Quantity) / n_distinct(InvoiceNo),
    FirstPurchase = min(Date),
    LastPurchase = max(Date),
    DaysSinceFirst = as.numeric(max(retail$Date) - min(Date)),
    DaysSinceLast = as.numeric(max(retail$Date) - max(Date))
  ) %>%
  ungroup()

# Add RFM scores
customer_data <- customer_data %>%
  left_join(rfm_data %>% select(CustomerID, R_Score, F_Score, M_Score, Customer_Segment),
            by = "CustomerID")

# Add country flag for UK vs International
customer_data$IsUK <- ifelse(customer_data$Country == "United Kingdom", 1, 0)

cat("Customer-level dataset:", nrow(customer_data), "customers\n")
cat("\nSummary:\n")
print(summary(customer_data[, c("TotalRevenue", "TotalOrders", "TotalItems", "AvgOrderValue")]))

# Save customer data
write.csv(customer_data, "data/customer_data.csv", row.names = FALSE)
cat("\nCustomer data saved to: data/customer_data.csv\n")

cat("\n\nEDA and RFM Analysis completed successfully!\n")
cat("Figures saved to: figures/\n")
