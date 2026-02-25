# CRM CA3 - Data Cleaning and Preparation
# This script loads, inspects, and cleans the Online Retail dataset

# Load required libraries
library(dplyr)
library(ggplot2)
library(lubridate)

# Set working directory
setwd("/home/jd/crm-ca3")

# ============================================================
# 1. LOAD DATA
# ============================================================
cat("Loading dataset...\n")
retail_raw <- read.csv("data/online_retail.csv", stringsAsFactors = FALSE)

cat("Dataset dimensions:", nrow(retail_raw), "rows,", ncol(retail_raw), "columns\n")
cat("\nColumn names:\n")
print(names(retail_raw))

cat("\nFirst few rows:\n")
print(head(retail_raw))

cat("\nData structure:\n")
str(retail_raw)

# ============================================================
# 2. DATA QUALITY ASSESSMENT
# ============================================================
cat("\n\n========================================\n")
cat("DATA QUALITY ASSESSMENT\n")
cat("========================================\n")

# 2.1 Check for duplicates
duplicates <- sum(duplicated(retail_raw))
cat("\n1. DUPLICATE ROWS:", duplicates, "\n")

# 2.2 Missing values analysis
cat("\n2. MISSING VALUES:\n")
missing_summary <- sapply(retail_raw, function(x) sum(is.na(x) | x == "" | x == " "))
print(missing_summary)

# Missing CustomerID
missing_customer_id <- sum(is.na(retail_raw$CustomerID) | retail_raw$CustomerID == "")
cat("\n   Missing CustomerID:", missing_customer_id, 
    "(", round(missing_customer_id/nrow(retail_raw)*100, 2), "%)\n")

# 2.3 Cancellation transactions (InvoiceNo starting with 'C')
cancellations <- sum(grepl("^C", retail_raw$InvoiceNo))
cat("\n3. CANCELLATION TRANSACTIONS:", cancellations, 
    "(", round(cancellations/nrow(retail_raw)*100, 2), "%)\n")

# 2.4 Negative/zero quantities
negative_qty <- sum(retail_raw$Quantity < 0, na.rm = TRUE)
zero_qty <- sum(retail_raw$Quantity == 0, na.rm = TRUE)
cat("\n4. QUANTITY ISSUES:\n")
cat("   Negative quantities:", negative_qty, "\n")
cat("   Zero quantities:", zero_qty, "\n")

# 2.5 Negative/zero prices
negative_price <- sum(retail_raw$UnitPrice < 0, na.rm = TRUE)
zero_price <- sum(retail_raw$UnitPrice == 0, na.rm = TRUE)
cat("\n5. PRICE ISSUES:\n")
cat("   Negative prices:", negative_price, "\n")
cat("   Zero prices:", zero_price, "\n")

# 2.6 Alien wordings in Description
blank_desc <- sum(retail_raw$Description == "" | is.na(retail_raw$Description))
cat("\n6. BLANK DESCRIPTIONS:", blank_desc, "\n")

# ============================================================
# 3. DATA CLEANING
# ============================================================
cat("\n\n========================================\n")
cat("DATA CLEANING\n")
cat("========================================\n")

# Start with raw data
retail_clean <- retail_raw

# Step 1: Remove duplicates
retail_clean <- retail_clean[!duplicated(retail_clean), ]
cat("After removing duplicates:", nrow(retail_clean), "rows\n")

# Step 2: Remove cancellation transactions
retail_clean <- retail_clean[!grepl("^C", retail_clean$InvoiceNo), ]
cat("After removing cancellations:", nrow(retail_clean), "rows\n")

# Step 3: Remove rows with missing CustomerID
retail_clean <- retail_clean[!is.na(retail_clean$CustomerID) & retail_clean$CustomerID != "", ]
cat("After removing missing CustomerID:", nrow(retail_clean), "rows\n")

# Step 4: Remove zero or negative quantities
retail_clean <- retail_clean[retail_clean$Quantity > 0, ]
cat("After removing non-positive quantities:", nrow(retail_clean), "rows\n")

# Step 5: Remove zero or negative prices
retail_clean <- retail_clean[retail_clean$UnitPrice > 0, ]
cat("After removing non-positive prices:", nrow(retail_clean), "rows\n")

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
cat("\n\n========================================\n")
cat("FEATURE ENGINEERING\n")
cat("========================================\n")

# Create TotalRevenue column
retail_clean$TotalRevenue <- retail_clean$Quantity * retail_clean$UnitPrice
cat("Created TotalRevenue column\n")

# Parse InvoiceDate
retail_clean$InvoiceDate <- as.POSIXct(retail_clean$InvoiceDate, format = "%m/%d/%Y %H:%M")
retail_clean$Date <- as.Date(retail_clean$InvoiceDate)
retail_clean$Month <- month(retail_clean$Date)
retail_clean$Year <- year(retail_clean$Date)
retail_clean$Weekday <- weekdays(retail_clean$Date)
retail_clean$Hour <- hour(retail_clean$InvoiceDate)
cat("Extracted date components: Date, Month, Year, Weekday, Hour\n")

# ============================================================
# 5. SUMMARY STATISTICS
# ============================================================
cat("\n\n========================================\n")
cat("CLEANED DATA SUMMARY\n")
cat("========================================\n")

cat("\nFinal dataset dimensions:", nrow(retail_clean), "rows,", ncol(retail_clean), "columns\n")

cat("\nNumeric summary:\n")
print(summary(retail_clean[, c("Quantity", "UnitPrice", "TotalRevenue")]))

cat("\nUnique values:\n")
cat("  Unique Invoices:", length(unique(retail_clean$InvoiceNo)), "\n")
cat("  Unique Customers:", length(unique(retail_clean$CustomerID)), "\n")
cat("  Unique Products:", length(unique(retail_clean$StockCode)), "\n")
cat("  Unique Countries:", length(unique(retail_clean$Country)), "\n")

cat("\nDate range:", as.character(min(retail_clean$Date, na.rm = TRUE)), 
    "to", as.character(max(retail_clean$Date, na.rm = TRUE)), "\n")

# ============================================================
# 6. SAVE CLEANED DATA
# ============================================================
write.csv(retail_clean, "data/online_retail_clean.csv", row.names = FALSE)
cat("\nCleaned data saved to: data/online_retail_clean.csv\n")

# ============================================================
# 7. DATA QUALITY ISSUES TABLE (for Report)
# ============================================================
cat("\n\n========================================\n")
cat("TABLE 1: DATA QUALITY ISSUES ADDRESSED\n")
cat("========================================\n")

issues_table <- data.frame(
  SN = c(1, 2, 3, 4, 5, 6),
  Issue = c(
    "Duplicate Records",
    "Missing CustomerID",
    "Cancellation Transactions",
    "Negative/Zero Quantities",
    "Zero Unit Prices",
    "Blank Descriptions"
  ),
  Count = c(
    duplicates,
    missing_customer_id,
    cancellations,
    negative_qty + zero_qty,
    zero_price,
    blank_desc
  ),
  Reason = c(
    "Duplicates can skew analysis and inflate metrics",
    "CustomerID required for customer-level analysis (RFM)",
    "Cancellations represent returned goods, not actual sales",
    "Invalid quantities cannot contribute to sales analysis",
    "Zero prices indicate free items or data entry errors",
    "Missing descriptions reduce data quality for product analysis"
  )
)

print(issues_table)

cat("\n\nData cleaning completed successfully!\n")
