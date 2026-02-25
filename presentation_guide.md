# Presentation Guide: CRM Analysis of Online Retail Data

This guide outlines the content, structure, and talking points for a 10-slide PowerPoint presentation based on your CRM analysis report. It ensures all CA3 assignment requirements are met, including the mandatory R models and data quality assessment.

---

## Slide 1: Title & Project Overview
**Goal:** Introduce the project scope and business objective.

*   **Title:** Optimizing Sales Strategy: Online Retail Transaction Analysis
*   **Subtitle:** Insights from Customer Relationship Management Data Mining
*   **Key Points:**
    *   **Objective:** Analyze transaction data to understand customer behavior and increase sales.
    *   **Dataset:** UK-based Online Retailer (Dec 2010 - Dec 2011).
    *   **Scope:** 541,909 raw transactions -> 397,884 clean records.
*   **Visuals:** 
    *   *Suggestion:* Use a clean, professional title slide layout.
    *   *Optional:* Screenshot of the first few rows of the dataset (use a tool like Snipping Tool on `head(retail)` output).

## Slide 2: Data Quality Assessment (Mandatory)
**Goal:** Demonstrate data cleaning and handling of issues (referencing Table 1 from requirements).

*   **Headline:** Ensuring Data Integrity for Reliable Analysis
*   **Key Points:**
    *   **Missing Customer IDs:** Removed 135k records (25%) - critical for customer tracking.
    *   **Cancellations:** Removed 9k returns (Invoice 'C') to avoid negative revenue.
    *   **Data Validity:** Removed negative quantities and zero prices.
    *   **Result:** High-quality dataset ready for modeling.
*   **Visuals:** 
    *   *Action:* Create a simple table in PowerPoint summarizing:
        | Issue | Count | Action Taken |
        |-------|-------|--------------|
        | Missing ID | 135,080 | Removed |
        | Cancellations | 9,288 | Removed |
        | Bad Data | 13,139 | Removed |

## Slide 3: Exploratory Data Analysis (Trends)
**Goal:** Show market context and seasonal trends.

*   **Headline:** Market Insights & Seasonal Trends
*   **Key Points:**
    *   **Seasonality:** Revenue peaks significantly in **November** (Pre-holiday).
    *   **Top Markets:** UK is dominant (82%), but EIRE/Netherlands have high potential.
*   **Visuals:** 
    *   `figures/03_monthly_revenue.png` (Shows the clear November peak)
    *   `figures/01_top_countries_revenue.png` (Shows market distribution)

## Slide 4: Customer Segmentation (RFM)
**Goal:** Explain how customers were grouped using Recency, Frequency, and Monetary scores.

*   **Headline:** Unlocking Value with RFM Segmentation
*   **Key Points:**
    *   **Pareto Principle:** **Champions & Loyal** (39% of customers) = **81% of Revenue**.
    *   **Risk:** **31%** are "Lost" or "At Risk".
*   **Visuals:** 
    *   `figures/06_rfm_segments.png` (The most important chart for this section)
    *   `figures/07_rfm_scatter.png` (Optional: shows the spread of customers)

## Slide 5: K-Means Clustering (Advanced Tech)
**Goal:** Show unsupervised learning results (Requirement: Clustering Techniques).

*   **Headline:** Identifying the "VIP" Signature via Clustering
*   **Key Points:**
    *   **Optimal K:** Elbow method showed **K=4** clusters.
    *   **Quality:** Silhouette Score **0.616** indicates strong separation.
    *   **Insight:** **VIP Cluster** (5.1%) generates **48% of revenue**.
*   **Visuals:** 
    *   `figures/13_elbow_plot.png` (Proof of methodology)
    *   `figures/17_kmeans_clusters.png` (Visualizes the distinct groups)
    *   *Optional:* `figures/18_silhouette.png` (If asked about validation)

## Slide 6: Linear Regression Model (Mandatory)
**Goal:** Quantify the relationship between orders and revenue.

*   **Headline:** Revenue Drivers: Linear Regression Analysis
*   **Key Points:**
    *   **Model:** `Revenue ~ Orders`
    *   **Result:** Strong positive correlation ($R^2 = 0.307$).
    *   **Impact:** Each new order adds ~£646 to customer value.
*   **Visuals:** 
    *   `figures/08_linear_regression.png` (Shows the trend line)

## Slide 7: Multiple Regression & VIF (Mandatory)
**Goal:** A more robust model to predict revenue and check for multicollinearity.

*   **Headline:** Drivers of Revenue: Multivariate Analysis
*   **Key Points:**
    *   **Model Fit:** Excellent ($R^2 = 0.855$).
    *   **Key Driver:** **Total Items** is the #1 predictor.
    *   **VIF Check:** All values < 2.5 (Pass). **No Multicollinearity**.
*   **Visuals:** 
    *   `figures/11_vif_analysis.png` (Evidence of valid model)
    *   `figures/10_coefficient_plot.png` (Shows which factors matter most)

## Slide 8: Prediction Capabilities
**Goal:** Explicitly address "What can/cannot be predicted" (Assignment Requirement).

*   **Headline:** Predictive Boundaries
*   **Key Points:**
    *   **CAN Predict:** Customer Value (CLV), Seasonal Peaks, Segment Migration.
    *   **CANNOT Predict:** New customer acquisition (no prospect data), specific product taste (no browsing data).
    *   **Accuracy:** Model predicts revenue well (see chart).
*   **Visuals:** 
    *   `figures/12_actual_vs_predicted.png` (Shows model accuracy)

## Slide 9: Business Problem & Opportunity
**Goal:** Synthesize findings into a clear business case.

*   **Headline:** The Retention Challenge & Opportunity
*   **Key Points:**
    *   **Problem:** High churn in "Lost" segment (25%).
    *   **Opportunity:** VIPs are hugely valuable but rare.
    *   **Strategy:** Move "Active" -> "VIP" and reactivate "At Risk".
*   **Visuals:** 
    *   `figures/14_kmeans_distribution.png` (Shows the size of the "Churned" vs "VIP" clusters)
    *   `figures/16_cluster_profile.png` (Comparison of cluster behaviors)

## Slide 10: Strategic Recommendations
**Goal:** Provide actionable steps to increase sales.

*   **Headline:** Strategies for Growth
*   **Key Points:**
    *   **1. VIP Concierge:** Lock in the top 5% (48% of revenue).
    *   **2. Win-Back:** Automated offers for "At Risk" (>100 days inactive).
    *   **3. Bulk Incentives:** Drive `TotalItems` to boost revenue.
    *   **4. Q4 Push:** Prepare for November peak.
*   **Visuals:** 
    *   *Action:* Create a simple checklist or roadmap graphic in PowerPoint.
