# UK-Based Online Retail CRM Analysis

CRM and customer analytics project using a UK online retail transaction dataset (Dec 2010 to Dec 2011). The goal is to explain customer behavior and identify practical ways to increase sales using segmentation and predictive modeling.

## Project Task
- Assess and clean transactional retail data for CRM use
- Segment customers using RFM and K-Means clustering
- Build regression models to understand revenue drivers
- Convert analysis outputs into actionable business recommendations

## Dataset Summary
- Source: Online Retail Transaction Data (Kaggle)
- Raw records: 541,909
- Cleaned records: 397,884
- Unique customers: 4,338
- Countries: 37
- Time window: December 2010 to December 2011

## Methods Used
- Data quality assessment and treatment
- Exploratory Data Analysis (EDA)
- RFM segmentation (Recency, Frequency, Monetary)
- K-Means clustering (K=4 via elbow method)
- Linear and multiple regression with VIF checks

## Key Findings
- Champions + Loyal customers (39.4% of customers) contribute about 80.7% of revenue
- K-Means identified a small high-value group:
  - Elite VIP + VIP customers (5.1%) contribute about 47.9% of total revenue
- Cluster quality was strong (silhouette score: 0.616)
- Simple regression (`Revenue ~ Orders`) shows positive association (`R² ≈ 0.307`)
- Multiple regression model fit is high (`R² ≈ 0.855`) with acceptable multicollinearity levels (VIF checks)

## Business Recommendations
- Prioritize retention and service for VIP/Elite segments
- Run targeted win-back campaigns for At Risk / Lost segments
- Plan inventory and campaigns around seasonal peak (notably November)
- Encourage higher basket size and repeat purchases for Active customers

## Repository Structure
- `analysis/`: R scripts for the end-to-end pipeline
  - `01_data_cleaning.R`
  - `02_eda_rfm.R`
  - `03_regression_corrected.R`
  - `04_kmeans_clustering.R`
- `data/`: source and processed datasets
- `figures/`: generated charts and model visuals
- `regression_outputs/`: regression artifacts and summaries
- `report.md`: full written report
- `presentation_guide.md`: slide-by-slide presentation outline

## How To Run
Prerequisites:
- R (recommended 4.2+)
- R packages: `dplyr`, `ggplot2`, `lubridate`, `scales`, `car`, `MASS`, `cluster`, `factoextra`, `tidyr`

Run pipeline from project root:

```bash
Rscript analysis/01_data_cleaning.R
Rscript analysis/02_eda_rfm.R
Rscript analysis/03_regression_corrected.R
Rscript analysis/04_kmeans_clustering.R
```

Outputs are written to `data/`, `figures/`, and `regression_outputs/`.

## Notable Visuals
- `figures/06_rfm_segments.png`
- `figures/13_elbow_plot.png`
- `figures/17_kmeans_clusters.png`
- `figures/08_linear_regression.png`
- `figures/12_actual_vs_predicted.png`

## Notes
- Large files are present in this repository (for example, some CSVs). GitHub may warn on files over recommended size limits.
- `.gitignore` excludes local environment files, presentation files (`*.pptx`), and other non-essential artifacts.
