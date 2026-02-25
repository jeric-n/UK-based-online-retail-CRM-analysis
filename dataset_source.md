https://www.kaggle.com/datasets/thedevastator/online-retail-transaction-data

# example download command

#!/bin/bash
curl -L -o ~/Downloads/online-retail-transaction-data.zip\
  https://www.kaggle.com/api/v1/datasets/download/thedevastator/online-retail-transaction-data

# About Dataset

Online Retail Transaction Data
UK Online Retail Sales and Customer Transaction Data


    Comprehensive Dataset on Online Retail Sales and Customer Data

    Welcome to this comprehensive dataset offering a wide array of information related to online retail sales. This data set provides an in-depth look at transactions, product details, and customer information documented by an online retail company based in the UK. The scope of the data spans vastly, from granular details about each product sold to extensive customer data sets from different countries.

    This transnational data set is a treasure trove of vital business insights as it meticulously catalogues all the transactions that happened during its span. It houses rich transactional records curated by a renowned non-store online retail company based in the UK known for selling unique all-occasion gifts. A considerable portion of its clientele includes wholesalers; ergo, this dataset can prove instrumental for companies looking for patterns or studying purchasing trends among such businesses.

    The available attributes within this dataset offer valuable pieces of information:

        InvoiceNo: This attribute refers to invoice numbers that are six-digit integral numbers uniquely assigned to every transaction logged in this system. Transactions marked with 'c' at the beginning signify cancellations - adding yet another dimension for purchase pattern analysis.

        StockCode: Stock Code corresponds with specific items as they're represented within the inventory system via 5-digit integral numbers; these allow easy identification and distinction between products.

        Description: This refers to product names, giving users qualitative knowledge about what kind of items are being bought and sold frequently.

        Quantity: These figures ascertain the volume of each product per transaction – important figures that can help understand buying trends better.

        InvoiceDate: Invoice Dates detail when each transaction was generated down to precise timestamps – invaluable when conducting time-based trend analysis or segmentation studies.

        UnitPrice: Unit prices represent how much each unit retails at — crucial for revenue calculations or cost-related analyses.

    Finally,

        Country: This locational attribute shows where each customer hails from, adding geographical segmentation to your data investigation toolkit.

    This dataset was originally collated by Dr Daqing Chen, Director of the Public Analytics group based at the School of Engineering, London South Bank University. His research studies and business cases with this dataset have been published in various papers contributing to establishing a solid theoretical basis for direct, data and digital marketing strategies.

    Access to such records can ensure enriching explorations or formulating insightful hypotheses about consumer behavior patterns among wholesalers. Whether it's managing inventory or studying transactional trends over time or spotting cancellation patterns - this dataset is apt for multiple forms of retail analysis

How to use the dataset

    1. Sales Analysis:

    Sales data forms the backbone of this dataset, and it allows users to delve into various aspects of sales performance.
    You can use the Quantity and UnitPrice fields to calculate metrics like revenue, and further combine it with InvoiceNo information to understand sales over individual transactions.
    2. Product Analysis:

    Each product in this dataset comes with its unique identifier (StockCode) and its name (Description). You could analyse which products are most popular based on Quantity sold or look at popularity per transaction by considering both Quantity and InvoiceNo.
    3. Customer Segmentation:

    If you associated specific business logic onto the transactions (such as calculating total amounts), then you could use standard machine learning methods or even RFM (Recency, Frequency, Monetary) segmentation techniques combining it with 'CustomerID' for your customer base to understand customer behavior better.
    Concatenating invoice numbers (which stand for separate transactions) per client will give insights about your clients as well.
    4. Geographical Analysis:

    The Country column enables analysts to study purchase patterns across different geographical locations.
    Practical applications

    Understand what products sell best where - It can help drive tailored marketing strategies.
    Anomalies detection – Identify unusual behaviors that might lead fraud investigations.
    Forecast Demand - Building time-series models can aid in predicting future sales.
    Promotional Strategy - Seeing what sells together frequently may help design product bundles or suggest products.
    Some useful tips

        Understand meaning behind each field:
        Each field describes an aspect of each transaction allowing us access analysis from multiple dimensions
        Cleaning data:
        Take care while using fields like Description since they are nominal fields and may require further cleaning to avoid any errors during an analysis.
        Exploring correlations with classification and regression algorithms can be instrumental in revealing complex relations.

    In conclusion, this dataset is a Pandora's box - teeming with insights. Talented data analysts can use it to extract meaningful data stories that drive valuable business decisions

Research Ideas

        Inventory Management: By analyzing the quantity and frequency of product sales, retailers can effectively manage their stock and predict future demand. This would help ensure that popular items are always available while less popular items aren't overstocked.

        Customer Segmentation: Data from different countries can be used to understand buying habits across different geographical locations. This will allow the retail company to tailor its marketing strategy for each specific region or country, leading to more effective advertising campaigns.

        Sales Trend Analysis: With data spanning almost a year, temporal patterns in purchasing behavior can be identified, including seasonality and other trends (like increase in sales during holidays). Techniques like time-series analysis could provide insights into peak shopping times or days of the week when sales are typically high.

        Predictive Analysis for Cross Selling & Upselling: Based on a customer's previous purchase history, predictive algorithms can be utilized to suggest related products which might interest the customer, enhancing upsell and cross-sell opportunities.

        Detecting Fraud: Analysing sale returns (marked with 'c' in InvoiceNo) across customers or regions could help pinpoint fraudulent activities or operational issues leading to those returns

Acknowledgements

    If you use this dataset in your research, please credit the original authors.
    Data Source https://data.world/uci

Columns

File: online_retail.csv
Column name 	Description
InvoiceNo 	A 6-digit number uniquely assigned to each transaction. If the number is prefixed with 'c', it indicates a cancellation. (Nominal)
StockCode 	A unique identifier for each product sold by the retailer. (Nominal)
Description 	The name or a brief description of the product. (Nominal)
Quantity 	The number of units of the product sold in each transaction. (Numeric)
InvoiceDate 	The date and time when the transaction was made. (Datetime)
UnitPrice 	The price per unit of the product in sterling. (Numeric)
Country 	The country where the customer resides. (Nominal)
