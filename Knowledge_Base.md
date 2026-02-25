# Knowledge Base: CRM and Data Mining Lessons

This document contains summaries of core concepts extracted from various CRM and Data Mining lessons.

---

## Lesson 1: Introductory Lesson on R

### Core Concepts
- **R Programming Language**: A high-level programming language used for statistical analysis and data visualization. It is a dialect of the S language.
- **R System Structure**: Comprises the base R system and additional packages available through CRAN (Comprehensive R Archive Network).
- **RStudio**: An integrated development environment (IDE) for R, featuring panes for scripting, console execution, environment management, and file/package browsing.
- **Basic Syntax**: 
    - **Assignment**: Uses `<-` (e.g., `x <- 10`).
    - **Case Sensitivity**: R is case-sensitive (e.g., `X` and `x` are different).
    - **Comments**: Prefixed with `#`.
    - **Functions**: Pieces of code written to perform specific tasks (e.g., `c()` for combining elements, `seq()` for sequences, `rep()` for repetitions).

### Key Takeaways
- R is widely used for data entry, pre-processing, visualization, and statistical modeling/machine learning.
- Efficient coding in R involves using scripts to save and run commands sequentially.
- Help for any function can be accessed using `?function_name` in the console.

**Source File**: `Lessons/Lesson 1 - R Basic.pdf`

---

## Lesson 1: Data Mining Applications in Marketing and Customer Relationship Management

### Core Concepts
- **Customer Relationship Management (CRM)**: A strategic approach focusing on managing interactions with current and potential customers to improve relationships and drive growth.
- **Data Mining**: The process of analyzing large datasets to uncover meaningful patterns, trends, and rules to inform business decisions.
- **Customer Lifecycle**:
    1. **Prospects**: Potential customers in the target market.
    2. **Responders**: Prospects who show interest.
    3. **New Customers**: Responders who make a first purchase.
    4. **Established Customers**: Customers with broadening or deepening relationships.
    5. **Former Customers**: Customers who have left (churned).
- **Customer Attrition (Churn)**:
    - **Voluntary**: Customer chooses to leave.
    - **Involuntary**: Company terminates the relationship (e.g., unpaid bills).
    - **Expected**: Customer is no longer in the target market.
- **Data Mining Approaches**:
    - **Hypothesis Testing**: Validating proposed explanations through data.
    - **Directed Data Mining**: Building models to predict or explain specific target variables (e.g., will a customer churn?).
    - **Undirected Data Mining**: Finding overall patterns without a predefined target (e.g., clustering).

### Key Takeaways
- Every stage of the customer lifecycle (Acquisition, Activation, CRM, Winback) offers opportunities for data mining.
- Predictive modeling helps refine targeting in acquisition and improve efficiency in retention.
- The goal of data mining in CRM is to transform data into actionable information that supports business decision-making.
- Data mining must be a continuous learning cycle: Identify Opportunity -> Transform Data -> Act on Information -> Measure Results.

**Source File**: `Lessons/Lesson 1 Data Mining Applications in Marketing and Customer Relationship Management.pdf`

---

## Lesson 2: Introductory Lesson on R (Part 2)

### Core Concepts
- **Data Frame**: A list of vectors of the same length, structured like a table with rows and columns.
    - `data.frame()`: Combines vectors into a data frame.
    - `str()`, `names()`, `summary()`: Functions to inspect the structure, variable names, and basic statistics of a data frame.
    - `df[row, col]`: Selecting specific data points or rows/columns.
    - `rbind()`: Adds a new row to an existing data frame.
- **Matrix**: A 2D rectangular layout of elements of the same type.
    - `matrix(data, nrow, ncol)`: Creates a matrix.
    - `rownames()`, `colnames()`: Names the dimensions.
    - `t()`: Transposes the matrix.
- **Array**: Multi-dimensional R objects that can store data in more than two dimensions.
    - `array(data, dim)`: Creates an array (e.g., `dim=c(3,4,2)` for 3 rows, 4 columns, across 2 matrices).
- **Factors**: Used for categorical variables with a limited set of values (levels).
    - `factor()`: Creates a factor.
    - `ordered=TRUE`: Creates an ordered factor for comparisons (e.g., low < medium < high).
- **Conditional Statements**:
    - `ifelse(test, x, y)`: Returns `x` if the test is true, else `y`.
    - **Nested ifelse**: Used for multiple conditions.
- **Data Loading**: `read.csv()` is used to load data from CSV files into R.

### Key Takeaways
- Data frames are the most common structure for data analysis in R.
- Handling missing values (`NA`) is crucial, often using `na.rm=TRUE` in statistical functions.
- Factors allow R to handle categorical data correctly in models.

**Source File**: `Lessons/Lesson 2 - R Basic.pdf`

---

## Lesson 2: Profiling and Predictive Modelling to Explain Customer Behaviour

### Core Concepts
- **Directed Data Mining**: Constructing models to infer or predict a specific target variable (e.g., response likelihood).
- **Uplift Modelling (Incremental Response)**: Estimates the net impact of a marketing message. Identifies four customer groups:
    - **Sure Things**: Will respond anyway.
    - **Persuadable**: Will respond only if contacted (The primary target).
    - **Lost Causes**: Will not respond regardless.
    - **Sleeping Dogs**: May respond negatively if contacted.
- **Model Types**:
    - **Prediction Model**: Uses historical data to predict future outcomes (Inputs from before the target timeframe).
    - **Profiling Model**: Describes current characteristics (Inputs and target from the same timeframe).
- **11-Step Methodology**:
    1. Translate business problem to data mining problem.
    2. Select appropriate data (Data warehouse, size, timeframes).
    3. Get to know the data (EDA, visualization).
    4. Create a model set (Customer signatures, Partitioning: Train/Validation/Test).
    5. Fix data problems (Missing values, outliers, inconsistent encoding).
    6. Transform data (Create derived fields).
    7. Build models (Training).
    8. Assess models (Accuracy, Lift, Profitability, ROC).
    9. Deploy models (Scoring environment).
    10. Assess results (ROI, Treatment group comparison).
    11. Begin again (Iterative learning).
- **Assessment Tools**:
    - **Lift Chart**: Measures model performance against random selection.
    - **ROC Chart**: Plots True Positive Rate vs. False Positive Rate; independent of class distribution.
    - **Profitability Curve**: Evaluates financial impact of targeting based on model scores.

### Key Takeaways
- Model stability is as important as performance; it must work on unseen data.
- Creating "Customer Signatures" (one row per customer) is a key step in building model sets.
- Data transformation (Step 6) is often where the most significant insights are "brought to the surface."

**Source File**: `Lessons/Lesson 2 Profiling and Predictive Modelling to Explain Customer Behaviour.pdf`

---

## Lesson 3: Introductory Lesson on R (Part 3)

### Core Concepts
- **Data Cleaning (Missing Data)**:
    - `is.na(df$col)`: Identifies missing values (`NA`).
    - `which(is.na(df$col))`: Returns indices of rows with `NA`.
    - **Imputation**: Replacing `NA` with the mean (e.g., `df$col[is.na(df$col)] <- mean(df$col, na.rm=TRUE)`).
- **Data Visualization Gears**:
    - **Base R**: Quick standard graphs.
    - **Lattice**: Scientific publications.
    - **ggplot2**: Consistent grammar of graphics, highly flexible, and standard for data visualization.
- **ggplot2 Syntax**:
    - `ggplot(data, aes(x=var1, y=var2, col=var3, fill=var4))` + `geom_THING()`
    - **Aesthetics (aes)**: Defines what variables map to axes, colors, and fills.
    - **Geometries (geom)**:
        - `geom_bar()`: Bar charts for discrete/categorical variables.
        - `geom_point()`: Scatterplots for numeric variables.
        - `geom_line()` / `geom_smooth()`: Trends and fitted lines.
        - `geom_boxplot()`: Visualizing distributions and medians.
- **Reproducibility**:
    - `set.seed(number)`: Ensures random processes (like sampling) produce the same results every time.
    - `sample(nrow(df), size)`: Takes a random sample of a specific size from the dataset.

### Key Takeaways
- `ggplot2` plots are built in layers using the `+` operator.
- Cleaning missing data is essential as many functions return `NA` if input data contains missing values.
- `set.seed()` is critical for collaborative data science to ensure all team members see the same results from random operations.

**Source File**: `Lessons/Lesson 3 - R Basic.pdf`

---

## Lesson 3: Data Mining Techniques to Understand The Customer – Part 1

### Core Concepts
- **Similarity Models**: Scores candidates based on their distance from a "Prototype" or "Ideal Case."
    - **Distance Function**: Typically Euclidean distance using standardized (z-scores) variables.
    - **Application**: Identifying new store locations or prospective readers by matching them to high-performing profiles.
- **Table Lookup Models**: Divides data into cells based on input dimensions; all records in a cell receive the same score (e.g., average response rate).
- **RFM Model (Recency, Frequency, Monetary)**:
    - **Recency**: Time since last purchase.
    - **Frequency**: Number of purchases in a given period.
    - **Monetary**: Total value spent.
    - **Logic**: Recent, frequent, and high-spending customers are most likely to respond to future offers.
- **Naïve Bayesian Models**: Estimates the odds of an event by multiplying overall odds by the likelihoods of individual, independent attributes. Better than table lookup for handling many inputs.
- **Linear Regression**: Fits a straight line to numeric variables by minimizing the sum of squared residuals (Least Squares).
    - **Goodness of Fit ($R^2$)**: The proportion of variance in the target explained by the model.
    - **Residuals**: The difference between predicted and actual values; should be unbiased and patternless.
- **Multiple Regression**: Uses several input variables to predict a target.
    - **Linear Independence**: Inputs should not be highly correlated (avoids multicollinearity).
    - **Variable Selection**: Forward selection, Stepwise, and Backward elimination.
- **Logistic Regression**: Designed for binary outcomes (e.g., Churn vs. Stay). Uses an S-shaped logistic curve to ensure predicted probabilities stay between 0 and 1.

### Key Takeaways
- RFM is a powerful, simple technique for harvesting existing customer value but not for new acquisition.
- Linear regression captures "global" patterns but may fail if relationships vary across local subgroups.
- Naïve Bayes assumes independence between variables, which simplifies calculation but may miss interactions.

**Source File**: `Lessons/Lesson 3 Data Mining Techniques to Understand The Customer – Part 1.pdf`

---

## Lesson 4: Introductory Lesson on R (Part 4)

### Core Concepts
- **Regression Analysis**: Statistical tool to characterize relationships between a dependent variable (Y) and independent variables (X).
- **Correlation**:
    - `cor(x, y)`: Measures the association between two variables.
- **Linear Regression**:
    - `lm(Y ~ X, data=df)`: Builds a simple linear model.
    - **Interpreting Output**:
        - **Significance of F**: Confirms the validity of the regression (Accept if < 0.05).
        - **Multiple R-squared**: Percentage of variation in Y explained by X.
        - **Adjusted R-squared**: Used for multiple predictors; adjusts for the number of variables.
        - **Coefficients**: Define the equation $\hat{y} = mx + c$.
        - **P-value**: Significance of each predictor (Accept if < 0.05).
- **Multiple Regression**:
    - `lm(Y ~ X1 + X2 + ..., data=df)`: Uses two or more predictors.
    - **stepAIC()**: Automatically selects the best model by minimizing the Akaike Information Criterion (AIC).
    - **Multicollinearity**: Occurs when predictors are correlated with each other. Check using **Variance Inflation Factor (VIF)**; values > 10 indicate high multicollinearity.
- **Categorical Variables**:
    - **Nominal**: No intrinsic order (e.g., gender, weather).
    - **Ordinal**: Clear ordering (e.g., shirt sizes, economic status).
    - **Dummy Coding**: R automatically converts factors into binary dummy variables (0 and 1) for regression.

### Key Takeaways
- Always check the Significance of F before interpreting other parts of a regression model.
- Multicollinearity can make coefficients unreliable; use `vif()` to detect it.
- Categorical data must be converted to factors for R to perform automatic dummy coding.

**Source File**: `Lessons/Lesson 4 - R Basic.pdf`

---

## Lesson 4: Data Mining Techniques to Understand The Customer – Part 2

### Core Concepts
- **Decision Trees**: Hierarchical rules that recursively split data into subsets with increasing purity of the target variable.
    - **Structure**: Root Node (all data), Branches (split rules), Leaf Nodes (final decisions/rules).
    - **Purity Measures**: Gini Index, Entropy (Information Gain), Chi-square.
    - **Regression Tree**: Estimates numeric values by minimizing target variance within leaves.
    - **Pruning**: Removing branches that overfit the training data to ensure the model generalizes to new data.
    - **Local Models**: Unlike global regression, trees segment the input space, handling diverse paths to the same outcome.
- **Artificial Neural Networks (ANN)**:
    - **Multi-Layer Perceptron (MLP)**: Input layer, Hidden layer(s), and Output layer.
    - **Artificial Neuron**: Combines weighted inputs and applies a nonlinear transfer function (e.g., S-shaped curve).
    - **Backpropagation**: An iterative process of adjusting weights by propagating prediction errors backward through the network.
    - **Opaqueness**: Neural networks are "black boxes"; sensitivity analysis (varying inputs to see output change) is used to assess feature importance.
    - **Pruning**: Removing unnecessary neurons/weights to prevent overfitting.

### Key Takeaways
- Decision trees are self-documenting, producing rules that are easy to translate into natural language or SQL.
- Neural networks are highly flexible but require extensive data preparation (standardization) and do not explain *why* a prediction was made.
- "Don't make it [the network] discover what you already know"—incorporate domain knowledge into data preparation.

**Source File**: `Lessons/Lesson 4 Data Mining Techniques to Understand The Customer – Part 2.pdf`

---

## Lesson 5: Introductory Lesson on R (Part 5)

### Core Concepts
- **Clustering**: An unsupervised learning technique used to organize data into meaningful groups (clusters) without predefined targets.
    - **Goal**: Maximize intra-cluster similarity and minimize inter-cluster similarity.
    - **Euclidean Distance**: The standard measure for dissimilarity between observations with continuous variables.
- **K-means Model**:
    - `kmeans(x, centers=k)`: Partitions data into $k$ pre-specified clusters.
    - **Centroid**: The center of a cluster (mean of all points in that cluster).
    - **Elbow Method**: A technique to find the optimal $k$ by plotting the Within Sum of Squares (WSS) and identifying the "bend" or knee.
- **Classification**: A supervised learning technique that learns from prior examples to categorize new data.
- **KNN (K-Nearest Neighbors)**:
    - **Lazy Learner**: Does not learn a model; it stores training data and classifies based on the nearest neighbors at prediction time.
    - **K Choice**: Often the square root of the sample size (typically an odd number).
    - **Data Normalization**: Essential for KNN to ensure variables with larger scales do not dominate the distance calculation.
    - `knn(train, test, cl, k)`: Performs classification.

### Key Takeaways
- K-means is unsupervised (targets unknown), while KNN is supervised (targets known).
- Evaluating clusters can be done using `table()`, `confusionMatrix()`, or `CrossTable()`.
- Data preparation for KNN must include normalization (scaling data between 0 and 1).

**Source File**: `Lessons/Lesson 5 - R Basic.pdf`

---

## Lesson 5: Fraud Detection using Memory-based Reasoning (MBR)

### Core Concepts
- **Memory-Based Reasoning (MBR)**: Solving new problems by identifying and applying knowledge from similar historical cases.
    - **Distance Function**: Measures how similar two records are.
    - **Combination Function**: Aggregates the results from the identified neighbors (e.g., voting).
- **Look-Alike Models**: The simplest form of MBR; predicts the target value for a new record based on its single nearest neighbor in the training set.
- **Performance Optimization**:
    - **R-Tree Indexing**: A grid-based indexing method to speed up neighbor searches in large, multi-dimensional datasets.
    - **Reducing Training Set**: Identifying clusters within categories to reduce the number of comparison points without losing accuracy.
- **Combination Functions**:
    - **Voting ("Democracy")**: Each neighbor gets one vote for its class.
    - **Weighted Voting**: Votes from closer neighbors are given more weight (inverse distance weighting).
- **Collaborative Filtering**: A variant of MBR for personalized recommendations (e.g., Netflix, Amazon).
    - **User Profiles**: Vectors representing user ratings for various items.
    - **Steps**: 1) Build Profile, 2) Compare Profiles (Euclidean, Cosine similarity, or Pearson correlation), 3) Predict Ratings.

### Key Takeaways
- MBR is highly adaptable as it learns continuously with new data but is computationally expensive for large datasets.
- "Obscure" items in collaborative filtering often provide better insights into user preferences than popular items.
- MBR for numeric targets ensures predictions stay within realistic ranges (no negative values if none exist in training data).

**Source File**: `Lessons/Lesson 5 Fraud Detection using Memory-based Reasoning (MBR).pdf`

---

## Lesson 6: Introductory Lesson on R (Part 6)

### Core Concepts
- **Text Analytics**: The process of analyzing and extracting insights from qualitative, unstructured text data.
- **Key R Packages**:
    - **tm**: Core package for text mining operations (e.g., `Corpus()`, `tm_map()`).
    - **SnowballC**: Used for **Stemming**—reducing words to their root form (e.g., "fishing" to "fish").
    - **wordcloud**: Visualizes word frequency; size represents prominence.
    - **syuzhet**: Calculates sentiment scores and extracts emotions.
- **Text Pre-processing Steps**:
    1. **Lower Case Conversion**: Standardize text.
    2. **Stop Word Removal**: Filtering out common words with little value (e.g., "the", "is").
    3. **Punctuation/Number Removal**: Cleaning the text of noise.
    4. **Stemming**: Consolidating word variations.
    5. **Term-Document Matrix (TDM)**: A table showing the frequency of each word across documents.
- **Sentiment & Emotion Analysis**:
    - **get_sentiment()**: Returns a score from -1 (Negative) to +1 (Positive).
    - **get_nrc_sentiment()**: Classifies text into 8 basic emotions (anger, fear, anticipation, trust, surprise, sadness, joy, disgust) and two sentiments.

### Key Takeaways
- Text mining requires significant "cleaning" before it can be used for analysis.
- Term-Document Matrices allow qualitative data to be analyzed using quantitative methods (e.g., word frequencies, bar plots).
- `findAssocs()` identifies words that are statistically correlated, helping uncover themes in customer feedback.

**Source File**: `Lessons/Lesson 6 - R Basic.pdf`

---

## Lesson 6: Customer Retention Using Survival Analysis

### Core Concepts
- **Survival Analysis (Time-to-Event Analysis)**: Predicting *when* a customer event (churn, migration, or purchase) will occur.
- **Survival Curve ($S(t)$)**:
    - Represents the probability that a customer remains active at tenure $t$.
    - Starts at 100% and declines over time; never increases.
    - **Customer Half-Life**: The median lifetime (when 50% of customers remain).
    - **Average Tenure**: The area under the survival curve.
- **Hazard Probability ($h(t)$)**: The likelihood that a customer will churn in the next interval, given they have stayed up to time $t$.
    - **Constant Hazard**: Risk is independent of tenure (rare).
    - **Bathtub-Shaped Hazard**: High risk early and late, with a stable middle (common in contract-based services).
- **Censoring**: Handling data where the "event" has not occurred yet (active customers) or was caused by a competing risk (forced attrition).
    - **Forced Attrition**: Customers who are terminated (e.g., non-payment) should be **censored** when modeling voluntary churn to avoid biasing the results.
- **Cox Proportional Hazards**: A model used to quantify the relative impact of various factors (covariates) on the survival of a customer.
- **Customer Value**: Expected Relationship Duration × Revenue per Period.
    - For existing customers, value is based on **Conditional Survival** ($S(t_{now} + t_{future}) / S(t_{now})$).

### Key Takeaways
- Survival analysis is more stable and comprehensive than simple retention curves because it uses data from all customers, regardless of their start date.
- It enables precise financial forecasting by estimating the expected remaining lifetime of each customer.
- Reactivation timing (win-back) can also be modeled as an inverse survival curve (1-survival).

**Source File**: `Lessons/Lesson 6 Customer Retention Using Survival Analysis.pdf`

---

## Lesson 7: Uncovering Customer Insights Using Pattern Discovery and Data Mining

### Core Concepts
- **Undirected Data Mining**: Exploration of data without a predefined target variable. It requires significant human judgment and domain expertise to interpret findings.
- **Key Approaches**:
    - **Exploratory Analysis**: Uncovering trends, distributions, and anomalies.
    - **Clustering**: Grouping similar records based on characteristics.
    - **Monte Carlo Simulation**: Repeated random sampling from known distributions to evaluate risks and potential outcomes (e.g., campaign profitability).
    - **Agent-Based Modelling**: Bottom-up simulation where autonomous "agents" (e.g., customers) interact according to specific rules.
- **Defining "Hidden" Target Variables**: When a label is missing (e.g., "profitability"), a structured approach is used: Identify dimensions -> Create metrics -> Validate -> Normalize -> Combine into a composite indicator.
- **Solar System Clusters**: A common pattern where a large central cluster (typical data) is surrounded by smaller, "orbiting" clusters that often represent high-value segments or anomalies (e.g., fraud).

### Key Takeaways
- Undirected mining is "hypothesis-generating," while directed mining is "hypothesis-testing."
- Simulation and forecasting (e.g., survival-based) are essential for projecting the long-term value of newly acquired customers.
- Human interpretation is the "bridge" between technical patterns and actionable business strategy.

**Source File**: `Lessons/Lesson 7 Uncovering Customer Insights Using Pattern Discovery and Data Mining.pdf`

---

## Lesson 8: Customer Segmentation Using Clustering Techniques

### Core Concepts
- **Automatic Cluster Detection**: Identifying groups of similar records without relying on a target variable.
- **Clustering Types**:
    - **Hard Clustering**: Each record belongs to exactly one cluster.
    - **Soft (Fuzzy) Clustering**: Records have degrees of membership in multiple clusters.
- **K-Means Algorithm**:
    - **Assignment Step**: Assigns records to the nearest cluster seed.
    - **Update Step**: Re-calculates centroids based on the average position of all members.
    - **Voronoi Diagram**: Visualizes cluster boundaries as the set of points equidistant from cluster centers.
- **Evaluating Cluster Quality**:
    - **Intra-cluster Similarity**: Members should be close to each other.
    - **Inter-cluster Separation**: Clusters should be distinct from one another.
    - **Silhouette Score**: A metric from -1 to +1; positive values indicate a point is well-matched to its cluster and poorly matched to neighbors.
- **K-Means Variations**:
    - **K-Medians**: Uses the median; robust to outliers.
    - **K-Medoids**: Uses actual data points as centers; useful for physical location planning.
    - **K-Modes**: Adapted for categorical data.
- **Data Preparation**: Scaling (z-scores) and weighting (assigning relative importance to variables) are mandatory.

### Key Takeaways
- Clustering simplifies complexity by segmenting "competing signals" into manageable groups.
- Centroids (cluster averages) are the primary tool for interpreting what a cluster represents.
- High-dimensional spaces are "sparse," making it difficult to find truly similar neighbors; keep clustering dimensions focused.

**Source File**: `Lessons/Lesson 8 Customer Segmentation Using Clustering Techniques.pdf`

---

## Lesson 9: Market Basket Analysis and Association Rules

### Core Concepts
- **Market Basket Analysis (MBA)**: Identifying relationships and co-occurrences in transaction data (e.g., "What products are bought together?").
- **Four Levels of MBA Data**: Stores, Customers, Orders (Baskets), and Items.
- **Association Rules**: "If [LHS], then [RHS]."
    - **Actionable Rules**: High-quality insights that can be leveraged (e.g., beer and diapers).
    - **Trivial Rules**: Obvious associations that offer no new value (e.g., maintenance agreements and appliances).
    - **Inexplicable Rules**: Random anomalies that lack clear interpretation.
- **Evaluation Metrics**:
    - **Support**: How frequently the items in the rule appear in the dataset.
    - **Confidence**: The probability that the RHS is present given the LHS is present.
    - **Lift**: Confidence / Prevalence(RHS). A lift > 1 indicates a meaningful association better than random chance.
    - **Chi-Square**: Statistical significance test to ensure the association isn't due to random noise.
- **Sequential Pattern Analysis**: Adding a time dimension to track the *order* of purchases (e.g., Buying a lawnmower, then a garden hose 6 weeks later). Requires unique customer identifiers.

### Key Takeaways
- MBA is powerful for cross-selling, but limited in domains with small product ranges (e.g., banking) where bundles dominate.
- Product Hierarchies help generalize rules (e.g., from "Fuji Apple" to "Fruit") to increase support and find broader patterns.
- Negative Lift: If lift < 1, the presence of one item *reduces* the likelihood of another; reversing the rule may reveal a strong "NOT" association.

**Source File**: `Lessons/Lesson 9 Market Basket Analysis and Association Rules to Uncover Customer Habits.pdf`

---

## Lesson 10: Working with Customer Data Part 1

### Core Concepts
- **Customer Signature**: A consolidated, one-row-per-customer representation of a customer at a specific point in time ("as of" date). It is the foundation for all predictive CRM models.
- **Customer Roles**: 
    - **Payer**: The one who pays (often the only one with a reliable record).
    - **Decider**: The one who chooses the product.
    - **User**: The end consumer.
- **Data Identification Challenges**:
    - **Anonymous Transactions**: Limited to single-event insights (e.g., cash purchases).
    - **Card/Cookie-linked**: Provides a partial view and may introduce bias (device-level vs. person-level).
    - **Account-linked**: Consolidating fragmented data (e.g., home vs. car insurance) into a unified customer view.
- **Householding**: Grouping individual customers into a single unit (household). Requires complex de-duplication and matching rules (names, addresses).
- **Aggregating Transactions**: Converting "vertical" transaction logs (many rows per customer) into "horizontal" signature fields (e.g., total spend, transaction count, days since last purchase).
- **Handling Missing Values**: Distinguish between **Unknown** (applicable but missing, e.g., birthdate) and **Non-Applicable** (e.g., spending in the last 6 months for a new customer).

### Key Takeaways
- Predictive models require targets to come from a later timeframe than inputs to ensure causality.
- Don't discard records with missing values; the fact that data is missing may itself be a strong predictor (e.g., customers who don't provide a phone number might be higher risk).
- Updating signatures is the first step in the scoring process.

**Source File**: `Lessons/Lesson 10 Working with Customer Data Part 1.pdf`

---

## Lesson 11: Working with Customer Data Part 2

### Core Concepts
- **Derived Variables**: New features created by transforming or combining raw data to incorporate domain expertise (e.g., Handset Churn Rate).
- **Transformations**:
    - **Standardizing**: Centering (moving the mean to 0) and Rescaling (adjusting the standard deviation to 1) for comparability across different units.
    - **Percentiles/Quantiles**: Ranking values relative to the population; excellent for handling skewed data and outliers.
    - **Rates**: Converting counts into time-based metrics (e.g., calls per month).
- **Binning (Discretization)**:
    - **Equal-width**: Fixed intervals.
    - **Equal-frequency**: Bins based on quantiles.
    - **Supervised Binning**: Using a target variable (e.g., a decision tree split) to define the most predictive bin boundaries.
- **Geocoding**: Converting addresses to coordinates to derive location-based variables, such as distance to the nearest store or relative wealth compared to the local community.
- **Handling Sparse Data**: In domains like banking, where customers only hold a few of many possible products, use **Account Set Patterns** (categorical labels for combinations) or dense summaries (total balance).

### Key Takeaways
- **Widening Narrow Data**: The core task of data mining is turning narrow transaction logs (few columns, many rows) into wide customer signatures (many columns, one row).
- **Ecological Fallacy**: Avoid assuming that group-level patterns (e.g., average school SAT scores) apply to every individual in that group.
- Ratios often reveal insights that absolute values mask (e.g., order value vs. return value to identify "renters").

**Source File**: `Lessons/Lesson 11 Working with Customer Data Part 2.pdf`

---

## Lesson 12: Listening to Customers – Text Mining for Sentiment Analysis

### Core Concepts
- **Text Mining**: Extracting structured features (keywords, topics, sentiment) from unstructured text data (emails, call transcripts, tweets).
- **Bag of Words**: Treating a document as an unordered collection of words, focusing on frequency rather than grammar.
- **Natural Language Processing (NLP)**: Advanced analysis that considers grammatical structure and semantic meaning.
- **Text Pre-processing**:
    - **Disambiguation**: Determining word meaning based on context (e.g., "close" as a verb vs. adjective).
    - **Stemming**: Reducing words to their root form (e.g., "deliveries" to "deliver").
    - **Stop Word Removal**: Filtering common words with low informational value (e.g., "the", "a").
- **Term-Document Matrix (TDM)**: A matrix where rows are documents and columns are terms; cells often contain **Inverse Document Frequency (IDF)** to weigh term importance.
- **Latent Semantic Indexing (LSI)**: Using **Singular Value Decomposition (SVD)** to reduce the TDM into a multidimensional space of latent themes or concepts.
- **Sentiment Analysis**: Automatically assessing the emotional tone of a text on a scale (e.g., -1 to +1).

### Key Takeaways
- Text mining is inherently iterative; cleaning, parsing, and synonym mapping are the most time-consuming steps.
- Naïve Bayesian models are highly effective for text classification (e.g., Spam vs. Not Spam) because they accumulate evidence from many sparse features.
- Search trends (e.g., Google Trends) can serve as leading indicators for near-term sales forecasting.

**Source File**: `Lessons/Lesson 12 Listening to customers _ text mining for sentiment analysis.pdf`

---

## Reference: Data Visualization with ggplot2

### Core Concepts
- **Grammar of Graphics**: Every graph is built from three components: a **data set**, a set of **geoms** (visual marks), and a **coordinate system**.
- **Aesthetic Mappings (aes)**: Mapping variables in the data to visual properties like `x`, `y`, `color`, `size`, and `shape`.
- **Key Geometries (Geoms)**:
    - **One Variable**: `geom_histogram()`, `geom_density()`, `geom_bar()` (for discrete).
    - **Two Variables**: `geom_point()` (scatter), `geom_boxplot()`, `geom_line()`, `geom_smooth()` (fitted lines).
    - **Three Variables**: `geom_tile()`, `geom_raster()`, `geom_contour()`.
- **Statistical Transformations (Stats)**: Calculations that happen before plotting (e.g., `stat="bin"` for histograms, `stat="identity"` to use raw values).
- **Position Adjustments**: `dodge` (side-by-side), `stack`, `fill` (normalized stack), `jitter` (add noise to avoid overplotting).
- **Faceting**: Dividing a plot into subplots based on a variable using `facet_grid()` or `facet_wrap()`.
- **Scales**: Control how data values map to aesthetics (e.g., `scale_color_gradient()`, `scale_x_log10()`).

### Key Takeaways
- Use `ggsave("plot.png")` to save the last plot created.
- `qplot()` (quick plot) is a shortcut for simpler graphs, while `ggplot()` offers full control.
- Themes (`theme_minimal()`, `theme_bw()`) allow for easy customization of plot appearance.

**Source File**: `Lessons/ggplot2-cheatsheet.pdf`

---
