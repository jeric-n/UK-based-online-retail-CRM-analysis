import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('data/customer_data.csv')

# Simple Linear Regression
model_simple = smf.ols('TotalRevenue ~ TotalOrders', data=df).fit()
print("Simple Linear Regression:")
print(f"Intercept: {model_simple.params['Intercept']:.2f}")
print(f"TotalOrders: {model_simple.params['TotalOrders']:.2f}")
print(f"R-squared: {model_simple.rsquared:.4f}")

# Multiple Regression
model_multiple = smf.ols('TotalRevenue ~ TotalOrders + TotalItems + TotalProducts + AvgItemPrice + DaysSinceFirst + IsUK', data=df).fit()
print("\nMultiple Regression:")
print(f"R-squared: {model_multiple.rsquared:.4f}")
print(f"TotalOrders Coeff: {model_multiple.params['TotalOrders']:.2f}")
print(f"TotalItems Coeff: {model_multiple.params['TotalItems']:.2f}")
print(f"IsUK Coeff: {model_multiple.params['IsUK']:.2f}")
