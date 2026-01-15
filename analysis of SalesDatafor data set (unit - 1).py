import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

# Load the dataset
S0=pd.read_csv("C:\\Users\\giris\\Documents\\Predective analysis\\unit-1\\SalesDatafor preprocessing(maneet kaur).csv")
S0
# Display structure and summary of the dataset
print(S0.info())
print(S0.describe())
#identify the NAs in data
print(S0.isna().sum())

#handle the NAs if available
#check for misspelled data


# Handling missing values
S1 = S0.dropna()  # Removing rows with missing values
print(S1.info())
print(S1.describe())

# Replace missing values with mean
for col in ["Sales", "Profit", "Unit Price"]:
    S0[col].fillna(S0[col].mean(), inplace=True)
print(S0.info())

#without loop
S0["Sales"].fillna(S0["Sales"].mean(), inplace=True)  
S0["Profit"].fillna(S0["Profit"].mean(), inplace=True)  
S0["Unit Price"].fillna(S0["Unit Price"].mean(), inplace=True)  

print(S0.info())


# Replace missing categorical values with random choice
for col in ["Order Priority", "Ship Mode", "Customer Name"]:
    S0[col].fillna(S0[col].mode()[0], inplace=True)

print(S0.info())

#without loop
S0["Order Priority"].fillna(S0["Order Priority"].mode()[0], inplace=True)  
S0["Ship Mode"].fillna(S0["Ship Mode"].mode()[0], inplace=True)                   #mod()[0] means the index of the most frequently appeared
S0["Customer Name"].fillna(S0["Customer Name"].mode()[0], inplace=True)  

print(S0.info())

#method 2
# Replace missing values with random values between min and max
for col in ["Sales", "Profit", "Unit Price"]:
    S0[col].fillna(np.random.uniform(S0[col].min(), S0[col].max()), inplace=True)

#without loop
S0["Sales"].fillna(np.random.uniform(S0["Sales"].min(), S0["Sales"].max()), inplace=True)  
S0["Profit"].fillna(np.random.uniform(S0["Profit"].min(), S0["Profit"].max()), inplace=True)  
S0["Unit Price"].fillna(np.random.uniform(S0["Unit Price"].min(), S0["Unit Price"].max()), inplace=True)  

print(S0.info())



# Detecting outliers and inliers
numeric_cols = ["Order Quantity", "Sales", "Profit", "Unit Price", "Shipping Cost"]
outliers = []
for col in numeric_cols:
    Q1 = S1[col].quantile(0.25)     #It marks the point below which 25% of the data falls.
    Q3 = S1[col].quantile(0.75)     #It marks the point below which 75% of the data falls.
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers.extend(S1[(S1[col] < lower_bound) | (S1[col] > upper_bound)].index)
outliers
S2 = S1.drop(set(outliers))
S2

# Fix categorical data noise
S2.loc[S2["Order Priority"] == "Loww", "Order Priority"] = "Low"

# Correlation analysis
print(S2[["Shipping Cost", "Order Quantity"]].corr())
sns.scatterplot(x=S2["Order Quantity"], y=S2["Shipping Cost"])
plt.show()

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
sns.boxplot(y=S2["Order Quantity"], ax=axes[0]).set(title="Order Quantity")
sns.boxplot(y=S2["Profit"], ax=axes[1]).set(title="Profit")
plt.show()

# Histogram for distributions
plt.figure(figsize=(10, 5))
sns.histplot(S2["Order Quantity"], bins=30, kde=True)
plt.title("Order Quantity Distribution")
plt.show()


# Checking skewness and normality
print("Sales Skewness:", skew(S2["Sales"]))
sns.histplot(S2["Sales"], kde=True)
plt.show()

# Transform skewed data
S2["Log_Sales"] = np.log(S2["Sales"] + 1)
print("Log Sales Skewness:", skew(S2["Log_Sales"]))

# Standardization (Z-score normalization)
S1[numeric_cols] = (S1[numeric_cols] - S1[numeric_cols].mean()) / S1[numeric_cols].std()

# Min-Max Normalization
S1[numeric_cols] = (S1[numeric_cols] - S1[numeric_cols].min()) / (S1[numeric_cols].max() - S1[numeric_cols].min())

# Create a new variable T.cost
S2["T.cost"] = S2["Unit.Price"] * S2["Order.Quantity"] + S2["Shipping.Cost"]
sns.histplot(S2["T.cost"], bins=30, kde=True)
plt.title("Total Cost Distribution")
plt.show()






