# Online Shoppers Purchasing Intention Analysis
# Day 1: Dataset Understanding

# Step 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the full dataset
df = pd.read_csv(r"E:\Online-Shoppers-ML-Project\data\online_shoppers_intention.csv")

# Step 3: Display the first five rows
print("First five rows:")
print(df.head())

# Step 4: Check the number of rows and columns
print("\nDataset shape:")
print(df.shape)

# Step 5: Display column names
print("\nColumn names:")
print(df.columns)

# Step 6: Check data types and non-null values
print("\nDataset information:")
df.info()

# Step 7: Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Step 8: Check duplicate rows
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

# Step 9: Display statistical summary
print("\nStatistical summary:")
print(df.describe())

# Step 10: Check purchase outcomes
print("\nPurchase outcome:")
print(df["Revenue"].value_counts())

print("\nPurchase outcome percentage:")
print(df["Revenue"].value_counts(normalize=True) * 100)

# Step 11: Check visitor types
print("\nVisitor types:")
print(df["VisitorType"].value_counts())

# Step 12: Check monthly session counts
print("\nMonth-wise session counts:")
print(df["Month"].value_counts())

# Step 13: Check unique values in important columns
print("\nUnique Revenue values:")
print(df["Revenue"].unique())

print("\nUnique Weekend values:")
print(df["Weekend"].unique())

print("\nUnique Month values:")
print(df["Month"].unique())

print("\nDay 1 dataset inspection completed.")



# ============================================================
# DAY 2: DATA CLEANING AND PREPROCESSING
# ============================================================

# Step 14: Created a copy of the original dataset
df_clean = df.copy()

print("\nOriginal dataset shape:")
print(df_clean.shape)

# Step 15: Check duplicate rows
print("\nDuplicate rows before cleaning:")
print(df_clean.duplicated().sum())

# Step 16: Remove duplicate rows
df_clean = df_clean.drop_duplicates()

print("\nDuplicate rows after cleaning:")
print(df_clean.duplicated().sum())

# Step 17: Check negative duration values
print("\nNegative duration values:")

print("Administrative Duration:",(df_clean["Administrative_Duration"] < 0).sum())

print("Informational Duration:",(df_clean["Informational_Duration"] < 0).sum())

print("Product Related Duration:",(df_clean["ProductRelated_Duration"] < 0).sum())

# Step 20: Check invalid bounce and exit rates
print("\nInvalid BounceRates:")
print(((df_clean["BounceRates"] < 0) |(df_clean["BounceRates"] > 1)).sum())

print("\nInvalid ExitRates:")
print(((df_clean["ExitRates"] < 0) |(df_clean["ExitRates"] > 1)).sum())

# Step 21: Display final dataset shape
print("\nFinal dataset shape:")
print(df_clean.shape)

# Step 22: Save cleaned dataset
df_clean.to_csv(r"E:\Online-Shoppers-ML-Project\data\online_shoppers_intention_cleaned.csv",index=False)
print("\nCleaned dataset saved successfully.")

# Step 23: Save cleaning summary
cleaning_summary = pd.DataFrame({"Description": ["Original rows","Rows after cleaning","Duplicate rows removed","Missing values after cleaning"],"Value": [len(df),len(df_clean),len(df) - len(df_clean),df_clean.isnull().sum().sum()]})
cleaning_summary.to_csv(r"E:\Online-Shoppers-ML-Project\output\results\cleaning_summary.csv",index=False)
print("\nDay 2 data cleaning completed.")
