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
