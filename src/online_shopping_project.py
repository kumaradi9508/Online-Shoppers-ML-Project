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


# ============================================================
# DAY 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

# Step 24: Load the cleaned dataset
df_clean = pd.read_csv(r"E:\Online-Shoppers-ML-Project\data\online_shoppers_intention_cleaned.csv")
print(df_clean)

print("\nCleaned dataset shape:")
print(df_clean.shape)

# Step 25: Purchase and non-purchase analysis
print("\nOBJECTIVE 1: PURCHASE ANALYSIS")
purchase_count = df_clean["Revenue"].value_counts()
print("\nPurchase counts:")
print(purchase_count)

purchase_percentage = df_clean["Revenue"].value_counts(normalize=True) * 100
print("\nPurchase percentages:")
print(purchase_percentage)

purchase_count.to_csv(r"E:\Online-Shoppers-ML-Project\output\results\purchase_analysis.csv")

plt.figure(figsize=(7, 5))
sns.countplot(data=df_clean, x="Revenue", hue='Revenue', palette={False: "#3B82F6", True: "#F97316"},legend=False)
plt.title("Purchase vs Non-Purchase Sessions")
plt.xlabel("Purchase Made")
plt.ylabel("Number of Sessions")
plt.tight_layout()
plt.savefig(r"E:\Online-Shoppers-ML-Project\output\figures\purchase_analysis.png")
plt.show()


# Step 26: Purchase rate by visitor type
print("\nOBJECTIVE 2: VISITOR TYPE ANALYSIS")

visitor_purchase = df_clean.groupby("VisitorType")["Revenue"].mean() * 100
print("\nPurchase rate by visitor type (%):")
print(visitor_purchase)

visitor_purchase.to_csv(r"E:\Online-Shoppers-ML-Project\output\results\visitor_purchase_rate.csv")

plt.figure(figsize=(8, 5))
sns.barplot(x=visitor_purchase.index, y=visitor_purchase.values, hue=visitor_purchase.index, palette=["#005F73", "#0A9396", "#94D2BD"], legend=False)
plt.title("Purchase Rate by Visitor Type")
plt.xlabel("Visitor Type")
plt.ylabel("Purchase Rate (%)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(r"E:\Online-Shoppers-ML-Project\output\figures\visitor_purchase_rate.png")
plt.show()

# Step 27: Browsing duration analysis
print("\nOBJECTIVE 3: BROWSING DURATION ANALYSIS")

duration_analysis = df_clean.groupby("Revenue")[["Administrative_Duration","Informational_Duration","ProductRelated_Duration"]].mean()
print("\nAverage browsing duration:")
print(duration_analysis)

duration_analysis.to_csv(r"E:\Online-Shoppers-ML-Project\output\results\duration_analysis.csv")

plt.figure(figsize=(8, 5))
sns.boxplot(data=df_clean,x="Revenue",y="ProductRelated_Duration",hue="Revenue",palette={False: "#8B5CF6", True: "#14B8A6"},legend=False)
plt.title("Product Page Duration by Purchase Outcome")
plt.xlabel("Purchase Made")
plt.ylabel("Product-Related Duration (seconds)")
plt.xticks([0, 1], ["No Purchase", "Purchase"])
plt.tight_layout()
plt.savefig(r"E:\Online-Shoppers-ML-Project\output\figures\duration_analysis.png")
plt.show()


# Step 28: Month-wise purchase rate
print("\nOBJECTIVE 4: MONTH-WISE PURCHASE ANALYSIS")

month_purchase = df_clean.groupby("Month")["Revenue"].mean() * 100

month_order = ["Feb", "Mar", "May", "June", "Jul","Aug", "Sep", "Oct", "Nov", "Dec"]

month_purchase = month_purchase.reindex([month for month in month_order if month in month_purchase.index])
print("\nMonth-wise purchase rate (%):")
print(month_purchase)

month_purchase.to_csv(r"E:\Online-Shoppers-ML-Project\output\results\month_purchase_rate.csv")

plt.figure(figsize=(10, 5))
plt.plot(month_purchase.index, month_purchase.values, color="#F59E0B", marker="o", markersize=7, linewidth=2)
plt.title("Monthly Purchase Rate Trend")
plt.xlabel("Month")
plt.ylabel("Purchase Rate (%)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(r"E:\Online-Shoppers-ML-Project\output\figures\month_purchase_rate.png")
plt.show()

# Step 29: Weekend purchase analysis
print("\nOBJECTIVE 5: WEEKEND PURCHASE ANALYSIS")

weekend_purchase = df_clean.groupby("Weekend")["Revenue"].mean() * 100
print("\nPurchase rate by weekend (%):")
print(weekend_purchase)

weekend_purchase.to_csv(r"E:\Online-Shoppers-ML-Project\output\results\weekend_purchase_rate.csv")
weekend_labels = weekend_purchase.rename(index={False: "Weekday", True: "Weekend"})

plt.figure(figsize=(7, 5))
ax = sns.barplot(x=weekend_labels.index,y=weekend_labels.values,palette=["#EC4899", "#BE185D"],hue=weekend_labels.index,legend=False)
plt.title("Purchase Rate: Weekday vs Weekend")
plt.xlabel("Day Type")
plt.ylabel("Purchase Rate (%)")
plt.tight_layout()
plt.savefig(r"E:\Online-Shoppers-ML-Project\output\figures\weekend_purchase_rate.png")
plt.show()

# Step 30: Special day purchase analysis
print("\nOBJECTIVE 6: SPECIAL DAY ANALYSIS")

specialday_purchase = df_clean.groupby("SpecialDay")["Revenue"].mean() * 100
print("\nPurchase rate by special day value (%):")
print(specialday_purchase)

specialday_purchase.to_csv(r"E:\Online-Shoppers-ML-Project\output\results\specialday_purchase_rate.csv")

plt.figure(figsize=(8, 5))
plt.plot(specialday_purchase.index, specialday_purchase.values, marker="o", color="purple")
plt.title("Purchase Rate by Special Day Value")
plt.xlabel("Special Day Value")
plt.ylabel("Purchase Rate (%)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(r"E:\Online-Shoppers-ML-Project\output\figures\specialday_purchase_rate.png")
plt.show()

# Step 31: Final EDA summary
print("\n========== DAY 3 SUMMARY ==========")
print("Total cleaned records:", len(df_clean))
print("Total visualizations created: 6")
print("Charts saved in output/figures/")
print("Analysis tables saved in output/results/")
print("\nDay 3 EDA completed successfully.")

# DAY 4: FEATURE ENGINEERING AND DATA PREPARATION
# Step 32: Load Cleaned Dataset
df = pd.read_csv(r"E:\Online-Shoppers-ML-Project\data\online_shoppers_intention_cleaned.csv")
print("Dataset Shape:", df.shape)

# Step 33: Separate features and target
X = df.drop("Revenue", axis=1)
y = df["Revenue"]

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)
print("Target Values:")
print(y.value_counts())

# Step 34: Encode categorical columns
X = pd.get_dummies(X, drop_first=True, dtype=int)
print("Shape After Encoding:", X.shape)
print(X.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)
print("Training Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)

# Step 35: Standard Scaling for Logistic Regression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_standard = scaler.fit_transform(X_train)
X_test_standard = scaler.transform(X_test)

print("Standard Scaled Training Shape:", X_train_standard.shape)
print("Standard Scaled Testing Shape:", X_test_standard.shape)

# Step 36: Min-Max Scaling for KNN
from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler()
X_train_minmax = minmax.fit_transform(X_train)
X_test_minmax = minmax.transform(X_test)

print("Min-Max Training Shape:", X_train_minmax.shape)
print("Min-Max Testing Shape:", X_test_minmax.shape)

# Step 37: Save Training and Testing Summary
split_summary = pd.DataFrame({"Dataset": ["Training", "Testing"],"Rows": [len(X_train), len(X_test)],"Purchase": [y_train.sum(), y_test.sum()],"Non_Purchase": [(y_train == 0).sum(), (y_test == 0).sum()]})
split_summary.to_csv(r"E:\Online-Shoppers-ML-Project\output\results\day4_split_summary.csv",index=False)
print(split_summary)
print("Day 4 Preprocessing Completed Successfully")

# Day 5: Implemented Machine Learning Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Step 38: Logistic Regression Model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_standard, y_train)
y_pred_model = model.predict(X_test_standard)

print("\n========== LOGISTIC REGRESSION ==========")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_model))
print("\nAccuracy:", accuracy_score(y_test, y_pred_model))
print("\nClassification Report:\n", classification_report(y_test, y_pred_model))

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_model),annot=True, fmt="d", cmap="Blues", xticklabels=["Not Purchased", "Purchased"], yticklabels=["Not Purchased", "Purchased"])
plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(r"E:\Online-Shoppers-ML-Project\output\figures\logistic_regression_cm.png")
plt.show()

# Step 39: Gaussian Naive Bayes Model
model = GaussianNB()
model.fit(X_train, y_train)
y_pred_nb = model.predict(X_test)

print("\n========== NAIVE BAYES ==========")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_nb))
print("\nAccuracy:", accuracy_score(y_test, y_pred_nb))
print("\nClassification Report:\n", classification_report(y_test, y_pred_nb))

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_nb), annot=True, fmt="d", cmap="Greens", xticklabels=["Not Purchased", "Purchased"], yticklabels=["Not Purchased", "Purchased"])
plt.title("Gaussian Naive Bayes - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(r"E:\Online-Shoppers-ML-Project\output\figures\naive_bayes_cm.png")
plt.show()

# Step 40: K-Nearest Neighbour Model
knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
knn.fit(X_train_minmax, y_train)
y_pred_knn = knn.predict(X_test_minmax)

print("\n========== KNN ==========")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
print("\nAccuracy:", accuracy_score(y_test, y_pred_knn))
print("\nClassification Report:\n", classification_report(y_test, y_pred_knn))

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt="d", cmap="Oranges", xticklabels=["Not Purchased", "Purchased"], yticklabels=["Not Purchased", "Purchased"])
plt.title("KNN - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(r"E:\Online-Shoppers-ML-Project\output\figures\knn_cm.png")
plt.show()

# Step 41: Decision Tree Model
classifier = DecisionTreeClassifier(criterion="entropy", random_state=42)
classifier.fit(X_train, y_train)
y_pred_dt = classifier.predict(X_test)

print("\n========== DECISION TREE ==========")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("\nAccuracy:", accuracy_score(y_test, y_pred_dt))
print("\nClassification Report:\n", classification_report(y_test, y_pred_dt))

#For plot the Decision-Tree Diagram
from sklearn.tree import plot_tree

plt.figure(figsize=(24, 12))
plot_tree(classifier, filled=True, feature_names=X.columns, class_names=["Not Purchased", "Purchased"], rounded=True, max_depth=2, fontsize=10)
plt.title("Decision Tree Diagram", fontsize=18)
plt.tight_layout()
plt.savefig(r"E:\Online-Shoppers-ML-Project\output\figures\decision_tree.png", dpi=300,bbox_inches="tight")
plt.show()

# Step 42: Day 5 Final Summary
print("Day 5 Machine Learning Completed Successfully")
print("Models Implemented:")
print("1. Logistic Regression")
print("2. Gaussian Naive Bayes")
print("3. K-Nearest Neighbors (KNN)")
print("4. Decision Tree")
print("Evaluation: Confusion Matrix, Accuracy, Precision, Recall, F1-Score")

# Day 6: Comparing all the Models and checking which is best
# Step 43: Model Comparison
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
model_comparison = pd.DataFrame({"Model": ["Logistic Regression","Gaussian Naive Bayes","KNN","Decision Tree"],
                                 "Accuracy": [accuracy_score(y_test, y_pred_model),accuracy_score(y_test, y_pred_nb),accuracy_score(y_test, y_pred_knn),accuracy_score(y_test, y_pred_dt)],
                                 "Precision": [precision_score(y_test, y_pred_model, pos_label=True),precision_score(y_test, y_pred_nb, pos_label=True),precision_score(y_test, y_pred_knn, pos_label=True),precision_score(y_test, y_pred_dt, pos_label=True)],
                                 "Recall": [recall_score(y_test, y_pred_model, pos_label=True),recall_score(y_test, y_pred_nb, pos_label=True),recall_score(y_test, y_pred_knn, pos_label=True),recall_score(y_test, y_pred_dt, pos_label=True)],
                                 "F1_Score": [f1_score(y_test, y_pred_model, pos_label=True),f1_score(y_test, y_pred_nb, pos_label=True),f1_score(y_test, y_pred_knn, pos_label=True),f1_score(y_test, y_pred_dt, pos_label=True)]
})

print("\n========== MODEL COMPARISON ==========")
print(model_comparison.round(4))
model_comparison.to_csv(r"E:\Online-Shoppers-ML-Project\output\results\model_comparison.csv",index=False)
print("\nModel comparison saved successfully.")


# Step 44: Model Comparison Graph
plt.figure(figsize=(10, 6))
colors = ["#7895B2", "#A7B7C7", "#8FA99B", "#C4B7A6"]
plt.bar(model_comparison["Model"],model_comparison["Accuracy"],color=colors)
plt.title("Accuracy Comparison of Machine Learning Models")
plt.xlabel("Machine Learning Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(r"E:\Online-Shoppers-ML-Project\output\figures\model_accuracy_comparison.png",dpi=300,bbox_inches="tight")
plt.show()
print("Model comparison graph saved successfully.")

