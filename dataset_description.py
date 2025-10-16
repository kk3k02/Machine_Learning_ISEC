# ===============================================
# DATASET DESCRIPTION - HEART DISEASE DATASET
# ===============================================

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use('TkAgg')  # lub 'QtAgg' jeśli masz zainstalowany Qt

import seaborn as sns

# Load dataset
df = pd.read_csv("heart.csv")

print("=== Preview of the Dataset ===")
print(df.head())  # zamiast display()

# -----------------------------
# Basic Dataset Information
# -----------------------------
print("\n=== Basic Information ===")
print(f"Number of instances (rows): {df.shape[0]}")
print(f"Number of features (columns): {df.shape[1]}")

print("\n=== Data Types ===")
print(df.dtypes)

# -----------------------------
# Summary Statistics
# -----------------------------
print("\n=== Descriptive Statistics (Numerical Variables) ===")
print(df.describe().T)

# -----------------------------
# Check for Missing Values
# -----------------------------
print("\n=== Missing Values per Column ===")
print(df.isnull().sum())

# -----------------------------
# Unique Values for Categorical/Ordinal Variables
# -----------------------------
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
print("\n=== Unique Values for Categorical / Ordinal Features ===")
for col in categorical_features:
    print(f"{col}: {df[col].unique()}")

# -----------------------------
# Target Variable Distribution
# -----------------------------
print("\n=== Target Variable Distribution ===")
print(df['target'].value_counts())
print("\nPercentage distribution:")
print(df['target'].value_counts(normalize=True) * 100)

# Plot target distribution
plt.figure(figsize=(5, 4))
sns.countplot(x='target', data=df, hue='target', palette='Set2', legend=False)
plt.title("Distribution of Target Variable (Heart Disease)")
plt.xlabel("Heart Disease (1 = Yes, 0 = No)")
plt.ylabel("Count")
plt.show()

# -----------------------------
# Feature Type Summary
# -----------------------------
print("\n=== Summary of Feature Types ===")
feature_summary = pd.DataFrame({
    'Feature Name': df.columns,
    'Data Type': df.dtypes.astype(str),
    'Unique Values': [df[col].nunique() for col in df.columns]
})
print(feature_summary)

# -----------------------------
# Correlation Matrix (for overview)
# -----------------------------
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Features")
plt.show()
