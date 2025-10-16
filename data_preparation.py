# ===============================================
# DATA PREPARATION – HEART DISEASE DATASET
# ===============================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load dataset
df = pd.read_csv("heart.csv")

# -----------------------------
# 1. DATA CLEANING
# -----------------------------

# Check for missing values
print("Missing values per column:\n", df.isnull().sum())

# Remove duplicate rows and reset index
duplicates = df.duplicated().sum()
print(f"Number of duplicated rows: {duplicates}")
df = df.drop_duplicates().reset_index(drop=True)

# Double-check for any NaN after cleaning
print("\nMissing values after cleaning:\n", df.isnull().sum())

# -----------------------------
# 2. FEATURE ENCODING
# -----------------------------

categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]

encoder = OneHotEncoder(drop="first", handle_unknown="ignore")
encoded_cat = encoder.fit_transform(df[categorical_features]).toarray()

encoded_df = pd.DataFrame(
    encoded_cat,
    columns=encoder.get_feature_names_out(categorical_features),
    index=df.index  # ✅ ensure same index alignment
)

df_encoded = pd.concat([df[numeric_features], encoded_df, df["target"]], axis=1)

print(f"\nData shape after encoding: {df_encoded.shape}")

# -----------------------------
# 3. NORMALIZATION / STANDARDIZATION
# -----------------------------

scaler = StandardScaler()
df_encoded[numeric_features] = scaler.fit_transform(df_encoded[numeric_features])

print("\nStandardization applied to numeric features.")

# -----------------------------
# 4. DATASET SPLITTING
# -----------------------------

# Remove any potential NaN rows in target (safety check)
df_encoded = df_encoded.dropna(subset=["target"])

X = df_encoded.drop("target", axis=1)
y = df_encoded["target"]

# Verify again
print(f"\nAny NaN in y? {y.isna().sum()}")

# Split data (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print("Data preparation completed successfully!")
