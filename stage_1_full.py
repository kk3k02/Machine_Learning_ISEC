# ===============================================
# HEART DISEASE DATASET – FULL PROJECT SCRIPT
# ===============================================

# ====== IMPORT LIBRARIES ======
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from sklearn import set_config

# ====== 1. LOAD DATASET ======
df = pd.read_csv("heart.csv")

print("=== Preview of the Dataset ===")
print(df.head())

# =====================================================
# SECTION 1 – DATASET DESCRIPTION AND EXPLORATORY ANALYSIS
# =====================================================

# --- Basic dataset information ---
print("\n=== Basic Information ===")
print(f"Number of instances (rows): {df.shape[0]}")
print(f"Number of features (columns): {df.shape[1]}")

print("\n=== Data Types ===")
print(df.dtypes)

# --- Summary statistics ---
print("\n=== Descriptive Statistics (Numerical Variables) ===")
print(df.describe().T)

# --- Missing values check ---
print("\n=== Missing Values per Column ===")
print(df.isnull().sum())

# --- Unique values for categorical/ordinal variables ---
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
print("\n=== Unique Values for Categorical / Ordinal Features ===")
for col in categorical_features:
    print(f"{col}: {df[col].unique()}")

# --- Target variable distribution ---
print("\n=== Target Variable Distribution ===")
print(df['target'].value_counts())
print("\nPercentage distribution:")
print(df['target'].value_counts(normalize=True) * 100)

# --- Plot: target variable distribution ---
plt.figure(figsize=(5, 4))
sns.countplot(x='target', data=df, hue='target', palette='Set2', legend=False)
plt.title("Distribution of Target Variable (Heart Disease)")
plt.xlabel("Heart Disease (1 = Yes, 0 = No)")
plt.ylabel("Count")
plt.show()

# --- Feature type summary ---
print("\n=== Summary of Feature Types ===")
feature_summary = pd.DataFrame({
    'Feature Name': df.columns,
    'Data Type': df.dtypes.astype(str),
    'Unique Values': [df[col].nunique() for col in df.columns]
})
print(feature_summary)

# --- Correlation matrix ---
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Features")
plt.show()

# =====================================================
# SECTION 2 – DATA PREPARATION
# =====================================================

# --- 1. DATA CLEANING ---
print("\n=== DATA CLEANING ===")
print("Missing values per column:\n", df.isnull().sum())

# Remove duplicate rows
duplicates = df.duplicated().sum()
print(f"Number of duplicated rows: {duplicates}")
df = df.drop_duplicates().reset_index(drop=True)

# Verify missing values again
print("\nMissing values after cleaning:\n", df.isnull().sum())

# --- 2. FEATURE ENCODING (manual demonstration) ---
numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]

encoder = OneHotEncoder(drop="first", handle_unknown="ignore")
encoded_cat = encoder.fit_transform(df[categorical_features]).toarray()

encoded_df = pd.DataFrame(
    encoded_cat,
    columns=encoder.get_feature_names_out(categorical_features),
    index=df.index
)

df_encoded = pd.concat([df[numeric_features], encoded_df, df["target"]], axis=1)
print(f"\nData shape after encoding: {df_encoded.shape}")

# --- 3. NORMALIZATION / STANDARDIZATION ---
scaler = StandardScaler()
df_encoded[numeric_features] = scaler.fit_transform(df_encoded[numeric_features])
print("\nStandardization applied to numeric features.")

# --- 4. DATASET SPLITTING ---
df_encoded = df_encoded.dropna(subset=["target"])
X = df_encoded.drop("target", axis=1)
y = df_encoded["target"]

print(f"\nAny NaN in y? {y.isna().sum()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print("Data preparation completed successfully!")

# =====================================================
# SECTION 3 – DATA PROCESSING PIPELINE
# =====================================================

print("\n=== DATA PROCESSING PIPELINE ===")

# Define transformers for pipeline
categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore")
numeric_transformer = StandardScaler()

# Combine transformations
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", categorical_transformer, categorical_features),
        ("numerical", numeric_transformer, numeric_features)
    ]
)

# Build pipeline (preprocessing + baseline model)
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Optional: display structure (in Jupyter it shows diagram)
set_config(display='diagram')
print(pipeline)

# Train/test split directly from cleaned raw data (not encoded one)
X_raw = df[categorical_features + numeric_features]
y_raw = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
)

# Fit and evaluate pipeline
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("\n=== Classification Report (Baseline Logistic Regression) ===")
print(classification_report(y_test, y_pred, digits=3))

# Transformed feature info
feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
print("\nNumber of transformed features:", len(feature_names))
print("First 10 transformed features:", feature_names[:10])

# Save pipeline
joblib.dump(pipeline, "heart_pipeline.joblib")
print("\nSaved full preprocessing + model pipeline to 'heart_pipeline.joblib'")

# =====================================================
# END OF SCRIPT
# =====================================================
