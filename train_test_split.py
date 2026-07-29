# train_test_split.py
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ============================================
# CREATE DIRECTORIES
# ============================================
os.makedirs("Dataset", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("="*60)
print("TRAIN-TEST SPLIT STARTED")
print("="*60)

# ============================================
# LOAD DATASET
# ============================================
try:
    # Try different paths
    if os.path.exists("Dataset/feature_engineering_dataset.csv"):
        df = pd.read_csv("Dataset/feature_engineering_dataset.csv")
    elif os.path.exists("../Dataset/feature_engineering_dataset.csv"):
        df = pd.read_csv("../Dataset/feature_engineering_dataset.csv")
    else:
        raise FileNotFoundError("Dataset not found!")
    
    print("✓ Dataset loaded successfully!")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {len(df.columns)}")
    
except FileNotFoundError as e:
    print(f"✗ Error: {e}")
    print("  Make sure 'feature_engineering_dataset.csv' exists in Dataset folder")
    sys.exit()

# ============================================
# DATA CLEANING - FIX THE MAIN ISSUE
# ============================================
print("\n" + "-"*60)
print("DATA CLEANING & PREPROCESSING")
print("-"*60)

# 1. Replace empty strings and spaces with NaN
df = df.replace(r'^\s*$', np.nan, regex=True)  # Replace empty/spaces with NaN
df = df.replace(' ', np.nan)  # Replace single space with NaN
df = df.replace('', np.nan)   # Replace empty string with NaN

# 2. Check for columns with string data
print("\nChecking data types...")
string_columns = df.select_dtypes(include=['object']).columns.tolist()
print(f"String columns found: {len(string_columns)}")
if string_columns:
    print(f"  Examples: {string_columns[:5]}")

# 3. Find target column
target_col = None
for col in df.columns:
    if 'churn' in col.lower():
        target_col = col
        break

if target_col is None:
    print("✗ Error: 'Churn' column not found!")
    print(f"  Available columns: {df.columns.tolist()[:10]}...")
    exit()

print(f"\n✓ Target column: '{target_col}'")

# 4. Separate features and target
X = df.drop(target_col, axis=1)
y = df[target_col]

# 5. Clean target variable
print(f"\nCleaning target variable...")
print(f"  Unique values before: {y.unique()}")

# Convert target to binary
if y.dtype == 'object':
    # Handle different formats
    y = y.map({'Yes': 1, 'No': 0, 'True': 1, 'False': 0, 
               'yes': 1, 'no': 0, 'true': 1, 'false': 0,
               1: 1, 0: 0})
    # Fill any remaining NaN with 0
    y = y.fillna(0)
else:
    # Ensure binary
    y = y.astype(int)

print(f"  Unique values after: {y.unique()}")
print(f"  Churn rate: {y.mean()*100:.2f}%")

# 6. Clean features - Convert all string columns to numeric
print(f"\nCleaning features...")

# First, check for any remaining string columns
string_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"String columns in features: {len(string_cols)}")

if string_cols:
    print(f"  Converting string columns to numeric...")
    
    for col in string_cols:
        try:
            # Try to convert to numeric, coercing errors to NaN
            X[col] = pd.to_numeric(X[col], errors='coerce')
            
            # If column became all NaN after conversion, it's categorical
            if X[col].isna().all():
                # Use label encoding for categorical
                le = LabelEncoder()
                # Fill NaN with 'Unknown' for encoding
                X[col] = X[col].fillna('Unknown')
                X[col] = le.fit_transform(X[col].astype(str))
                print(f"    ✓ {col}: Label encoded (categorical)")
            else:
                # Fill remaining NaN with mean
                X[col] = X[col].fillna(X[col].mean())
                print(f"    ✓ {col}: Converted to numeric")
                
        except Exception as e:
            # If conversion fails, use label encoding
            try:
                le = LabelEncoder()
                X[col] = X[col].fillna('Unknown')
                X[col] = le.fit_transform(X[col].astype(str))
                print(f"    ✓ {col}: Label encoded (fallback)")
            except:
                print(f"    ✗ {col}: Could not convert, dropping...")
                X = X.drop(col, axis=1)

# 7. Check for any remaining non-numeric columns
non_numeric_cols = X.select_dtypes(include=['object']).columns.tolist()
if non_numeric_cols:
    print(f"\n⚠ Warning: Still have {len(non_numeric_cols)} non-numeric columns:")
    print(f"  {non_numeric_cols[:5]}")
    # Drop them
    X = X.drop(non_numeric_cols, axis=1)
    print(f"  Dropped non-numeric columns")

# 8. Handle missing values
print(f"\nHandling missing values...")
missing_before = X.isnull().sum().sum()
print(f"  Missing values before: {missing_before}")

# Fill remaining NaN with 0 or mean
X = X.fillna(0)
print(f"  Missing values after: {X.isnull().sum().sum()}")

# 9. Ensure all data is numeric
print(f"\nFinal data types...")
print(f"  X shape: {X.shape}")
print(f"  X has {X.select_dtypes(include=['object']).shape[1]} object columns")

if X.select_dtypes(include=['object']).shape[1] > 0:
    print("✗ Error: Still have object columns!")
    print(f"  Columns: {X.select_dtypes(include=['object']).columns.tolist()}")
    # Force convert all to numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(0)

# ============================================
# TRAIN-TEST SPLIT
# ============================================
print("\n" + "-"*60)
print("TRAIN-TEST SPLIT")
print("-"*60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"X_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")
print(f"y_train : {y_train.shape} (Churn: {y_train.mean()*100:.2f}%)")
print(f"y_test  : {y_test.shape} (Churn: {y_test.mean()*100:.2f}%)")

# ============================================
# SAVE FILES
# ============================================
X_train.to_csv("Dataset/X_train.csv", index=False)
X_test.to_csv("Dataset/X_test.csv", index=False)
y_train.to_csv("Dataset/y_train.csv", index=False)
y_test.to_csv("Dataset/y_test.csv", index=False)

print("\n✓ Files saved successfully:")
print(f"  → Dataset/X_train.csv")
print(f"  → Dataset/X_test.csv")
print(f"  → Dataset/y_train.csv")
print(f"  → Dataset/y_test.csv")

# ============================================
# DATA SUMMARY
# ============================================
print("\n" + "="*60)
print("DATA SUMMARY")
print("="*60)
print(f"Total samples: {len(df)}")
print(f"Features: {X.shape[1]}")
print(f"Target: {target_col}")
print(f"Churn distribution:")
print(f"  Class 0 (No Churn): {sum(y==0)} ({sum(y==0)/len(y)*100:.2f}%)")
print(f"  Class 1 (Churn): {sum(y==1)} ({sum(y==1)/len(y)*100:.2f}%)")
print("="*60)