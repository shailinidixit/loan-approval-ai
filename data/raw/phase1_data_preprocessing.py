# phase1_data_preprocessing.py (fixed)
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

# Step 1: Load dataset
try:
    df = pd.read_csv(r'C:\Users\shara\Documents\loan-projects\loan-approval-ai\data\raw\loan_approval_dataset.csv', encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv('data/raw/loan_data.csv', encoding='latin-1')

# --- FIX: strip whitespace from column names immediately ---
df.columns = df.columns.str.strip()

print("Data loaded successfully. Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# Step 2: Handle missing values
for col in df.columns:
    # prefer explicit assignment instead of inplace to avoid FutureWarning
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        # safe mode: if mode() returns empty, fallback to empty string
        mode_vals = df[col].mode()
        mode_val = mode_vals.iloc[0] if not mode_vals.empty else ""
        df[col] = df[col].fillna(mode_val)

print("Missing values handled.")
print(df.isnull().sum())

# Step 3: Create derived features
# Total assets = sum of all asset columns that actually exist
asset_cols = ['residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']
asset_cols = [c for c in asset_cols if c in df.columns]  # keep only existing
if asset_cols:
    # coerce asset cols to numeric (safe)
    for c in asset_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    df['TotalAssets'] = df[asset_cols].sum(axis=1)
else:
    # if none found, create TotalAssets as 0 to avoid KeyError
    print("Warning: no asset columns found with expected names; creating TotalAssets=0.")
    df['TotalAssets'] = 0

# Debt to income ratio: loan amount / income
# ensure loan_amount and income_annum exist and numeric
df['loan_amount'] = pd.to_numeric(df.get('loan_amount', 0), errors='coerce').fillna(0)
df['income_annum'] = pd.to_numeric(df.get('income_annum', 0), errors='coerce').fillna(0)
df['Debt_to_Income'] = df['loan_amount'] / (df['income_annum'] / 1000 + 1e-6)  # prevent div by zero

# Credit category based on cibil_score (coerce if present)
if 'cibil_score' in df.columns:
    df['cibil_score'] = pd.to_numeric(df['cibil_score'], errors='coerce').fillna(0)
    df['Credit_Category'] = df['cibil_score'].apply(lambda x: 'Good' if x >= 700 else 'Poor')
else:
    df['Credit_Category'] = 'Poor'  # or 'Unknown' depending on preference

# Step 4: Encode categorical variables
le = LabelEncoder()
binary_cols = ['education', 'self_employed', 'loan_status']
for col in binary_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))
    else:
        print(f"Warning: {col} not in dataframe, skipping encoding for it.")

# One-hot encode 'Credit_Category' (Good/Poor) if present
if 'Credit_Category' in df.columns:
    df = pd.get_dummies(df, columns=['Credit_Category'], drop_first=True)

# Step 5: Prepare features and target
drop_cols = [c for c in ['loan_id'] if c in df.columns]
X = df.drop(columns=drop_cols + ['loan_status'])
y = df['loan_status']

# Step 6: Train/test split (safe stratify: will raise if inappropriate)
# If stratify fails due to single-class, catch and do non-stratified split
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
except ValueError as e:
    print("Stratify failed:", e)
    print("Proceeding without stratify.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

print("Train/Test shapes -> X_train:", X_train.shape, "X_test:", X_test.shape)

# Step 7: Save processed files
os.makedirs('data/processed', exist_ok=True)
X_train.to_csv('data/processed/X_train.csv', index=False, encoding='utf-8')
X_test.to_csv('data/processed/X_test.csv', index=False, encoding='utf-8')
y_train.to_csv('data/processed/y_train.csv', index=False, encoding='utf-8')
y_test.to_csv('data/processed/y_test.csv', index=False, encoding='utf-8')

print("Phase 1 preprocessing complete. Files saved in data/processed/")
