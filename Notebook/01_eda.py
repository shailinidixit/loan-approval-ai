import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the processed training data from Phase 1
X_train = pd.read_csv(r'C:\Users\shail\Documents\loan-projects\loan-approval-ai\data\processed\X_train.csv')
y_train = pd.read_csv(r'C:\Users\shail\Documents\loan-projects\loan-approval-ai\data\processed\Y_train.csv')

# Combine features and target into a single dataframe
train_df = X_train.copy()
train_df['loan_status'] = y_train


# Recreate Credit_Category_Good if missing
if 'Credit_Category_Good' not in train_df.columns:
    train_df['Credit_Category_Good'] = (train_df['cibil_score'] >= 700).astype(int)
    print("✅ Created 'Credit_Category_Good' column based on cibil_score.")
else:
    print("✅ 'Credit_Category_Good' already exists.")

# Print first 5 rows to check
print(train_df.head())
# Print info to check column types and missing values
print(train_df.info())

numerical_cols = ['income_annum', 'loan_amount', 'cibil_score', 'TotalAssets', 'Debt_to_Income']

# Histograms to see distribution

'''numerical_cols = ['income_annum', 'loan_amount', 'cibil_score', 'TotalAssets', 'Debt_to_Income']

# Histograms to see distribution
train_df[numerical_cols].hist(bins=20, figsize=(12,8))
plt.tight_layout()
plt.show()

for col in numerical_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='loan_status', y=col, data=train_df)
    plt.title(f'{col} by Loan Status')
    plt.show()
'''
categorical_cols = ['education', 'self_employed', 'Credit_Category_Good']

for col in categorical_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(x=col, hue='loan_status', data=train_df)
    plt.title(f'{col} count by Loan Status')
    plt.show()

print(train_df.columns.tolist())