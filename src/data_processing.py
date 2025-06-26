import pandas as pd
import numpy as np
import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# --- 1. Load cleaned data ---
input_path = 'data/processed/cleaned_data.csv'
df = pd.read_csv(input_path)

# --- 2. Convert datetime ---
df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], utc=True)
df['hour'] = df['TransactionStartTime'].dt.hour
df['day'] = df['TransactionStartTime'].dt.day
df['day_of_week'] = df['TransactionStartTime'].dt.dayofweek
df['month'] = df['TransactionStartTime'].dt.month
df['year'] = df['TransactionStartTime'].dt.year

# --- 3. Create Aggregate Features ---
agg = df.groupby('CustomerId')['Amount'].agg([
    ('total_amount_per_customer', 'sum'),
    ('avg_amount_per_customer', 'mean'),
    ('transaction_count', 'count'),
    ('std_amount_per_customer', 'std')
]).reset_index()

# Merge back to df
df = df.merge(agg, on='CustomerId', how='left')

# --- 4. Drop high-cardinality / redundant IDs ---
df.drop(columns=['TransactionId', 'BatchId', 'SubscriptionId', 'TransactionStartTime'], inplace=True)

# --- 5. Encode categorical variables ---
label_cols = ['AccountId', 'CustomerId', 'ProviderId', 'ProductId']
onehot_cols = ['ProductCategory', 'ChannelId']



# Apply label encoding
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Prepare for pipeline
numeric_features = ['Amount', 'Value', 'hour', 'day', 'day_of_week', 'month', 'year',
                    'total_amount_per_customer', 'avg_amount_per_customer', 
                    'transaction_count', 'std_amount_per_customer']
categorical_features = onehot_cols

# --- 6. Pipeline ---
numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Apply transformation
features_array = preprocessor.fit_transform(df)
features_columns = (
    numeric_features +
    list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(onehot_cols))
)

# Create final DataFrame
X = pd.DataFrame(features_array, columns=features_columns)

# Optionally include target
if 'FraudResult' in df.columns:
    X['FraudResult'] = df['FraudResult'].values

# --- 7. Save processed data ---
output_path = 'data/processed/features.csv'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
X.to_csv(output_path, index=False)

print(f"✅ Model-ready features saved to: {output_path}")
