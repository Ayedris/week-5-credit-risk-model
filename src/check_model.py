#!/usr/bin/env python3
# check_model.py

import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient

def preprocess_for_inference(df):
    """EXACTLY match training preprocessing"""
    # 1. Drop same columns as training
    df = df.drop([
        'TransactionId', 'BatchId', 'AccountId', 
        'SubscriptionId', 'CustomerId', 'ProviderId',
        'ProductId', 'FraudResult', 'is_high_risk'
    ], axis=1, errors='ignore')
    
    # 2. Process datetime identically to training
    if 'TransactionStartTime' in df.columns:
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        df['TransactionHour'] = df['TransactionStartTime'].dt.hour
        df['TransactionDay'] = df['TransactionStartTime'].dt.day
        df['TransactionDayOfWeek'] = df['TransactionStartTime'].dt.dayofweek
        df = df.drop('TransactionStartTime', axis=1)
    else:
        # Create defaults if missing
        df['TransactionHour'] = 0
        df['TransactionDay'] = 1 
        df['TransactionDayOfWeek'] = 0
    
    # 3. Ensure all expected features exist
    required_features = [
        'ProductCategory', 'ChannelId', 'Amount', 'Value', 'PricingStrategy',
        'Outlier_Amount_IQR', 'Outlier_Value_IQR', 'Z_Amount', 'Z_Value',
        'Outlier_Amount_Z', 'Outlier_Value_Z', 'Recency', 'Frequency',
        'Monetary', 'TransactionHour', 'TransactionDay', 'TransactionDayOfWeek'
    ]
    
    # Add missing features with defaults
    for f in required_features:
        if f not in df.columns:
            df[f] = 0
    
    return df[required_features]

def main():
    # Configure MLflow
    mlflow.set_tracking_uri("file:///C:/Users/ayedr/week-5-credit-risk-model/mlruns")
    
    try:
        # 1. Load the latest model
        client = MlflowClient()
        runs = mlflow.search_runs(order_by=["start_time DESC"])
        if runs.empty:
            raise ValueError("No MLflow runs found!")
            
        latest_run = runs.iloc[0]
        model = mlflow.sklearn.load_model(f"runs:/{latest_run.run_id}/random_forest_model")
        
        # 2. Load and preprocess data
        raw_data = pd.read_csv('data/processed/features_with_risk.csv')
        X = preprocess_for_inference(raw_data)
        
        # 3. Validate feature alignment
        print("\n[1] Feature Validation")
        print(f"Model expects {len(model.feature_names_in_)} features")
        print(f"Data has {len(X.columns)} features")
        
        if list(X.columns) != list(model.feature_names_in_):
            raise ValueError("Feature mismatch!\nModel expects:\n" +
                           f"{model.feature_names_in_}\n\nData has:\n{X.columns}")
            
        # 4. Data leakage check
        print("\n[2] Data Leakage Check")
        print("FraudResult in original data?:", 'FraudResult' in raw_data.columns)
        print("FraudResult in features?:", 'FraudResult' in X.columns)
        
        # 5. Feature importance
        print("\n[3] Feature Importance Analysis")
        importance = pd.Series(model.feature_importances_, 
                             index=model.feature_names_in_)
        importance.sort_values().plot(kind='barh', figsize=(10, 6))
        plt.title('Random Forest Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.show()
        
        print("\n✅ All checks completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nTROUBLESHOOTING:")
        print("1. Verify training script ran successfully")
        print("2. Check feature preprocessing matches exactly")
        print("3. Run 'mlflow ui' to inspect logged models")

if __name__ == "__main__":
    main()