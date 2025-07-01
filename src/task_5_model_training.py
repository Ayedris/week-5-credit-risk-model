#!/usr/bin/env python3
# task_5_model_training.py

import pandas as pd
import numpy as np
import mlflow
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from lime.lime_tabular import LimeTabularExplainer
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

def preprocess_data(df):
    """Handle non-numeric columns and missing values"""
    # 1. Drop unnecessary IDs and targets
    df = df.drop([
        'TransactionId', 'BatchId', 'AccountId', 
        'SubscriptionId', 'CustomerId', 'ProviderId', 
        'ProductId', 'FraudResult'  # Explicitly remove target leakage
    ], axis=1)
    
    # 2. Handle TransactionStartTime - create all time features
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['TransactionHour'] = df['TransactionStartTime'].dt.hour
    df['TransactionDay'] = df['TransactionStartTime'].dt.day
    df['TransactionDayOfWeek'] = df['TransactionStartTime'].dt.dayofweek
    df = df.drop('TransactionStartTime', axis=1)
    
    # 3. Convert boolean columns to numeric (0/1)
    bool_cols = ['Outlier_Amount_IQR', 'Outlier_Value_IQR', 
                'Outlier_Amount_Z', 'Outlier_Value_Z']
    for col in bool_cols:
        df[col] = df[col].astype(int)
    
    # 4. Encode categoricals
    cat_cols = ['ProductCategory', 'ChannelId']
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    
    # 5. Handle missing values
    print("\nMissing values before treatment:")
    print(df.isna().sum())
    
    # Fill numerical NaNs with median
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    # Verify
    print("\nMissing values after treatment:")
    print(df.isna().sum())
    
    return df

def main():
    # 1. Load processed data
    df = pd.read_csv(r'C:\Users\ayedr\week-5-credit-risk-model\data\processed\features_with_risk.csv')
    
    # 2. Preprocess data
    print("\n[1] Preprocessing data...")
    df_processed = preprocess_data(df)
    
    # 3. Verify data
    print("\nFinal columns:", df_processed.columns.tolist())
    print("Risk distribution:\n", df_processed['is_high_risk'].value_counts(normalize=True))
    
    # 4. Prepare features/target
    X = df_processed.drop('is_high_risk', axis=1)
    y = df_processed['is_high_risk']
    
    # Enforce feature order and validate
    required_features = [
        'ProductCategory', 'ChannelId', 'Amount', 'Value', 'PricingStrategy',
        'Outlier_Amount_IQR', 'Outlier_Value_IQR', 'Z_Amount', 'Z_Value',
        'Outlier_Amount_Z', 'Outlier_Value_Z', 'Recency', 'Frequency',
        'Monetary', 'TransactionHour', 'TransactionDay', 'TransactionDayOfWeek'
    ]
    
    missing_features = set(required_features) - set(X.columns)
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    
    X = X[required_features]
    
    # 5. Verify no missing values before SMOTE
    assert not X.isna().any().any(), "NaN values still exist in features!"
    assert not y.isna().any(), "NaN values still exist in target!"
    
    # 6. Handle class imbalance
    print("\n[2] Balancing Classes...")
    try:
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        print("Used SMOTE. Balanced distribution:", pd.Series(y_res).value_counts())
    except ValueError as e:
        print(f"SMOTE failed: {e}")
        print("Falling back to RandomOverSampler")
        ros = RandomOverSampler(random_state=42)
        X_res, y_res = ros.fit_resample(X, y)
        print("Used RandomOverSampler. Balanced distribution:", pd.Series(y_res).value_counts())
    
    # 7. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42
    )
    
    # 8. Model Training
    print("\n[3] Starting Model Training...")
    with mlflow.start_run():
        model = RandomForestClassifier(
            class_weight='balanced',
            n_estimators=100,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Log metrics
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
        print(f"Model accuracy: {accuracy:.2f}")
        
        # SHAP Analysis
        print("\n[4] Generating SHAP Explanations...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
        
        # LIME Analysis
        print("\n[5] Generating LIME Explanation...")
        lime_explainer = LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=X_train.columns.tolist(),
            class_names=['low_risk', 'high_risk'],
            mode='classification'
        )
        exp = lime_explainer.explain_instance(
            data_row=X_train.iloc[0].values,
            predict_fn=model.predict_proba,
            num_features=10
        )
        exp.save_to_file('lime_explanation.html')
        
        # Save model
        mlflow.sklearn.log_model(model, "random_forest_model")
        print("\n[6] MLflow Run Details:")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Artifacts: {mlflow.active_run().info.artifact_uri}")
    
    print("\n[7] Training complete!")

if __name__ == "__main__":
    main()