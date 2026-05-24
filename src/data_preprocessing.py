import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_clean_data(filepath):
    # Load dataset
    df = pd.read_csv(filepath)
    
    # Drop CustomerID as it's useless for prediction
    df.drop(['customerID'], axis=1, inplace=True, errors='ignore')
    
    # TotalCharges has some blank spaces; convert to numeric and handle NaNs
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() == 2:
            df[col] = le.fit_transform(df[col])
        else:
            # One-hot encode multi-class columns
            df = pd.get_dummies(df, columns=[col], drop_first=True)
            
    return df

def prepare_datasets(df, target_col='Churn'):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split into Train and Test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test