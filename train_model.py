#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train XGBoost model to predict heart failure patient survival
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
import os
import subprocess

def download_dataset():
    """Download the heart failure dataset if it doesn't exist"""
    if not os.path.exists('heart_failure_clinical_records_dataset.csv'):
        print("Downloading dataset...")
        subprocess.run(["wget", "-q", "https://cdn.iisc.talentsprint.com/CDS/Datasets/heart_failure_clinical_records_dataset.csv"])
        print("Dataset downloaded successfully.")
    else:
        print("Dataset already exists.")

def handle_outliers(df, colm):
    """Change the values of outlier to upper and lower whisker values"""
    q1 = df.describe()[colm].loc["25%"]
    q3 = df.describe()[colm].loc["75%"]
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    for i in range(len(df)):
        if df.loc[i, colm] > upper_bound:
            df.loc[i, colm] = upper_bound
        if df.loc[i, colm] < lower_bound:
            df.loc[i, colm] = lower_bound
    return df

def preprocess_data(df):
    """Handle outliers in the dataset"""
    print("Preprocessing data...")
    outlier_colms = ['creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']
    for colm in outlier_colms:
        df = handle_outliers(df, colm)
    return df

def train_model(X_train, y_train):
    """Train XGBoost classifier"""
    print("Training model...")
    xgb_clf = XGBClassifier(n_estimators=200, max_depth=4, max_leaves=5, random_state=42)
    xgb_clf.fit(X_train, y_train)
    return xgb_clf

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate model performance"""
    print("Evaluating model performance...")
    # Accuracy
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Testing accuracy: {test_acc:.4f}")

    # F1-score
    train_f1 = f1_score(y_train, model.predict(X_train))
    test_f1 = f1_score(y_test, model.predict(X_test))
    print(f"Training F1 score: {train_f1:.4f}")
    print(f"Testing F1 score: {test_f1:.4f}")

def save_model(model, filename="xgboost-model.pkl"):
    """Save trained model to file"""
    print(f"Saving model to {filename}...")
    joblib.dump(model, filename)
    print("Model saved successfully.")

def main():
    """Main function to execute the training pipeline"""
    # Download dataset
    # download_dataset()
    
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
    print(f"Dataset shape: {df.shape}")
    
    # Preprocess data
    df_processed = preprocess_data(df)
    
    # Split dataset
    print("Splitting data into train and test sets...")
    X = df_processed.iloc[:, :-1].values
    y = df_processed['DEATH_EVENT'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=123)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_train, y_train, X_test, y_test)
    
    # Save model
    save_model(model)

if __name__ == "__main__":
    main()