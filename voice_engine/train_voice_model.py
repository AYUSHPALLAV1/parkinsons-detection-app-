import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def train_voice_model(csv_path):
    """Train voice models using SVM, RF, and potentially others."""
    df = pd.read_csv(csv_path)
    
    # Drop non-feature columns
    X = df.drop(['label', 'subject_id', 'subset'], axis=1)
    y = df['label']
    
    # Fill NaN values (e.g., from Parselmouth if pitch not found)
    X = X.fillna(X.mean())
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. SVM (RBF Kernel) - Recommended as first choice
    print("Training SVM...")
    svm_param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf']
    }
    svm_grid = GridSearchCV(SVC(probability=True), svm_param_grid, refit=True, verbose=0, cv=5)
    svm_grid.fit(X_train_scaled, y_train)
    
    svm_pred = svm_grid.predict(X_test_scaled)
    print(f"SVM Accuracy: {accuracy_score(y_test, svm_pred):.4f}")
    print(classification_report(y_test, svm_pred))
    
    # 2. Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    print(f"RF Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
    
    # Save the best model (SVM) and the scaler
    os.makedirs('../models', exist_ok=True)
    joblib.dump(svm_grid.best_estimator_, '../models/voice_svm_model.pkl')
    joblib.dump(scaler, '../models/voice_scaler.pkl')
    print("Best model and scaler saved to models/ folder.")
    
    return svm_grid.best_estimator_, scaler

if __name__ == "__main__":
    if os.path.exists('voice_features.csv'):
        train_voice_model('voice_features.csv')
    else:
        print("voice_features.csv not found. Please run process_dataset.py first.")
