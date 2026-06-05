import os
import glob
import joblib
import pandas as pd
from sklearn.svm import SVC

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from voice_engine.preprocessing import preprocess_pipeline
from voice_engine.feature_extraction import extract_all_features

def add_temporal_features_local(feat_dict):
    mfcc_means = [feat_dict[f'mfcc_{i}_mean'] for i in range(13)]
    for i in range(len(mfcc_means)-1):
        feat_dict[f'mfcc_diff_{i}'] = mfcc_means[i+1] - mfcc_means[i]
    return feat_dict

HC_DIR = r"c:\Users\ayush\Downloads\fianl mera project\26_29_09_2017_KCL\26-29_09_2017_KCL\ReadText\HC"
PD_DIR = r"c:\Users\ayush\Downloads\fianl mera project\26_29_09_2017_KCL\26-29_09_2017_KCL\ReadText\PD"

hc_files = glob.glob(os.path.join(HC_DIR, "*.wav"))[:3]
pd_files = glob.glob(os.path.join(PD_DIR, "*.wav"))[:3]

FEATURES = []
LABELS = []

print("Processing HC...")
for file in hc_files:
    print(f"Processing {os.path.basename(file)}")
    y, sr = preprocess_pipeline(file)
    if y is not None:
        feat = extract_all_features(y, sr)
        feat = add_temporal_features_local(feat)
        FEATURES.append(feat)
        LABELS.append(0)

print("Processing PD...")
for file in pd_files:
    print(f"Processing {os.path.basename(file)}")
    y, sr = preprocess_pipeline(file)
    if y is not None:
        feat = extract_all_features(y, sr)
        feat = add_temporal_features_local(feat)
        FEATURES.append(feat)
        LABELS.append(1)

df = pd.DataFrame(FEATURES)
voice_scaler = joblib.load(r"c:\Users\ayush\Downloads\fianl mera project\models\voice_scaler.pkl")

expected_columns = getattr(voice_scaler, 'feature_names_in_', None)
if expected_columns is not None:
    for column in expected_columns:
        if column not in df.columns:
            df[column] = 0.0
    df = df.reindex(expected_columns, axis=1).fillna(0.0)

print("Training SVM...")
X_scaled = voice_scaler.transform(df)
svm = SVC(probability=True, kernel='rbf', C=1.0)
svm.fit(X_scaled, LABELS)

joblib.dump(svm, r"c:\Users\ayush\Downloads\fianl mera project\models\voice_svm_model.pkl")
print("Saved voice_svm_model.pkl successfully!")
