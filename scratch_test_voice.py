import sys
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from voice_engine.preprocessing import preprocess_pipeline
from voice_engine.feature_extraction import extract_all_features

file_path = r"c:\Users\ayush\Downloads\fianl mera project\26_29_09_2017_KCL\26-29_09_2017_KCL\ReadText\PD\ID02_pd_2_0_0.wav"
y, sr = preprocess_pipeline(file_path)
features = extract_all_features(y, sr)

import joblib
import pandas as pd

features_df = pd.DataFrame([features])
voice_scaler = joblib.load(r"c:\Users\ayush\Downloads\fianl mera project\models\voice_scaler.pkl")
voice_model = joblib.load(r"c:\Users\ayush\Downloads\fianl mera project\models\voice_xgb_model.pkl")

expected_columns = getattr(voice_scaler, 'feature_names_in_', None)
if expected_columns is not None:
    for column in expected_columns:
        if column not in features_df.columns:
            features_df[column] = 0.0
    features_df = features_df.reindex(expected_columns, axis=1).fillna(0.0)

print("Predicting with XGBoost...")
X_scaled = voice_scaler.transform(features_df)
prob = voice_model.predict_proba(X_scaled)
print(f"Prediction: {prob}")
