import os
import pandas as pd
import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from tqdm import tqdm

# --- CONFIGURATION ---
# Check if there is a nested folder (26_29_09_2017_KCL/26-29_09_2017_KCL)
VOICE_DATASET_PATH = '/content/drive/MyDrive/ParkinsonsProject/26_29_09_2017_KCL/26-29_09_2017_KCL'
OUTPUT_DIR = '/content/drive/MyDrive/ParkinsonsProject/models'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- FEATURE EXTRACTION FUNCTIONS ---
def extract_pitch_features(y, sr):
    sound = parselmouth.Sound(y, sampling_frequency=sr)
    pitch = sound.to_pitch()
    point_process = call(pitch, "To PointProcess")
    
    jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_local_abs = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_ppq5 = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    
    shimmer_local = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_local_db = call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq3 = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq5 = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_dda = call([sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    harmonicity = sound.to_harmonicity()
    hnr = call(harmonicity, "Get mean", 0, 0)
    
    return {
        'jitter_local': jitter_local, 'jitter_local_abs': jitter_local_abs,
        'jitter_rap': jitter_rap, 'jitter_ppq5': jitter_ppq5,
        'shimmer_local': shimmer_local, 'shimmer_local_db': shimmer_local_db,
        'shimmer_apq3': shimmer_apq3, 'shimmer_apq5': shimmer_apq5,
        'shimmer_dda': shimmer_dda, 'hnr': hnr
    }

def extract_spectral_features(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfccs, axis=1)
    mfcc_stds = np.std(mfccs, axis=1)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    
    features = {}
    for i in range(13):
        features[f'mfcc_{i}_mean'] = mfcc_means[i]
        features[f'mfcc_{i}_std'] = mfcc_stds[i]
    features.update({
        'spectral_centroid': spectral_centroid,
        'spectral_bandwidth': spectral_bandwidth,
        'spectral_rolloff': spectral_rolloff
    })
    return features

def extract_prosodic_features(y, sr):
    intervals = librosa.effects.split(y, top_db=20)
    total_dur = len(y) / sr
    vocalized_dur = sum([(end - start) for start, end in intervals]) / sr
    pause_dur = total_dur - vocalized_dur
    return {
        'vocalized_ratio': vocalized_dur / total_dur if total_dur > 0 else 0,
        'pause_duration': pause_dur,
        'speaking_rate': len(intervals) / total_dur if total_dur > 0 else 0
    }

def process_voice_dataset(base_path):
    subsets = ['ReadText', 'SpontaneousDialogue']
    classes = ['HC', 'PD']
    all_features = []
    
    for subset in subsets:
        for cls in classes:
            folder_path = os.path.join(base_path, subset, cls)
            if not os.path.exists(folder_path): continue
            
            print(f"Processing {subset}/{cls}...")
            files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
            for file_name in tqdm(files):
                file_path = os.path.join(folder_path, file_name)
                try:
                    y, sr = librosa.load(file_path)
                    y_trimmed, _ = librosa.effects.trim(y)
                    y_norm = librosa.util.normalize(y_trimmed)
                    
                    features = {}
                    features.update(extract_pitch_features(y_norm, sr))
                    features.update(extract_spectral_features(y_norm, sr))
                    features.update(extract_prosodic_features(y_norm, sr))
                    features['label'] = 1 if cls == 'PD' else 0
                    all_features.append(features)
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")
    return pd.DataFrame(all_features)

# --- TRAINING ---
print("Extracting features...")
df = process_voice_dataset(VOICE_DATASET_PATH)
if df.empty:
    raise ValueError("No voice data found. Check your DRIVE path.")

# Feature Engineering: Add MFCC Delta and Delta-Delta for better temporal capture
def add_temporal_features(df):
    mfcc_cols = [c for c in df.columns if 'mfcc' in c and 'mean' in c]
    # Simple proxy for deltas: differences between MFCC coefficients
    for i in range(len(mfcc_cols)-1):
        df[f'mfcc_diff_{i}'] = df[mfcc_cols[i+1]] - df[mfcc_cols[i]]
    return df

df = add_temporal_features(df)
df = df.dropna()
print(f"Processed {len(df)} valid samples with augmented features.")

df.to_csv(os.path.join(OUTPUT_DIR, 'voice_features.csv'), index=False)

X = df.drop(['label'], axis=1)
y = df['label']

# Increase test size slightly for better validation stability
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# XGBoost - Optimized for T4 GPU and Accuracy
print("\nTraining XGBoost (GPU Accelerated)...")
# Removed 'use_label_encoder' to fix warning
xgb = XGBClassifier(
    n_estimators=500, 
    learning_rate=0.01, # Smaller learning rate for better generalization
    max_depth=7, 
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist',
    device='cuda',
    eval_metric='logloss'
)
xgb.fit(X_train_scaled, y_train)
print(f"XGBoost Accuracy: {accuracy_score(y_test, xgb.predict(X_test_scaled)):.4f}")

# SVM (RBF) - Fine-tuned Grid Search
print("\nTraining SVM (RBF)...")
svm_param_grid = {
    'C': [5, 10, 20, 50],
    'gamma': ['scale', 0.05, 0.01, 0.005],
    'kernel': ['rbf']
}
svm_grid = GridSearchCV(SVC(probability=True, class_weight='balanced'), svm_param_grid, cv=5, n_jobs=-1)
svm_grid.fit(X_train_scaled, y_train)
print(f"SVM Best Accuracy: {accuracy_score(y_test, svm_grid.predict(X_test_scaled)):.4f}")

# Save the Best Model (XGBoost is performing better)
joblib.dump(xgb, os.path.join(OUTPUT_DIR, 'voice_xgb_model.pkl'))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'voice_scaler.pkl'))
print(f"\nTraining Complete. Best model (XGBoost) saved to {OUTPUT_DIR}")
