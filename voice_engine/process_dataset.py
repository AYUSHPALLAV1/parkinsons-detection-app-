import os
import pandas as pd
import numpy as np
from preprocessing import preprocess_pipeline
from feature_extraction import extract_all_features
from tqdm import tqdm

def process_voice_dataset(base_path):
    """Iterate through the voice dataset and extract features for all samples."""
    # The dataset has directories: ReadText/HC, ReadText/PD, SpontaneousDialogue/HC, SpontaneousDialogue/PD
    subsets = ['ReadText', 'SpontaneousDialogue']
    classes = ['HC', 'PD']
    
    all_features = []
    
    for subset in subsets:
        for cls in classes:
            folder_path = os.path.join(base_path, '26-29_09_2017_KCL', subset, cls)
            if not os.path.exists(folder_path):
                print(f"Folder not found: {folder_path}")
                continue
            
            print(f"Processing {subset}/{cls}...")
            files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
            
            for file_name in tqdm(files):
                file_path = os.path.join(folder_path, file_name)
                
                # Preprocess
                y, sr = preprocess_pipeline(file_path)
                if y is None:
                    continue
                
                # Extract features
                try:
                    features = extract_all_features(y, sr)
                    features['label'] = 1 if cls == 'PD' else 0 # PD=1, HC=0
                    features['subject_id'] = file_name.split('_')[0]
                    features['subset'] = subset
                    all_features.append(features)
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")
    
    df = pd.DataFrame(all_features)
    df.to_csv('voice_features.csv', index=False)
    print("Feature extraction complete. Saved to voice_features.csv.")
    return df

if __name__ == "__main__":
    dataset_path = r'c:\Users\ayush\Downloads\fianl mera project\26_29_09_2017_KCL'
    process_voice_dataset(dataset_path)
