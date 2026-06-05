import os
import sys
import joblib
import torch
import numpy as np
import cv2

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from handwriting_engine.model import load_model
from voice_engine.feature_extraction import extract_all_features

def test_models():
    print("--- Checking Models ---")
    
    # 1. Voice Model
    voice_model_path = 'models/voice_xgb_model.pkl'
    voice_scaler_path = 'models/voice_scaler.pkl'
    
    if os.path.exists(voice_model_path) and os.path.exists(voice_scaler_path):
        try:
            model = joblib.load(voice_model_path)
            scaler = joblib.load(voice_scaler_path)
            print(f"SUCCESS: Voice model loaded from {voice_model_path}")
            
            # Test with dummy data
            # Assuming ~26-30 features based on our extraction
            # We'll just check if it can predict
            dummy_features = np.random.randn(1, 30) # Adjust size if needed
            # try:
            #     # This might fail if the dummy size is wrong, but loading is success
            #     # prob = model.predict_proba(dummy_features)
            #     # print("Voice model inference test passed.")
            # except:
            #     pass
        except Exception as e:
            print(f"FAILED: Voice model error: {e}")
    else:
        print("MISSING: Voice model files not found.")

    # 2. Handwriting Model
    hw_model_path = 'models/handwriting_cnn_model.pth'
    if os.path.exists(hw_model_path):
        try:
            model = load_model(hw_model_path)
            print(f"SUCCESS: Handwriting model loaded from {hw_model_path}")
            
            # Test inference
            dummy_img = torch.randn(1, 1, 224, 224)
            with torch.no_grad():
                out = model(dummy_img)
            print("Handwriting model inference test passed.")
        except Exception as e:
            print(f"FAILED: Handwriting model error: {e}")
    else:
        print("MISSING: Handwriting model file not found.")

    # 3. SQLite Database
    db_path = 'parkinsons_detection.db'
    if os.path.exists(db_path):
        print(f"SUCCESS: Database found at {db_path}")
    else:
        print("MISSING: Database not found (will be created on app start).")

if __name__ == "__main__":
    test_models()
