import os
import sys
import tempfile
from flask import Flask, render_template, request, jsonify, send_from_directory
from datetime import datetime
import joblib
import torch
import numpy as np
import pandas as pd
import cv2

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from voice_engine.preprocessing import preprocess_pipeline
from voice_engine.feature_extraction import extract_all_features
from handwriting_engine.preprocessing import preprocess_image
from handwriting_engine.model import HandwritingCNN, load_model
from eye_tracking.feature_extraction import extract_advanced_eye_features
from fusion_engine.fusion import MultimodalFusionEngine
from backend.database import DatabaseManager

app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'frontend', 'templates'),
            static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'frontend', 'static'))

# Initialize components
db = DatabaseManager()
fusion_engine = MultimodalFusionEngine()

# Model paths - using the best models from Colab
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')

VOICE_MODEL_PATH = os.path.join(MODELS_DIR, 'voice_xgb_model.pkl')
VOICE_SCALER_PATH = os.path.join(MODELS_DIR, 'voice_scaler.pkl')
HANDWRITING_MODEL_PATH = os.path.join(MODELS_DIR, 'handwriting_cnn_model.pth')

# Load models with error handling
voice_model = None
voice_scaler = None
try:
    if os.path.exists(VOICE_MODEL_PATH):
        voice_model = joblib.load(VOICE_MODEL_PATH)
        voice_scaler = joblib.load(VOICE_SCALER_PATH)
        print("Voice model (XGBoost) loaded successfully.")
    else:
        # Fallback to SVM if XGBoost not found
        SVM_PATH = os.path.join(MODELS_DIR, 'voice_svm_model.pkl')
        if os.path.exists(SVM_PATH):
            voice_model = joblib.load(SVM_PATH)
            voice_scaler = joblib.load(VOICE_SCALER_PATH)
            print("Voice model (SVM) loaded successfully.")
except Exception as e:
    print(f"Error loading voice model: {e}")

handwriting_model = None
try:
    if os.path.exists(HANDWRITING_MODEL_PATH):
        handwriting_model = load_model(HANDWRITING_MODEL_PATH)
        print("Handwriting CNN model loaded successfully.")
except Exception as e:
    print(f"Error loading handwriting model: {e}")

def json_error(message, status_code=400):
    return jsonify({'error': message}), status_code

def align_features_for_model(df_features, scaler):
    """Align runtime features with the exact training schema when available."""
    expected_columns = getattr(scaler, 'feature_names_in_', None)
    if expected_columns is None:
        return df_features.reindex(sorted(df_features.columns), axis=1)

    missing_columns = [column for column in expected_columns if column not in df_features.columns]
    for column in missing_columns:
        df_features[column] = 0.0

    return df_features.reindex(expected_columns, axis=1).fillna(0.0)

def create_temp_upload(suffix):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_path = temp_file.name
    temp_file.close()
    return temp_path

def normalize_probability(value):
    if value is None:
        return None

    try:
        return min(max(float(value), 0.0), 1.0)
    except (TypeError, ValueError):
        return None

def collect_prediction_inputs(data):
    predictions = {}
    for key in ('voice', 'eye', 'handwriting'):
        normalized = normalize_probability(data.get(key))
        if normalized is not None:
            predictions[key] = normalized
    return predictions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/assessment')
def assessment():
    return render_template('assessment.html')

@app.route('/dashboard')
def dashboard():
    history = db.get_history()
    return render_template('dashboard.html', history=history)

@app.route('/api/voice_assessment', methods=['POST'])
def voice_assessment():
    """Handle voice assessment request."""
    if 'audio' not in request.files:
        return json_error('No audio file provided')
    if not voice_model or not voice_scaler:
        return json_error('Voice model is unavailable.', 503)
    
    audio_file = request.files['audio']
    temp_path = create_temp_upload('.wav')
    audio_file.save(temp_path)
    
    try:
        y, sr = preprocess_pipeline(temp_path)
        if y is None:
            return json_error('Invalid audio quality')
            
        features = extract_all_features(y, sr)
        
        # Add temporal features (MFCC diffs) to match colab training
        def add_temporal_features_local(feat_dict):
            mfcc_means = [feat_dict[f'mfcc_{i}_mean'] for i in range(13)]
            for i in range(len(mfcc_means)-1):
                feat_dict[f'mfcc_diff_{i}'] = mfcc_means[i+1] - mfcc_means[i]
            return feat_dict

        features = add_temporal_features_local(features)
        
        # Convert features to DataFrame with correct feature names to match training
        df_features = pd.DataFrame([features])
        df_features = align_features_for_model(df_features, voice_scaler)
        
        print(f"Features extracted: {len(df_features.columns)}")

        X_scaled = voice_scaler.transform(df_features)
        prob = voice_model.predict_proba(X_scaled)[0][1]
        return jsonify({'probability': normalize_probability(prob)})
    except Exception as e:
        print(f"Voice assessment error: {e}")
        return json_error(str(e), 500)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/api/handwriting_assessment', methods=['POST'])
def handwriting_assessment():
    """Handle handwriting assessment request."""
    if 'image' not in request.files:
        return json_error('No image provided')
    if not handwriting_model:
        return json_error('Handwriting model is unavailable.', 503)
    
    image_file = request.files['image']
    temp_path = create_temp_upload('.png')
    image_file.save(temp_path)
    
    try:
        processed_img = preprocess_image(temp_path)
        if processed_img is None:
            return json_error('Invalid image')

        img_tensor = torch.tensor(processed_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            prob = handwriting_model(img_tensor).item()

        return jsonify({'probability': normalize_probability(prob)})
    except Exception as e:
        return json_error(str(e), 500)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/api/eye_assessment', methods=['POST'])
def eye_assessment():
    """Handle eye tracking data assessment request."""
    data = request.json
    if not data or 'session_data' not in data:
        return json_error('No session data provided')
    
    try:
        if not isinstance(data['session_data'], list) or len(data['session_data']) < 2:
            return json_error('Session data must contain at least two samples.')

        # Extract 25 clinical oculomotor features
        features = extract_advanced_eye_features(data['session_data'])
        
        # Simple heuristic for probability based on key clinical markers:
        # 1. High SWJ count
        # 2. Low pursuit gain
        # 3. High fixation dispersion
        # 4. Low blink rate
        
        score = 0.0
        if features.get('swj_count', 0) > 5: score += 0.3
        if features.get('pursuit_gain', 1.0) < 0.7: score += 0.3
        if features.get('blink_rate', 15) < 8: score += 0.2
        if features.get('fix_dispersion', 0) > 0.05: score += 0.2
        
        prob = min(max(score, 0.0), 0.95)
        
        return jsonify({
            'probability': float(prob),
            'features': features
        })
    except Exception as e:
        return json_error(str(e), 500)

@app.route('/api/fuse_results', methods=['POST'])
def fuse_results():
    """Combine all module predictions into final verdict."""
    data = request.json
    if not data:
        return json_error('No data provided')

    predictions = collect_prediction_inputs(data)
    if not predictions:
        return json_error('At least one valid module probability is required.')

    final_prob = fusion_engine.fuse(predictions)
    verdict = fusion_engine.get_clinical_verdict(final_prob)
    
    # Save to DB
    db.save_assessment(
        voice_prob=predictions.get('voice'),
        eye_prob=predictions.get('eye'),
        handwriting_prob=predictions.get('handwriting'),
        final_prob=final_prob,
        verdict=verdict
    )
    
    return jsonify({
        'final_probability': float(final_prob),
        'verdict': verdict
    })

# Route to serve animation frames from the specific local directory
@app.route('/frames/<filename>')
def serve_frame(filename):
    # Using absolute path from project root for reliability
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    frames_dir = os.path.join(base_dir, 'ezgif-8a104e2ac92bc98b-jpg')
    return send_from_directory(frames_dir, filename)

if __name__ == '__main__':
    print("Starting Flask application...")
    print("TIP: Use http://localhost:5000 in your browser for full hardware permissions (camera/mic).")
    app.run(debug=True, port=5000)
