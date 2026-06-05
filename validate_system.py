import os
import sys
import joblib
import torch
import numpy as np
import cv2
import requests

# Add current directory to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

def test_system_with_samples():
    print("=== MULTIMODAL SYSTEM VALIDATION ===")
    
    # Define sample paths
    voice_sample = r"c:\Users\ayush\Downloads\fianl mera project\26_29_09_2017_KCL\26-29_09_2017_KCL\ReadText\PD\ID02_pd_2_0_0.wav"
    # For handwriting, we'll use a placeholder or the first image in the dataset since user provided an image in chat
    hw_sample_dir = r"c:\Users\ayush\Downloads\fianl mera project\Dataset\Dataset\Parkinson"
    hw_sample = None
    if os.path.exists(hw_sample_dir):
        files = [f for f in os.listdir(hw_sample_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if files:
            hw_sample = os.path.join(hw_sample_dir, files[0])

    # 1. Test Voice Model via API
    print(f"\n[1/4] Testing Voice Engine with: {os.path.basename(voice_sample)}")
    voice_prob = 0.0
    if os.path.exists(voice_sample):
        try:
            with open(voice_sample, 'rb') as f:
                files = {'audio': (os.path.basename(voice_sample), f, 'audio/wav')}
                r = requests.post('http://127.0.0.1:5000/api/voice_assessment', files=files)
                res = r.json()
                voice_prob = res.get('probability', 0.0)
                print(f"Result: {voice_prob*100:.2f}% Parkinson's Probability")
        except Exception as e:
            print(f"Error testing voice API: {e}")
    else:
        print("Voice sample not found.")

    # 2. Test Handwriting Model via API
    print(f"\n[2/4] Testing Handwriting Engine with: {os.path.basename(hw_sample) if hw_sample else 'N/A'}")
    hw_prob = 0.0
    if hw_sample and os.path.exists(hw_sample):
        try:
            with open(hw_sample, 'rb') as f:
                files = {'image': (os.path.basename(hw_sample), f, 'image/png')}
                r = requests.post('http://127.0.0.1:5000/api/handwriting_assessment', files=files)
                res = r.json()
                hw_prob = res.get('probability', 0.0)
                print(f"Result: {hw_prob*100:.2f}% Parkinson's Probability")
        except Exception as e:
            print(f"Error testing handwriting API: {e}")
    else:
        print("Handwriting sample not found.")

    # 3. Test Eye Tracking (Simulated Session)
    print("\n[3/4] Testing Eye Tracking Engine (Simulated Data)")
    # Generate 30s of simulated gaze data
    eye_data = []
    for i in range(300): # 10Hz for 30s
        eye_data.append({
            'timestamp': i * 0.1,
            'gaze_x': 0.5 + 0.1 * np.sin(i * 0.1),
            'gaze_y': 0.5 + 0.1 * np.cos(i * 0.1),
            'ear': 0.3 + 0.05 * np.random.randn()
        })
    
    eye_prob = 0.0
    try:
        r = requests.post('http://127.0.0.1:5000/api/eye_assessment', json={'session_data': eye_data})
        res = r.json()
        eye_prob = res.get('probability', 0.0)
        print(f"Result: {eye_prob*100:.2f}% Parkinson's Probability")
    except Exception as e:
        print(f"Error testing eye API: {e}")

    # 4. Test Multimodal Fusion
    print("\n[4/4] Testing Multimodal Fusion Engine")
    try:
        fusion_data = {
            'voice': voice_prob,
            'handwriting': hw_prob,
            'eye': eye_prob
        }
        r = requests.post('http://127.0.0.1:5000/api/fuse_results', json=fusion_data)
        res = r.json()
        print(f"FINAL VERDICT: {res.get('verdict')}")
        print(f"COMBINED CONFIDENCE: {res.get('final_probability')*100:.2f}%")
    except Exception as e:
        print(f"Error testing fusion API: {e}")

if __name__ == "__main__":
    # Ensure Flask is running before starting this script
    test_system_with_samples()
