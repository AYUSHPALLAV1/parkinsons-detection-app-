import librosa
import numpy as np
import scipy.signal
import soundfile as sf

def load_audio(file_path, sr=None):
    """Load audio and check quality."""
    try:
        y, sr = librosa.load(file_path, sr=sr)
        # Quality Check: Simple SNR estimate or duration check
        if len(y) < sr * 0.5: # At least 0.5 seconds
            return None, sr
        return y, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def remove_silence(y, top_db=20):
    """Remove silence from audio."""
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trimmed

def normalize_audio(y):
    """Normalize audio amplitude."""
    return librosa.util.normalize(y)

def pre_emphasis(y, coeff=0.97):
    """Apply pre-emphasis filter."""
    return np.append(y[0], y[1:] - coeff * y[:-1])

def apply_windowing(y, frame_length=2048, hop_length=512):
    """Windowing using Hamming window."""
    # Librosa's stft already applies windowing, but for feature extraction we might want frames
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    window = scipy.signal.windows.hamming(frame_length).reshape(-1, 1)
    return frames * window

def preprocess_pipeline(file_path):
    """Complete preprocessing pipeline."""
    y, sr = load_audio(file_path)
    if y is None:
        return None, None
    
    y = remove_silence(y)
    y = normalize_audio(y)
    y = pre_emphasis(y)
    # Windowing is usually part of feature extraction (like MFCC)
    return y, sr
