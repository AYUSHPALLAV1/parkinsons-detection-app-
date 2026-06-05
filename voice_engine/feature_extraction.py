import librosa
import numpy as np
import scipy.stats
import parselmouth # Needs `pip install praat-parselmouth`
from parselmouth.praat import call

def extract_pitch_features(y, sr):
    """Extract pitch-related features using Parselmouth."""
    # Convert librosa array to Parselmouth Sound
    sound = parselmouth.Sound(y, sampling_frequency=sr)
    pitch = sound.to_pitch()
    
    # MDVP-Style Jitter (Vocal Fold Irregularity)
    point_process = call(pitch, "To PointProcess")
    jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_local_absolute = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_ppq5 = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    
    # MDVP-Style Shimmer (Amplitude Modulation)
    shimmer_local = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_local_db = call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq3 = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq5 = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_dda = call([sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    # Harmonic-to-Noise Ratio (HNR)
    harmonicity = sound.to_harmonicity()
    hnr = call(harmonicity, "Get mean", 0, 0)
    
    return {
        'jitter_local': jitter_local,
        'jitter_local_absolute': jitter_local_absolute,
        'jitter_rap': jitter_rap,
        'jitter_ppq5': jitter_ppq5,
        'shimmer_local': shimmer_local,
        'shimmer_local_db': shimmer_local_db,
        'shimmer_apq3': shimmer_apq3,
        'shimmer_apq5': shimmer_apq5,
        'shimmer_dda': shimmer_dda,
        'hnr': hnr
    }

def extract_spectral_features(y, sr):
    """Extract MFCCs and spectral features using librosa."""
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
    """Extract prosodic & temporal features."""
    # Duration, speaking rate (vocalized vs total), pause duration
    # Simple estimate of pause duration using silence detection
    intervals = librosa.effects.split(y, top_db=20)
    total_duration = len(y) / sr
    vocalized_duration = sum([(end - start) for start, end in intervals]) / sr
    pause_duration = total_duration - vocalized_duration
    
    return {
        'vocalized_duration_ratio': vocalized_duration / total_duration if total_duration > 0 else 0,
        'pause_duration': pause_duration,
        'speaking_rate': len(intervals) / total_duration if total_duration > 0 else 0
    }

def extract_all_features(y, sr):
    """Combine all feature extraction steps."""
    features = {}
    features.update(extract_pitch_features(y, sr))
    features.update(extract_spectral_features(y, sr))
    features.update(extract_prosodic_features(y, sr))
    return features
