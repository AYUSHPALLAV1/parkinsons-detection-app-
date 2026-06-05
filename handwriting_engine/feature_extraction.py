import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops # Needs `pip install scikit-image`

def extract_spiral_area_ratio(image, contours):
    """Spiral area / bounding box area."""
    if not contours:
        return 0
    
    # Get the largest contour (the spiral)
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    
    # Bounding box
    x, y, w, h = cv2.boundingRect(c)
    bbox_area = w * h
    
    return area / bbox_area if bbox_area > 0 else 0

def extract_hu_moments(image):
    """7 rotation-invariant shape moments."""
    # Ensure uint8
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    moments = cv2.moments(image)
    hu = cv2.HuMoments(moments)
    # Log transform for better numerical stability
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    return hu_log.flatten()

def extract_glcm_texture(image):
    """Contrast, energy, homogeneity, correlation."""
    # Ensure uint8
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Calculate GLCM
    glcm = graycomatrix(image, [1], [0], 256, symmetric=True, normed=True)
    
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    
    return {
        'glcm_contrast': contrast,
        'glcm_energy': energy,
        'glcm_homogeneity': homogeneity,
        'glcm_correlation': correlation
    }

def extract_all_handwriting_features(image):
    """Extract all classical handwriting features from a preprocessed image."""
    # Ensure uint8 for contour extraction
    if image.max() <= 1.0:
        img_u8 = (image * 255).astype(np.uint8)
    else:
        img_u8 = image.astype(np.uint8)
        
    contours, _ = cv2.findContours(img_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    features = {}
    features['spiral_area_ratio'] = extract_spiral_area_ratio(img_u8, contours)
    
    hu = extract_hu_moments(img_u8)
    for i, val in enumerate(hu):
        features[f'hu_moment_{i}'] = val
        
    features.update(extract_glcm_texture(img_u8))
    
    return features
