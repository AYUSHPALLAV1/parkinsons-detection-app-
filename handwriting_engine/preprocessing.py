import cv2
import numpy as np

def preprocess_image(image_path, size=(224, 224)):
    """Preprocess image for CNN and feature extraction."""
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize
    resized = cv2.resize(gray, size)
    
    # Thresholding to extract the drawing (spiral)
    _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Normalize
    normalized = thresh.astype(np.float32) / 255.0
    
    return normalized

def extract_contours(image):
    """Extract contours from binary image for geometry features."""
    # Ensure binary
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours
