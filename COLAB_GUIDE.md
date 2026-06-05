# Google Colab Training Guide for Parkinson's Early Detection System

This guide provides the necessary steps and code cells to train your models on Google Colab using your datasets uploaded to Google Drive.

## Step 1: Upload Datasets to Google Drive
Ensure your datasets are uploaded to your Google Drive in the following structure:
- `My Drive/ParkinsonsProject/26_29_09_2017_KCL/`
- `My Drive/ParkinsonsProject/Dataset/Dataset/`

## Step 2: Open Google Colab and Mount Drive
Run the following in the first cell:
```python
from google.colab import drive
drive.mount('/content/drive')
```

## Step 3: Install Dependencies
```python
!pip install librosa praat-parselmouth mediapipe opencv-python scikit-learn scikit-image torch torchvision joblib tqdm xgboost
```

## Step 4: Run Voice Training
Create a new cell and paste the code from `voice_engine/colab_voice_training.py`. This script will:
- Extract MDVP-style features, spectral features, and prosodic features.
- Perform a Stratified Split (80/20).
- Train SVM (RBF), Random Forest, and XGBoost.
- Save the best model and scaler.

## Step 5: Run Handwriting Training
Create a new cell and paste the code from `handwriting_engine/colab_handwriting_training.py`. This script will:
- Preprocess spiral images.
- Train a CNN using PyTorch (30-50 epochs recommended).
- Extract classical features for an SVM fallback.
- Save the models.

## Step 6: Download Models
After training, the models will be saved to your Drive. Download them and place them in your local `models/` directory:
- `voice_svm_model.pkl`
- `voice_scaler.pkl`
- `handwriting_cnn_model.pth`
- `handwriting_scaler.pkl` (if applicable)
