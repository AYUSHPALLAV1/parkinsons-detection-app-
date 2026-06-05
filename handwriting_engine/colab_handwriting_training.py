import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import graycomatrix, graycoprops
import joblib
from tqdm import tqdm

# --- CONFIGURATION ---
HW_DATASET_PATH = '/content/drive/MyDrive/ParkinsonsProject/Dataset/Dataset'
OUTPUT_DIR = '/content/drive/MyDrive/ParkinsonsProject/models'
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# --- CNN ARCHITECTURE ---
class HandwritingCNN(nn.Module):
    def __init__(self):
        super(HandwritingCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

# --- DATASET LOADER ---
class HandwritingDataset(Dataset):
    def __init__(self, file_list, labels, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.file_list)
        
    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Return a blank image if loading fails
            img = np.zeros((224, 224), dtype=np.uint8)
        else:
            img = cv2.resize(img, (224, 224))
        
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        img = img.astype(np.float32) / 255.0
        
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(img).unsqueeze(0) # (1, 224, 224)
            
        return img, torch.tensor([self.labels[idx]], dtype=torch.float32)

# --- CLASSICAL FEATURE EXTRACTION ---
def extract_classical_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return [0] * 5
    img = cv2.resize(img, (224, 224))
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Area ratio
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        area_ratio = area / (w * h) if w*h > 0 else 0
    else:
        area_ratio = 0
        
    # Texture (GLCM) - use smaller levels to speed up
    glcm = graycomatrix(thresh, [1], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    
    return [area_ratio, contrast, energy, homogeneity, correlation]

# --- PREPARE DATA ---
print(f"Searching for dataset in: {HW_DATASET_PATH}")
if not os.path.exists(HW_DATASET_PATH):
    print(f"ERROR: Path does not exist: {HW_DATASET_PATH}")
    print("Contents of /content/drive/MyDrive/ParkinsonsProject/ :", os.listdir('/content/drive/MyDrive/ParkinsonsProject/'))
else:
    print(f"Contents of {HW_DATASET_PATH}:", os.listdir(HW_DATASET_PATH))

file_list = []
labels = []

# First, collect all file paths (fast)
for cls_name, label in [('Healthy', 0), ('Parkinson', 1)]:
    folder = os.path.join(HW_DATASET_PATH, cls_name)
    if not os.path.exists(folder):
        print(f"Warning: Folder not found: {folder}")
        continue
    
    print(f"Scanning {cls_name} folder...")
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for f in files:
        file_list.append(os.path.join(folder, f))
        labels.append(label)

print(f"Total samples found: {len(file_list)}")

# Now, extract classical features (slow, with progress bar)
print("Extracting classical features...")
classical_features = []
for path in tqdm(file_list):
    classical_features.append(extract_classical_features(path))

# --- TRAIN CNN ---
if len(file_list) == 0:
    raise ValueError("No images found. Please check your HW_DATASET_PATH and folder names (Healthy/Parkinson).")

X_train_f, X_test_f, y_train, y_test = train_test_split(file_list, labels, test_size=0.2, random_state=42, stratify=labels)
train_dataset = HandwritingDataset(X_train_f, y_train)
test_dataset = HandwritingDataset(X_test_f, y_test)

# Optimized for T4 GPU and slow Drive I/O: reduce workers if it hangs
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, num_workers=1, pin_memory=True)

model = HandwritingCNN().to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\nTraining CNN (PyTorch)...")
epochs = 30
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for imgs, targets in train_loader:
        imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Eval
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            outputs = model(imgs)
            predicted = (outputs > 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {100*correct/total:.2f}%")

# --- TRAIN SVM (FALLBACK) ---
print("\nTraining SVM (Classical Features)...")
X_class = np.array(classical_features)
y_class = np.array(labels)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, random_state=42, stratify=y_class)

scaler = StandardScaler()
X_train_c_scaled = scaler.fit_transform(X_train_c)
X_test_c_scaled = scaler.transform(X_test_c)

svm = SVC(probability=True)
svm.fit(X_train_c_scaled, y_train_c)
print(f"SVM Accuracy: {svm.score(X_test_c_scaled, y_test_c):.4f}")

# --- SAVE MODELS ---
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'handwriting_cnn_model.pth'))
joblib.dump(svm, os.path.join(OUTPUT_DIR, 'handwriting_svm_model.pkl'))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'handwriting_scaler.pkl'))
print(f"\nTraining Complete. Models saved to {OUTPUT_DIR}")
