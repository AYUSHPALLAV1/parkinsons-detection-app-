import torch
import torch.nn as nn
import torch.nn.functional as F

class HandwritingCNN(nn.Module):
    """Simple CNN architecture for spiral drawing classification."""
    def __init__(self):
        super(HandwritingCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # Input image is 224x224. After 3 pooling layers: 224/2/2/2 = 28x28
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1) # Sigmoid for binary classification
        
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        # Convolutional blocks
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(-1, 128 * 28 * 28)
        
        # Fully connected blocks
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        
        return x

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(path, device='cpu'):
    model = HandwritingCNN()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
