import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import cv2

class LaneDetectionCNN(nn.Module):
    def __init__(self, input_shape):
        super(LaneDetectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.dropout = nn.Dropout(0.5)

        # Calculate flat size dynamically
        self._to_linear = None
        self._calculate_flat_size(input_shape)

        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, 1)  # Single output neuron for regression

    def _calculate_flat_size(self, input_shape):
        x = torch.zeros(1, *input_shape)
        x = self._forward_conv(x)
        self._to_linear = x.numel()

    def _forward_conv(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1) 
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training function
def train_model(model, dataloader, criterion, optimizer, n_epochs=10):
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for Xbatch, ybatch in dataloader:
            optimizer.zero_grad()
            y_pred = model(Xbatch)
            loss = criterion(y_pred, ybatch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(dataloader):.4f}")


if __name__ == "__main__":
    # Dummy training data
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((120, 160)),
        transforms.ToTensor()
    ])

    # Simulated dataset (replace with real data)
    X_dummy = torch.rand(10, 3, 120, 160)  # 10 RGB images
    y_dummy = torch.rand(10, 1) * 40 - 20  # Random values between -20 and 20

    dataset = TensorDataset(X_dummy, y_dummy)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    input_shape = (3, 120, 160)  # Input image size (3 channels, height, width)
    model = LaneDetectionCNN(input_shape)

    criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, dataloader, criterion, optimizer, n_epochs=5)

    