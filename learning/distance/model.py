import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
import os
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
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
        x = torch.tanh(self.fc2(x))
        return x

# Training function
def train_model(model, dataloader, criterion, optimizer, n_epochs=10):
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for Xbatch, ybatch in dataloader:
            # Move inputs and labels to device
            Xbatch, ybatch = Xbatch.to(device), ybatch.to(device)

            optimizer.zero_grad()
            y_pred = model(Xbatch)
            loss = criterion(y_pred, ybatch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(dataloader):.4f}")


class ImageDataset(Dataset):
    def __init__(self, image_folder, label_folder, transform=None):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.image_files = sorted(os.listdir(image_folder))  # Ensure consistent order
        self.label_files = sorted(os.listdir(label_folder))  # Ensure consistent order

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Load corresponding label
        label_path = os.path.join(self.label_folder, self.label_files[idx])
        with open(label_path, "r") as f:
            label = float(f.read().strip())  # Read distance as float

        return image, torch.tensor([label], dtype=torch.float32)

def get_dataloader(image_folder, label_folder, batch_size):
    """
    Create a DataLoader for the dataset.

    Parameters:
    - image_folder: Path to the folder containing images.
    - label_folder: Path to the folder containing labels.
    - batch_size: Batch size for the DataLoader.
    - input_shape: Tuple (height, width) for resizing images.

    Returns:
    - DataLoader object for training.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor without resizing
    ])

    dataset = ImageDataset(image_folder, label_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def get_input_shape(image_folder):
    """
    Dynamically determine the input shape from the first image in the dataset.

    Parameters:
    - image_folder: Path to the folder containing images.

    Returns:
    - Tuple representing the input shape (channels, height, width).
    """
    # Get the first image in the folder
    image_files = sorted(os.listdir(image_folder))
    if not image_files:
        raise ValueError(f"No images found in folder: {image_folder}")

    # Load the first image
    img_path = os.path.join(image_folder, image_files[0])
    with Image.open(img_path) as img:
        width, height = img.size  # Image dimensions
        channels = len(img.getbands())  # Number of color channels (e.g., RGB = 3)

    return (channels, height, width)



if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    IMAGE_FOLDER = "road_images/trail1/images"
    LABEL_FOLDER = "road_images/trail1/labels"
    batch_size = 32

    input_shape = get_input_shape(image_folder=IMAGE_FOLDER)
    print(f"Determined input shape: {input_shape}")


    dataloader = get_dataloader(IMAGE_FOLDER, LABEL_FOLDER, batch_size)

    num_images = len(dataloader.dataset)
    print(f"Number of images in the dataset: {num_images}")


    model = LaneDetectionCNN(input_shape).to(device)

    criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, dataloader, criterion, optimizer, n_epochs=5)

    




    ## notes: 
    ## to display an image: 

    # for Xbatch, ybatch in dataloader:
    #     sample = Xbatch[0]
    #     image_array = (sample.permute(1, 2, 0).numpy() * 255).astype("uint8")
    #     print(type(image_array))  
    #     image = Image.fromarray(image_array)
    #     image.show()
    #     break;