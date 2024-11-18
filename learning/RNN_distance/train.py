import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import random


import torch
import torch.nn as nn

class LaneDetectionCNN(nn.Module):
    def __init__(self, input_shape=(3, 224, 224)):
        super(LaneDetectionCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

        # Calculate flat size dynamically for compatibility with RNNs
        self._to_linear = None
        self._calculate_flat_size(input_shape)

    def _calculate_flat_size(self, input_shape):
        """Pass a dummy tensor through the convolutional layers to determine the flattened size."""
        x = torch.zeros(1, *input_shape)
        x = self._forward_conv(x)
        self._to_linear = x.numel()

    def _forward_conv(self, x):
        """Forward pass through convolutional layers."""
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.maxpool1(x)
        x = torch.relu(self.conv3(x))
        x = self.maxpool2(x)
        x = torch.relu(self.conv4(x))
        x = self.maxpool3(x)
        x = torch.relu(self.conv5(x))
        x = self.maxpool4(x)
        x = self.dropout(x)
        return x

    def forward(self, x):
        """
        Forward pass through the CNN.
        Returns flattened features suitable for RNN input.
        """
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)  # Flatten for RNN input
        return x



class LaneDetectionRNN(nn.Module):
    def __init__(self, input_shape, rnn_hidden_size=128, rnn_num_layers=3, dropout=0.2):
        super(LaneDetectionRNN, self).__init__()
        # CNN for feature extraction
        self.cnn = LaneDetectionCNN(input_shape=input_shape)
        
        # RNN for sequential processing
        self.rnn = nn.LSTM(
            input_size=self.cnn._to_linear,  # Use CNN output size
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
            dropout=dropout if rnn_num_layers > 1 else 0.0
        )
        
        # Fully connected layer for final prediction
        self.fc = nn.Linear(rnn_hidden_size, 1)

    def forward(self, x):
        """
        Forward pass for a batch of image sequences.
        x: Tensor of shape (batch_size, seq_length, channels, height, width)
        """
        batch_size, seq_length, channels, height, width = x.size()

        # Flatten sequence dimension and process each image through CNN
        x = x.view(batch_size * seq_length, channels, height, width)
        cnn_features = self.cnn(x)  # Shape: (batch_size * seq_length, cnn_output_size)
        
        # Reshape back to sequence format
        cnn_features = cnn_features.view(batch_size, seq_length, -1)
        
        # Process through RNN
        rnn_out, _ = self.rnn(cnn_features)
        
        # Predict distance for each timestep
        predictions = torch.tanh(self.fc(rnn_out)).squeeze(-1)  # Shape: (batch_size, seq_length)
        return predictions


class LaneDetectionRNN(nn.Module):
    def __init__(self, input_shape, rnn_hidden_size=128):
        super(LaneDetectionRNN, self).__init__()
        self.cnn = LaneDetectionCNN(input_shape)
        self.rnn = nn.LSTM(
            input_size=self.cnn._to_linear,  # Feature size from CNN
            hidden_size=rnn_hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Linear(rnn_hidden_size, 1)  # Predict distance for each timestep

    def forward(self, x):
        batch_size, seq_length, channels, height, width = x.size()
        x = x.view(batch_size * seq_length, channels, height, width)  # Flatten sequence dimension
        cnn_features = self.cnn(x)  # Extract features
        cnn_features = cnn_features.view(batch_size, seq_length, -1)  # Restore sequence dimension
        rnn_out, _ = self.rnn(cnn_features)  # Process with RNN
        predictions = torch.tanh(self.fc(rnn_out))  # Predict for each timestep
        return predictions


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random
from torch.utils.data import DataLoader, random_split


class SequentialImageDataset(Dataset):
    """
    Dataset for loading sequences of images and corresponding labels.
    Divides the dataset into fixed-length sequences, shuffles them, and provides
    each sequence for training or evaluation.
    """
    def __init__(self, image_folder, label_folder, seq_length=100, transform=None):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.seq_length = seq_length
        self.transform = transform

        # Load and sort image and label files
        self.image_files = sorted(os.listdir(image_folder))
        self.label_files = sorted(os.listdir(label_folder))
        # Ensure dataset size is divisible by seq_length
        self.num_sequences = len(self.image_files) // seq_length

        self.image_files = self.image_files[:self.num_sequences * seq_length]
        self.label_files = self.label_files[:self.num_sequences * seq_length]

        # Split into sequences
        self.sequences = [
            (self.image_files[i:i + seq_length], self.label_files[i:i + seq_length])
            for i in range(0, len(self.image_files), seq_length)
        ]


        # Shuffle sequences
        random.shuffle(self.sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        image_sequence = []
        label_sequence = []

        # Load images and labels for the sequence
        image_files, label_files = self.sequences[idx]
        for img_file, lbl_file in zip(image_files, label_files):
            # Load and transform image
            img_path = os.path.join(self.image_folder, img_file)
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            image_sequence.append(image)

            # Load label
            lbl_path = os.path.join(self.label_folder, lbl_file)
            with open(lbl_path, "r") as f:
                label = float(f.read().strip())
            label_sequence.append(label)

        # Stack images and labels into tensors
        image_sequence = torch.stack(image_sequence)  # Shape: (seq_length, channels, height, width)
        label_sequence = torch.tensor(label_sequence, dtype=torch.float32)  # Shape: (seq_length,)
        return image_sequence, label_sequence

def get_sequential_dataloader(
    image_folder, label_folder, batch_size, seq_length=100, train_fraction=0.8, val_fraction=0.1, test_fraction=0.1
):
    """
    Create DataLoaders for training, validation, and testing datasets, ensuring distinct splits.

    Parameters:
    - image_folder: Path to the folder containing images.
    - label_folder: Path to the folder containing labels.
    - batch_size: Number of sequences per batch.
    - seq_length: Number of images per sequence.
    - train_fraction: Fraction of the data to use for training.
    - val_fraction: Fraction of the data to use for validation.
    - test_fraction: Fraction of the data to use for testing.

    Returns:
    - train_loader: DataLoader for training.
    - val_loader: DataLoader for validation.
    - test_loader: DataLoader for testing.
    """
    if not (0.0 < train_fraction + val_fraction + test_fraction <= 1.0):
        raise ValueError("Fractions for train, validation, and test must sum to 1.0 or less.")

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to tensor
    ])

    # Load the full dataset
    dataset = SequentialImageDataset(image_folder, label_folder, seq_length=seq_length, transform=transform)

    # Compute sizes for train, validation, and test splits
    total_size = len(dataset)
    train_size = int(total_size * train_fraction)
    val_size = int(total_size * val_fraction)
    test_size = total_size - train_size - val_size

    # Perform the splits
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    IMAGE_FOLDER = "training_images/trail2/images"
    LABEL_FOLDER = "training_images/trail2/labels"
    batch_size = 4  # Number of sequences per batch
    seq_length = 100  # Number of images per sequence

    # Define split fractions
    train_fraction = 0.8
    val_fraction = 0.1
    test_fraction = 0.1

    # Create DataLoaders
    train_loader, val_loader, test_loader = get_sequential_dataloader(
        IMAGE_FOLDER, LABEL_FOLDER, batch_size, seq_length,
        train_fraction=train_fraction, val_fraction=val_fraction, test_fraction=test_fraction
    )

    # Test the training DataLoader
    print("Testing Train DataLoader:")
    for images, labels in train_loader:
        print(f"Image batch shape: {images.shape}")  # Expected: (batch_size, seq_length, channels, height, width)
        print(f"Label batch shape: {labels.shape}")  # Expected: (batch_size, seq_length)
        break

    # Test the validation DataLoader
    print("\nTesting Validation DataLoader:")
    for images, labels in val_loader:
        print(f"Image batch shape: {images.shape}")  # Expected: (batch_size, seq_length, channels, height, width)
        print(f"Label batch shape: {labels.shape}")  # Expected: (batch_size, seq_length)
        break

    # Test the test DataLoader
    print("\nTesting Test DataLoader:")
    for images, labels in test_loader:
        print(f"Image batch shape: {images.shape}")  # Expected: (batch_size, seq_length, channels, height, width)
        print(f"Label batch shape: {labels.shape}")  # Expected: (batch_size, seq_length)
        break
print("number of data: ",list(map(len, [train_loader, val_loader, test_loader])))



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Assuming the following classes and functions are already defined:
# - LaneDetectionRNN
# - SequentialImageDataset

# -------------------
# Validation Function
# -------------------
def validate_model(model, dataloader, criterion, device):
    """
    Validate the LaneDetectionRNN model on the validation set.
    
    Parameters:
    - model: The RNN model to validate.
    - dataloader: DataLoader providing validation data.
    - criterion: Loss function (e.g., MSELoss).
    - device: Device to run the model on ('cpu' or 'cuda').
    
    Returns:
    - Average validation loss.
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            predictions = model(images)  # Shape: (batch_size, seq_length)
            predictions = predictions.squeeze(-1)

            # Compute loss
            loss = criterion(predictions, labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss




# -------------------
# Training Function with Validation and Loss Tracking
# -------------------
def train_model_with_validation(
    model, train_loader, val_loader, criterion, optimizer, device, n_epochs=10
):
    """
    Train the LaneDetectionRNN model with validation after each epoch and track loss.
    
    Parameters:
    - model: The RNN model to train.
    - train_loader: DataLoader providing training data.
    - val_loader: DataLoader providing validation data.
    - criterion: Loss function (e.g., MSELoss).
    - optimizer: Optimizer (e.g., Adam).
    - device: Device to run the model on ('cpu' or 'cuda').
    - n_epochs: Number of epochs to train for.

    Returns:
    - train_losses: List of average training losses for each epoch.
    - val_losses: List of average validation losses for each epoch.
    """
    model.to(device)

    # Lists to store loss values
    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):
        # Training
        model.train()
        total_loss = 0.0
        for batch_idx, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader),
                                                desc=f"Epoch {epoch + 1}/{n_epochs}", unit="batch"):

            images, labels = images.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            predictions = model(images)
            predictions = predictions.squeeze(-1)

            # Compute loss
            loss = criterion(predictions, labels)
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        avg_val_loss = validate_model(model, val_loader, criterion, device)
        val_losses.append(avg_val_loss)

        print(
            f"Epoch [{epoch + 1}/{n_epochs}] - "
            f"Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}"
        )
    
    return train_losses, val_losses




# -------------------
# TRAIN
# -------------------
if __name__ == "__main__":
    # Dataset paths
    IMAGE_FOLDER = "training_images/trail2/images"
    LABEL_FOLDER = "training_images/trail2/labels"

    batch_size = 12  # Number of sequences per batch
    seq_length = 20  # Number of images per sequence
    n_epochs = 10
    learning_rate = 0.001

    # Define split fractions
    train_fraction = 0.85  # 80% for training
    val_fraction = 0.1    # 10% for validation
    test_fraction = 0.05   # 10% for testing

    # Create DataLoaders using the improved get_sequential_dataloader function
    train_loader, val_loader, test_loader = get_sequential_dataloader(
        IMAGE_FOLDER, LABEL_FOLDER, batch_size=batch_size, seq_length=seq_length,
        train_fraction=train_fraction, val_fraction=val_fraction, test_fraction=test_fraction
    )

    # Initialize the model
    input_shape = (3, 480, 640)  # Update based on actual image dimensions
    model = LaneDetectionRNN(input_shape=input_shape, rnn_hidden_size=128)

    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Mean squared error loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Train the model with validation
    train_losses, val_losses = train_model_with_validation(
        model, train_loader, val_loader, criterion, optimizer, device, n_epochs
    )

    # Save the trained model
    torch.save(model.state_dict(), "lane_detection_rnn.pth")
    print("Model saved to 'lane_detection_rnn.pth'")

    # Evaluate on test set
    test_loss = validate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
