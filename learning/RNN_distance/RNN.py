import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
import random
from tqdm import tqdm
import numpy as np

class LaneDetectionCNN(nn.Module):
    def __init__(self, input_shape=(3, 224, 224)):
        super(LaneDetectionCNN, self).__init__()

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

        self._to_linear = None
        self._calculate_flat_size(input_shape)

    def _calculate_flat_size(self, input_shape):
        """Pass a dummy tensor through the convolutional layers to determine the flattened size."""
        x = torch.zeros(1, *input_shape)
        x = self._forward_conv(x)
        self._to_linear = x.numel()

    def _forward_conv(self, x):
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
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)  # Flatten for RNN input
        return x

class LaneDetectionRNN(nn.Module):
    def __init__(self, input_shape, rnn_hidden_size=128, num_frequencies=6):
        super(LaneDetectionRNN, self).__init__()
        self.cnn = LaneDetectionCNN(input_shape)
        self.rnn = nn.LSTM(
            input_size=self.cnn._to_linear + 2 * 2 * num_frequencies, 
            hidden_size=rnn_hidden_size,
            num_layers=5,
            batch_first=True
        )
        self.fc = nn.Linear(rnn_hidden_size, 1)

    def forward(self, x, actions, hidden_state=None):
        batch_size, seq_length, channels, height, width = x.size()
        x = x.view(batch_size * seq_length, channels, height, width)
        cnn_features = self.cnn(x)
        cnn_features = cnn_features.view(batch_size, seq_length, -1)

        # Shift the actions so for each image, the action that goes with it as input is that of the time step of the previous image.
        shifted_actions = torch.zeros_like(actions) 
        shifted_actions[:, 1:] = actions[:, :-1]  # Shift actions by one timestep
        shifted_actions[:, 0] = 0.0

        rnn_input = torch.cat((cnn_features, shifted_actions), dim=-1)  # (batch_size, seq_length, feature_size)

        # Process with RNN
        rnn_out, hidden_state = self.rnn(rnn_input, hidden_state)
        predictions = torch.tanh(self.fc(rnn_out))
        return predictions, hidden_state


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random
from torch.utils.data import DataLoader, random_split


class SequentialImageDataset(Dataset):
    def __init__(self, image_folder, label_folder, action_folder, seq_length=100, transform=None, num_frequencies=6):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.action_folder = action_folder
        self.seq_length = seq_length
        self.num_frequencies = num_frequencies
        self.transform = transform

        # Load and sort
        self.image_files = sorted(os.listdir(image_folder))
        self.label_files = sorted(os.listdir(label_folder))
        self.action_files = sorted(os.listdir(action_folder))

        # mkae sure  dataset size is divisible by seq_length
        self.num_sequences = len(self.image_files) // seq_length
        self.image_files = self.image_files[:self.num_sequences * seq_length]
        self.label_files = self.label_files[:self.num_sequences * seq_length]
        self.action_files = self.action_files[:self.num_sequences * seq_length]

        self.sequences = [
            (self.image_files[i:i + seq_length],
             self.label_files[i:i + seq_length],
             self.action_files[i:i + seq_length])
            for i in range(0, len(self.image_files), seq_length)
        ]
        random.shuffle(self.sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        image_sequence = []
        label_sequence = []
        action_sequence = []

        image_files, label_files, action_files = self.sequences[idx]
        for img_file, lbl_file, act_file in zip(image_files, label_files, action_files):
            # load and transform image
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

            # Load actions
            act_path = os.path.join(self.action_folder, act_file)
            with open(act_path, "r") as f:
                speed, angular_velocity = map(float, f.read().strip().split())

                # Compute Fourier features
                speed_features = compute_fourier_features(torch.tensor(speed), self.num_frequencies)
                angular_features = compute_fourier_features(torch.tensor(angular_velocity), self.num_frequencies)

                action_sequence.append(torch.cat([speed_features, angular_features]))


        image_sequence = torch.stack(image_sequence) 
        label_sequence = torch.tensor(label_sequence, dtype=torch.float32)
        action_sequence = torch.stack(action_sequence)
        return image_sequence, label_sequence, action_sequence

def compute_fourier_features(value, num_frequencies=6):
    frequencies = 2 ** torch.arange(num_frequencies, dtype=torch.float32)
    features = torch.cat([
        torch.sin(frequencies * value),
        torch.cos(frequencies * value)
    ])
    return features

def get_sequential_dataloader(
    image_folder, label_folder, action_folder, batch_size, seq_length=100, train_fraction=0.8, val_fraction=0.1, test_fraction=0.1
):
    if not (0.0 < train_fraction + val_fraction + test_fraction <= 1.0):
        raise ValueError("Fractions for train, validation, and test must sum to 1.0.")

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to tensor
    ])

    # load the dataset
    dataset = SequentialImageDataset(image_folder, label_folder, action_folder, seq_length=seq_length, transform=transform)

    # Compute sizes for train validation and test
    total_size = len(dataset)
    train_size = int(total_size * train_fraction)
    val_size = int(total_size * val_fraction)
    test_size = total_size - train_size - val_size

    # split
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader # return the DataLoaders


# This function evaluates the model
def validate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, labels, actions in dataloader:
            images, labels, actions = images.to(device), labels.to(device), actions.to(device)

            predictions,_ = model(images, actions)
            predictions = predictions.squeeze(-1)

            # Compute loss
            loss = criterion(predictions, labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

# training function
def train_model(
    model, train_loader, val_loader, criterion, optimizer, device, n_epochs=10
):
    model.to(device)

    # Lists to store loss values
    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for images, labels, actions in tqdm(train_loader, total=len(train_loader),
                                                desc=f"Epoch {epoch + 1}/{n_epochs}", unit="batch"):

            images, labels, actions = images.to(device), labels.to(device), actions.to(device)

            optimizer.zero_grad()
            predictions, _ = model(images, actions)
            predictions = predictions.squeeze(-1)

            loss = criterion(predictions, labels)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()



        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # validation
        avg_val_loss = validate_model(model, val_loader, criterion, device)
        val_losses.append(avg_val_loss)

        print(
            f"Epoch [{epoch + 1}/{n_epochs}] - "
            f"Train Loss: {avg_train_loss:.7f}, Validation Loss: {avg_val_loss:.7f}"
        )

    return train_losses, val_losses

if __name__ == "__main__":
    # Dataset paths
    IMAGE_FOLDER = "training_images/trail2/images"
    LABEL_FOLDER = "training_images/trail2/labels"
    ACTION_FOLDER = "training_images/trail2/actions"

    batch_size = 10  # number of sequences per batch
    seq_length = 20  # number of images per sequence
    n_epochs = 10
    learning_rate = 0.001

    # Define splits
    train_fraction = 0.85 
    val_fraction = 0.1 
    test_fraction = 0.05

    # create DataLoaders
    train_loader, val_loader, test_loader = get_sequential_dataloader(
        IMAGE_FOLDER, LABEL_FOLDER, ACTION_FOLDER, batch_size=batch_size, seq_length=seq_length,
        train_fraction=train_fraction, val_fraction=val_fraction, test_fraction=test_fraction
    )

    # Initialize
    input_shape = (3, 480, 640)
    model = LaneDetectionRNN(input_shape=input_shape, rnn_hidden_size=128)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Train the model with validation
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, device, n_epochs)

    # Save the model for ftutre
    torch.save(model.state_dict(), "models/lane_detection_rnn7.pth")
    print("Model saved to 'lane_detection_rnn7.pth'")

    # Evaluate on test set
    test_loss = validate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.7f}")



def predict_dist(model, im , action, hidden_state, device):
    image_tensor = torch.from_numpy(np.transpose(im, (2, 0, 1))).float() / 255.0

    action_0 = compute_fourier_features(action[0])
    action_1 = compute_fourier_features(action[1])

    action_tensor = torch.cat([action_0, action_1])


    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0).to(device)

    action_tensor = action_tensor.unsqueeze(0).unsqueeze(0).to(device) 

    # Forward pass through the model
    with torch.no_grad():
        prediction, hidden_state = model(image_tensor, action_tensor, hidden_state)


    # Extract the scalar prediction
    predicted_distance = prediction.item()

    return predicted_distance, hidden_state



def load_RNN_model(model_path, input_shape, device, rnn_hidden_size=128):
    model = LaneDetectionRNN(input_shape=input_shape, rnn_hidden_size=rnn_hidden_size)
    # Load state dictionary
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"RNN model loaded successfully from '{model_path}'")
    return model