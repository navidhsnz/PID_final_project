import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import random
import numpy as np
import cv2
from tqdm import tqdm

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


        # self.fc1 = nn.Linear(self._to_linear, 200) 

    def _calculate_flat_size(self, input_shape):
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
        x = x.view(x.size(0), -1)  
        # x = torch.relu(self.fc1(x))
        return x

def apply_preprocessing(image):
    image_array = np.array(image)

    blurred_image_array = cv2.GaussianBlur(image_array, (0, 0), 0.1)
    channels = [image_array[:, :, i] for i in range(image_array.shape[2])]
    h, w, _ = image_array.shape
    
    imghsv = cv2.cvtColor(blurred_image_array, cv2.COLOR_RGB2HSV)
    img = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    mask_ground = np.ones(img.shape, dtype=np.uint8)  # Start with a mask of ones (white)


    one_third_height = h // 3
    # crop_height = h * 2 // 5 
    mask_ground[:one_third_height, :] = 0  # Mask the top 1/3 of the image
    
    #gaussian filter
    sigma = 4.5
    img_gaussian_filter = cv2.GaussianBlur(img,(0,0), sigma)
    
    sobelx = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,0,1)
    # Compute the magnitude of the gradients
    Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)
    threshold = 51


    white_lower_hsv = np.array([0, 0, 143])         # CHANGE ME
    white_upper_hsv = np.array([228, 60, 255])   # CHANGE ME
    yellow_lower_hsv = np.array([10, 50, 100])        # CHANGE ME
    yellow_upper_hsv = np.array([70, 255, 255])  # CHANGE ME

    mask_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)
    mask_yellow = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv)

    # crop two fifth of the image on the right for the yello mask
    height, width = mask_yellow.shape 
    crop_width = width * 2 // 5 
    crop_width_2 = width * 1 // 2 
    crop_mask_11 = np.zeros_like(mask_yellow, dtype=np.uint8)
    crop_mask_11[:, :width - crop_width] = 1 
    mask_yellow = mask_yellow * crop_mask_11

    # crop two fifth of the image on the left for the white mask
    crop_mask_22 = np.zeros_like(mask_white, dtype=np.uint8)
    crop_mask_22[:, crop_width:] = 1 
    mask_white = mask_white * crop_mask_22


    mask_mag = (Gmag > threshold)

    # np.savetxt("mask.txt", mask_white, fmt='%d', delimiter=',')
    # exit()
    crop_width_3 = width * 1 // 10 
    crop_mask_33 = np.zeros_like(mask_yellow, dtype=np.uint8)
    crop_mask_33[:, :width - crop_width_3] = 1 

    crop_mask_44 = np.zeros_like(mask_white, dtype=np.uint8)
    crop_mask_44[:, crop_width_3:] = 1 

    final_mask = mask_ground * mask_mag * 255 
    mask_white = mask_ground * mask_white
    mask_yellow = mask_ground * mask_yellow
    # Convert the NumPy array back to a PIL image

    channels[0] =  final_mask #np.zeros_like(channels[0])
    channels[1] =  mask_white
    channels[2] =  mask_yellow
    
    filtered_image = np.stack(channels, axis=-1)
    filtered_image = Image.fromarray(filtered_image)
    return  filtered_image

def get_transform():
    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.Lambda(apply_preprocessing),
        transforms.ToTensor(),
    ])

    return transform

class LaneDetectionRNN(nn.Module):
    def __init__(self, input_shape, rnn_hidden_size=128):
        super(LaneDetectionRNN, self).__init__()
        self.cnn = LaneDetectionCNN(input_shape)
        self.rnn = nn.LSTM(
            input_size=self.cnn._to_linear, #200,#self.cnn._to_linear, #self.cnn._to_linear,  
            hidden_size=rnn_hidden_size,
            num_layers=3,
            batch_first=True
        )
        self.fc = nn.Linear(rnn_hidden_size, 1) 

    def forward(self, x, hidden_state=None):
        batch_size, seq_length, channels, height, width = x.size()
        x = x.view(batch_size * seq_length, channels, height, width)  
        cnn_features = self.cnn(x) 
        cnn_features = cnn_features.view(batch_size, seq_length, -1) 
        rnn_out, hidden_state = self.rnn(cnn_features, hidden_state) 
        predictions = 0.5 * torch.tanh(self.fc(rnn_out))  
        return predictions, hidden_state


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random
from torch.utils.data import DataLoader, random_split


class SequentialImageDataset(Dataset):
    
    def __init__(self, image_folder, label_folder, seq_length=100, transform=None):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.seq_length = seq_length
        self.transform = transform

        self.image_files = sorted(os.listdir(image_folder))
        self.label_files = sorted(os.listdir(label_folder))
        self.num_sequences = len(self.image_files) // seq_length

        self.image_files = self.image_files[:self.num_sequences * seq_length]
        self.label_files = self.label_files[:self.num_sequences * seq_length]

        self.sequences = [
            (self.image_files[i:i + seq_length], self.label_files[i:i + seq_length])
            for i in range(0, len(self.image_files), seq_length)
        ]


        random.shuffle(self.sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        image_sequence = []
        label_sequence = []

        image_files, label_files = self.sequences[idx]
        for img_file, lbl_file in zip(image_files, label_files):
            img_path = os.path.join(self.image_folder, img_file)
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            image_sequence.append(image)

            lbl_path = os.path.join(self.label_folder, lbl_file)
            with open(lbl_path, "r") as f:
                label = float(f.read().strip())
            label_sequence.append(label)

        image_sequence = torch.stack(image_sequence)
        label_sequence = torch.tensor(label_sequence, dtype=torch.float32) 
        return image_sequence, label_sequence

def get_sequential_dataloader(
    image_folder, label_folder, batch_size, seq_length=100, train_fraction=0.8, val_fraction=0.1, test_fraction=0.1
):

    if not (0.0 < train_fraction + val_fraction + test_fraction <= 1.0):
        raise ValueError("fractions must sum to 1.0 or less")

    transform = get_transform()

    dataset = SequentialImageDataset(image_folder, label_folder, seq_length=seq_length, transform=transform)

    total_size = len(dataset)
    train_size = int(total_size * train_fraction)
    val_size = int(total_size * val_fraction)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    if len(train_dataset)!=0:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    else: train_loader= None
    if len(val_dataset)!=0:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else: val_loader= None
    if len(test_dataset)!=0:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else: test_loader= None

    return train_loader, val_loader, test_loader


# testing the dataloaders and image preprocesing outputs
# if __name__ == "__main__":
#     image_folder = "../dataset/images"
#     label_folder = "../dataset/labels"
#     batch_size = 10  
#     seq_length = 20 

#     train_fraction = 0.85
#     val_fraction = 0.1
#     test_fraction = 0.05

#     train_loader, val_loader, test_loader = get_sequential_dataloader(
#         image_folder, label_folder, batch_size, seq_length,
#         train_fraction=train_fraction, val_fraction=val_fraction, test_fraction=test_fraction
#     )

#     # print("Testing Train DataLoader:")
#     # for images, labels in train_loader:
#     #     print(f"Image batch shape: {images.shape}")  
#     #     print(f"Label batch shape: {labels.shape}") 
#     #     break

#     # print("\nTesting Validation DataLoader:")
#     # for images, labels in val_loader:
#     #     print(f"Image batch shape: {images.shape}") 
#     #     print(f"Label batch shape: {labels.shape}") 
#     #     break

#     # print("\nTesting Test DataLoader:")
#     # for images, labels in test_loader:
#     #     print(f"Image batch shape: {images.shape}") 
#     #     print(f"Label batch shape: {labels.shape}") 
#     #     break
#     # print("number of data: ",list(map(len, [train_loader, val_loader, test_loader])))

# import matplotlib.pyplot as plt

# for images, labels in train_loader:
#         print(f"Image batch shape: {images.shape}") 
#         print(f"Label batch shape: {labels.shape}") 

#         sequence = images[0] 
#         label_sequence = labels[0] 

#         print("Displaying images from the first sequence...")
#         for i in range(len(sequence)):
#             image = sequence[i].permute(1, 2, 0).numpy() 
#             plt.imshow(image)
#             plt.title(f"Label: {label_sequence[i].item()}")
#             plt.axis('off')
#             plt.show()

#         break 


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# validation
def validate_model(model, dataloader, criterion, device):
    model.eval() 
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            predictions, _  = model(images) 
            predictions = predictions.squeeze(-1)

            loss = criterion(predictions, labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss




# training
def train_model_with_validation(model, train_loader, val_loader, criterion, optimizer, device, n_epochs=10):

    model.to(device)
    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for batch_idx, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader),
                                                desc=f"Epoch {epoch + 1}/{n_epochs}", unit="batch"):

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            predictions, _ = model(images)
            predictions = predictions.squeeze(-1)

            loss = criterion(predictions, labels)
            total_loss += loss.item()

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


def train_model_with_validation_and_checkpoints(model, train_loader, val_loader, criterion, optimizer, device, n_epochs, save_epochs):
    model.to(device)
    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for images_, labels_ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}"):
            images, labels = images_.to(device), labels_.to(device)

            optimizer.zero_grad()
            predictions, _ = model(images)
            predictions = predictions.squeeze(-1)

            loss = criterion(predictions, labels)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        avg_val_loss = validate_model(model, val_loader, criterion, device)
        val_losses.append(avg_val_loss)

        print(
            f"Epoch [{epoch + 1}/{n_epochs}] - "
            f"Train Loss: {avg_train_loss:.7f}, Validation Loss: {avg_val_loss:.7f}")

        if epoch in save_epochs:
            checkpoint_path = f"lane_detection_rnn_new_RNN2_{epoch}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved to '{checkpoint_path}'")
    
    return train_losses, val_losses


# train
if __name__ == "__main__":
    image_folder = "../dataset/images"
    label_folder = "../dataset/labels"

    batch_size = 10  
    seq_length = 20 
    n_epochs = 20
    learning_rate = 0.001

    train_fraction = 0.85  
    val_fraction = 0.1    
    test_fraction = 0.05  

    train_loader, val_loader, test_loader = get_sequential_dataloader(
        image_folder, label_folder, batch_size=batch_size, seq_length=seq_length,
        train_fraction=train_fraction, val_fraction=val_fraction, test_fraction=test_fraction
    )

    input_shape = (3, 480, 640) 
    model = LaneDetectionRNN(input_shape=input_shape, rnn_hidden_size=128)

    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_losses, val_losses = train_model_with_validation_and_checkpoints(
        model, train_loader, val_loader, criterion, optimizer, device, n_epochs, [10,15,20,25,30])

    torch.save(model.state_dict(), "lane_detection_rnn_new_RNN2.pth")
    print("model saved to 'lane_detection_rnn_new_RNN2.pth'")

    test_loss = validate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.7f}")



    with open("train_losses_RNN2.txt", "w") as train_file:
        for loss in train_losses:
            train_file.write(f"{loss}\n")

    with open("val_losses_RNN2.txt", "w") as val_file:
        for loss in val_losses:
            val_file.write(f"{loss}\n")

    print("train and losses saved in 'train_losses_RNN2.txt' and 'val_losses_RNN2.txt'")



##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################

# the following functions are to be used when using a trained model for real time predictions in the simulation. 
# They are called in run_simulation.py:


def load_RNN_model(model_path, input_shape, device, rnn_hidden_size=128):
    model = LaneDetectionRNN(input_shape=input_shape, rnn_hidden_size=rnn_hidden_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"RNN model loaded successfully from '{model_path}'")
    return model



import cv2
import numpy as np

def show_image(img, line1, line2, line3):
    transform = get_transform()
    img = transform(img)
    img = img.permute(1, 2, 0).cpu().numpy()
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.59
    color = (255, 255, 255)
    thickness = 2

    img_bgr = cv2.putText(img_bgr, line1, (10, 60), font, font_scale, color, thickness, cv2.LINE_AA)
    img_bgr = cv2.putText(img_bgr, line2, (10, 90), font, font_scale, color, thickness, cv2.LINE_AA)
    img_bgr = cv2.putText(img_bgr, line3, (10, 30), font, font_scale, color, thickness, cv2.LINE_AA)

    cv2.imshow("Image input of the model", img_bgr)
    cv2.waitKey(1)

def predict_dist(model, im, action, hidden_state, device):
    transform = get_transform()
    im = transform(im)
    image_tensor = im.unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction, hidden_state = model(image_tensor, hidden_state)

    predicted_distance = prediction.item()

    return predicted_distance,hidden_state