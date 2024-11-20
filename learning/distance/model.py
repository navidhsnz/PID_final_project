import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
import os
import torch.optim as optim
from torchvision import transforms,models
from torchvision.transforms.functional import crop
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import cv2
from torchsummary import summary

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

        # Calculate flat size dynamically
        self._to_linear = None
        self._calculate_flat_size(input_shape)

        self.fc1 = nn.Linear(self._to_linear, 7*7*3) # 7x7 image with 3 channels
        self.fc2 = nn.Linear(7*7*3, 1)  # Single output neuron for regression

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
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = x*0.5 #Scale output to be between -0.5 and 0.5
        return x

# summary(LaneDetectionCNN().to(device), (3, 224, 224))


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

# Prepare dataset, split into train, validation, and test sets
def prepare_dataset(image_folder, label_folder):
    # Preprocessing : Define transformations to apply to each image
    transform = transforms.Compose([
        #Crop only the top 2/3 of the image, removing the top 1/3
        transforms.Lambda(apply_preprocessing),
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.GaussianBlur(3, sigma=(0.1, 2.0)),  # Apply Gaussian blur
        transforms.ToTensor(),  # Convert to tensor without resizing
    ])

    dataset = ImageDataset(image_folder, label_folder, transform=transform)

    train_dataset, val_test_set = random_split(dataset, [3000,1000]) # Lets start with 75-15-10 split
    val_dataset, test_dataset = random_split(val_test_set, [600,400])
    
    return train_dataset, val_dataset, test_dataset

def get_dataloaders(image_folder, label_folder, batch_size):
    """
    Create DataLoaders for the dataset.

    Parameters:
    - image_folder: Path to the folder containing images.
    - label_folder: Path to the folder containing labels.
    - batch_size: Batch size for the DataLoader.
    - input_shape: Tuple (height, width) for resizing images.

    Returns:
    - DataLoaders object for training, val and test.
    """
    train,val,test = prepare_dataset(image_folder, label_folder)
    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=True)

    return train_dataloader,val_dataloader,test_dataloader


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

    return (channels, 224, 224) #temporary for tests

def apply_preprocessing(image):
    """
    Apply preprocessing transformations to the input image.

    Parameters:
    - image: PIL Image object.
    """
    
    image_array = np.array(image)
    channels = [image_array[:, :, i] for i in range(image_array.shape[2])]
    h, w, _ = image_array.shape
    
    imghsv = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)
    
    img = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    mask_ground = np.ones(img.shape, dtype=np.uint8)  # Start with a mask of ones (white)
    mid_point = img.shape[0] // 2  # Integer division to get the middle row index

    # Set the top half of the image to 0 (black)
    mask_ground[:mid_point-30, :] = 0  # Mask the top half (rows 0 to mid_point-1)
    
    #gaussian filter
    sigma = 3.5
    img_gaussian_filter = cv2.GaussianBlur(img,(0,0), sigma)
    
    sobelx = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,0,1)
    # Compute the magnitude of the gradients
    Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)
    threshold = 35
    mask_mag = (Gmag > threshold)
        #4 Mask yellow and white

    white_lower_hsv = np.array([0,(0*255)/100,(60*255)/100]) # [0,0,50] - [230,100,255]
    white_upper_hsv = np.array([150,(40*255)/100,(100*255)/100])   # CHANGE ME

    yellow_lower_hsv = np.array([(30*179)/360, (30*255)/100, (30*255)/100])        # CHANGE ME
    yellow_upper_hsv = np.array([(90*179)/360, (110*255)/100, (100*255)/100])  # CHANGE ME
    mask_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)
    mask_yellow = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv)

    mask_sobelx_pos = (sobelx > 0)
    mask_sobelx_neg = (sobelx < 0)
    mask_sobely_pos = (sobely > 0)
    mask_sobely_neg = (sobely < 0)

    final_mask = mask_ground * mask_mag  * (mask_white + mask_yellow) #* (mask_sobelx_neg * mask_sobely_neg + mask_sobelx_pos* mask_sobely_pos)
    # Convert the NumPy array back to a PIL image
    for channel in channels:
        channel *= final_mask
    filtered_image = np.stack(channels, axis=-1)
    filtered_image = Image.fromarray(filtered_image)
    return filtered_image

def apply_preprocessing2(image):
    """
    Apply preprocessing transformations to the input image.

    Parameters:
    - image: PIL Image object.
    """
    
    # Crop the top 1/3 of the image
    height = image.size[1]
    crop_height = height // 3
    
    image = crop(image, crop_height,0, image.size[1], image.size[0])
    # Convert the NumPy array back to a PIL image
    return image

def temp(image):
    image_array = np.array(image)
    # Crop the top 1/3 of the image
    height = image.size[1]
    crop_height = height // 3
    #adding black mask on top 1/3
    cropped_image_array = image_array
    cropped_image_array[:crop_height, :, :] = 0  # Set the top 1/3 to black
    
    #cropped_image_array = image_array[crop_height:, :, :]  # Crop the top 1/3
    
    # Convert the NumPy array back to a PIL image
    cropped_image_pil = Image.fromarray(cropped_image_array)
    return cropped_image_pil