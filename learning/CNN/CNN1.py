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

        self.fc1 = nn.Linear(self._to_linear, 1000) 
        self.fc2 = nn.Linear(1000, 1000) 
        self.fc3 = nn.Linear(1000, 1) 

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
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = x*0.5 #sscale output to be between -0.5 and 0.5
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
        # transforms.Lambda(apply_preprocessing),
        transforms.ToTensor(),
    ])

    return transform



#  #display image after filter application
# # image = Image.open('example.png') 
# ii=0
# while True:
#     image = cv2.imread(f"training_images/trail2/images/image_{ii}.jpg")

#     # print(image.shape)

#     # create an transform for crop top 1/3 of the image
#     transform = get_transform()
    
#     image_new = transform(image) 
    
#     # display(image)
#     # display(image_crop)
#     image_new = np.array(image_new)
#     cv2.imshow("Image", image_new)
#     cv2.imshow("Imageee", image)
#     print(f"image_{ii}")
#     cv2.waitKey(0)
#     ii+=1



# Dataset class
# class SequentialImageDataset(Dataset):
#     def __init__(self, image_folder, label_folder, transform=None):
#         self.image_folder = image_folder
#         self.label_folder = label_folder
#         self.transform = transform

#         self.image_files = sorted(os.listdir(image_folder))
#         self.label_files = sorted(os.listdir(label_folder))

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.image_folder, self.image_files[idx])
#         lbl_path = os.path.join(self.label_folder, self.label_files[idx])

#         image = Image.open(img_path).convert("RGB")
#         if self.transform:
#             image = self.transform(image)

#         with open(lbl_path, "r") as f:
#             label = float(f.read().strip())

#         return image, torch.tensor(label, dtype=torch.float32)



class LaneDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_filenames = sorted(os.listdir(image_dir))
        self.label_filenames = sorted(os.listdir(label_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Load label
        label_path = os.path.join(self.label_dir, self.label_filenames[idx])
        with open(label_path, 'r') as file:
            label = float(file.readline().strip())

        return image, torch.tensor(label, dtype=torch.float32)
    


def loader_creator(image_dir, label_dir, transform=None, batch_size=32, train_split=0.7, val_split=0.2):
    
    full_dataset = LaneDataset(image_dir, label_dir, transform=transform)

    train_size = int(train_split * len(full_dataset))
    val_size = int(val_split * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# # the follwoin code check the images and labels are properly laoded and can be displayed
# if __name__ == "__main__":
#     image_folder = "training_images/trail2/images"
#     lablel_folder = "training_images/trail2/labels"
#     batch_size = 10

#     train_fraction = 0.85
#     val_fraction = 0.1
#     test_fraction = 0.05

#     transform = get_transform()
#     train_loader, val_loader, test_loader = loader_creator(
#         image_folder, lablel_folder, transform, batch_size, train_fraction, val_fraction)

#     print("testing train dataLoader:")
#     for images, labels in train_loader:
#         print(f"Image batch shape: {images.shape}")  
#         print(f"Label batch shape: {labels.shape}")  
#         break

#     print("\n\ntesting validation dataLoader:")
#     for images, labels in val_loader:
#         print(f"Image batch shape: {images.shape}") 
#         print(f"Label batch shape: {labels.shape}")  
#         break

#     print("\ntesting test dataLoader:")
#     for images, labels in test_loader:
#         print(f"Image batch shape: {images.shape}") 
#         print(f"Label batch shape: {labels.shape}") 
#         break
#     print("number of data: ",list(map(len, [train_loader, val_loader, test_loader])))


#     import matplotlib.pyplot as plt

#     for images, labels in train_loader:
#             print(f"Image batch shape: {images.shape}") 
#             print(f"Label batch shape: {labels.shape}") 

#             iimg = images[0]  
#             llbl = labels[0]  

#             image = iimg.permute(1, 2, 0).numpy() 
#             plt.imshow(image)
#             plt.title(f"Label: {llbl.item()}")
#             plt.axis('off')
#             plt.show()

#             break 



def validation(model, val_loader, criterion, device):
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            running_val_loss += loss.item()

    epoch_val_loss = running_val_loss / len(val_loader)

    return epoch_val_loss



# training
def train(model, dataloader, val_loader, criterion, optimizer, epochs, snapshot_epochs):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            

        epoch_loss = running_loss / len(dataloader)
        train_losses.append(epoch_loss)


        epoch_val_loss = validation(model, val_loader, criterion, device)
        val_losses .append(epoch_val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {epoch_loss:.8f}, Validation Loss: {epoch_val_loss:.8f}")

        if epoch in snapshot_epochs:
            checkpoint_path = f"lane_detection_rnn_new_CNN1_{epoch}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved to '{checkpoint_path}'")

    return train_losses, val_losses



# run this part for trianing the model. 
if __name__ == "__main__":

    transform = get_transform()

    train_loader, val_loader, test_loader = loader_creator(
        image_dir="training_images/trail2/images",
        label_dir="training_images/trail2/labels",
        transform=transform,
        batch_size=16,
        train_split=0.8,
        val_split=0.2
        ) 

    epochs_num=20

    input_shape = (3, 480, 640)
    model = LaneDetectionCNN(input_shape=input_shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device is {device}")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # train model
    train_losses, val_losses = train(model, train_loader, val_loader, criterion, optimizer, epochs_num, snapshot_epochs=[5,10,19])

    # save the model
    torch.save(model.state_dict(), "lane_detection_CNN1.pth")

    with open("train_losses_CNN1.txt", "w") as train_file:
        for loss in train_losses:
            train_file.write(f"{loss}\n")

    with open("val_losses_CNN1.txt", "w") as val_file:
        for loss in val_losses:
            val_file.write(f"{loss}\n")

    print("Train and validation losses saved to 'train_losses_CNN1.txt' and 'val_losses_CNN1.txt'")



##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################

# the following functions are to be used when using a trained model for real time predictions in the simulation. 
# They are called in run_simulation.py:

def predict_dist(model, im, device):
    transform = get_transform()
    im = transform(im)
    im = im.unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(im)

    predicted_distance = prediction.item()

    return predicted_distance


def load_CNN_model(model_path, input_shape, device, rnn_hidden_size=128):
    model = LaneDetectionCNN(input_shape)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"CNN model loaded successfully from '{model_path}'")
    return model

# def show_image(img):
#     transform = get_transform()
#     img = transform(img)
#     img = img.permute(1, 2, 0).cpu().numpy()
#     img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     cv2.imshow("Image", img_bgr)
#     # cv2.imshow("Imageee", im_pp)
#     # im = np.array(Image.fromarray(obs))
#     cv2.waitKey(1)

import cv2
import numpy as np

def show_image(img, line1, line2, line3):
    transform = get_transform()
    img = transform(img)
    img = img.permute(1, 2, 0).cpu().numpy()
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.59
    color = (0, 0, 0)
    thickness = 2

    img_bgr = cv2.putText(img_bgr, line1, (10, 60), font, font_scale, color, thickness, cv2.LINE_AA)
    img_bgr = cv2.putText(img_bgr, line2, (10, 90), font, font_scale, color, thickness, cv2.LINE_AA)
    img_bgr = cv2.putText(img_bgr, line3, (10, 30), font, font_scale, color, thickness, cv2.LINE_AA)

    cv2.imshow("Image input of the model", img_bgr)
    cv2.waitKey(1)