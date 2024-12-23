from PIL import Image
import argparse
import sys
# import wx
from tkinter import Tk, Canvas
import gym
import numpy as np
import pyglet
from pyglet.window import key
from tkinter import Tk, Canvas
import cv2
import time
import os
from PIL import Image, ImageTk
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




class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0
        self.pid_enabled = False
        self.pid_action = np.array([0.3, 0.0]) # initial pid action

    def compute(self, error, dt):
        # Proportional term
        proportional = self.kp * error

        # Integral term
        self.integral += error * dt
        integral = self.ki * self.integral

        # Derivative term
        derivative = self.kd * (error - self.previous_error) / dt
        self.previous_error = error

        # PID output
        return proportional + integral + derivative
    
    def set_action(self, dist, dt):
        error = -dist 
        pid_output = self.compute(error, dt)
        self.pid_action[1] = -pid_output # np.clip(pid_output, -1.0, 1.0)

    

class SaveImages:
    def __init__(self, data_set_name=None):
        if data_set_name!=None:
            self.IMAGE_FOLDER = f"training_images/{data_set_name}/images"
            self.DISTANCE_FOLDER = f"training_images/{data_set_name}/labels"
            self.ACTION_FOLDER = f"training_images/{data_set_name}/actions"
            os.makedirs(self.IMAGE_FOLDER, exist_ok=True)
            os.makedirs(self.DISTANCE_FOLDER, exist_ok=True)
            os.makedirs(self.ACTION_FOLDER, exist_ok=True)
            self.initialized = True
        else:
            self.initialized = False
        self.image_id = 0
        self.capturing_images = False

    
    def toggle_capturing(self):
        if not self.initialized:
            print("Dataset not initialized for storying images. please pass a name during initialization.")
            time.sleep(1)
            return
        self.capturing_images = not self.capturing_images
        if self.capturing_images:
            print("Started capturing images and distances.")
        else:
            print("Stopped capturing images and distances.")
        time.sleep(1)

    def save_image_enabled(self):
        return self.capturing_images


    def save_image_distance_action(self, obs, distance, action_taken):
        if not self.initialized:
            print("Dataset not initialized for storying images. please pass a name during initialization.")
            exit(-1)
        # Save the image
        im = Image.fromarray(obs)
        image_path = os.path.join(self.IMAGE_FOLDER, f"{self.image_id}.png")
        im.save(image_path)

        # Save the distance in a text file
        distance_path = os.path.join(self.DISTANCE_FOLDER, f"{self.image_id}.txt")
        with open(distance_path, "w") as f:
            f.write(str(distance))

        # Save the action in a separate text file
        action_path = os.path.join(self.ACTION_FOLDER, f"{self.image_id}.txt")
        with open(action_path, "w") as f:
            f.write(f"{action_taken[0]} {action_taken[1]}")

        self.image_id +=1


# def apply_preprocessing(image):
#     """
#     Apply preprocessing transformations to the input image.

#     Parameters:
#     - image: PIL Image object.
#     """
#     image_array = np.array(image)
#     channels = [image_array[:, :, i] for i in range(image_array.shape[2])]
#     h, w, _ = image_array.shape
    
#     imghsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
#     img = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

#     mask_ground = np.ones(img.shape, dtype=np.uint8)  # Start with a mask of ones (white)


#     one_third_height = h // 3
#     mask_ground[:one_third_height, :] = 0  # Mask the top 1/3 of the image
    
#     #gaussian filter
#     sigma = 4.5
#     img_gaussian_filter = cv2.GaussianBlur(img,(0,0), sigma)
    
#     sobelx = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,1,0)
#     sobely = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,0,1)
#     # Compute the magnitude of the gradients
#     Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)
#     threshold = 50


#     white_lower_hsv = np.array([0, 0, 153])         # CHANGE ME
#     white_upper_hsv = np.array([228, 69, 255])   # CHANGE ME
#     yellow_lower_hsv = np.array([15, 30, 100])        # CHANGE ME
#     yellow_upper_hsv = np.array([35, 254, 255])  # CHANGE ME

#     mask_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)
#     mask_yellow = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv)


#     mask_mag = (Gmag > threshold)

#     # np.savetxt("mask.txt", mask_white, fmt='%d', delimiter=',')
#     # exit()

#     final_mask = mask_ground * mask_mag * 255 
#     mask_white = mask_ground * mask_white
#     mask_yellow = mask_ground * mask_yellow
#     # Convert the NumPy array back to a PIL image

#     channels[0] =  final_mask
#     channels[1] =  mask_white
#     channels[2] =  mask_yellow
    
#     filtered_image = np.stack(channels, axis=-1)
#     # filtered_image = Image.fromarray(filtered_image)
#     return  filtered_image