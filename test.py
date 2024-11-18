#!/usr/bin/env python
#

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""
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

from gym_duckietown.envs import DuckietownEnv

# from experiments.utils import save_img

# Global counter for image and text file IDs
pid_enabled = False
step_counter = 0
image_id = 0
capturing_images = False
IMAGE_FOLDER = "training_images/trail4/images"
DISTANCE_FOLDER = "training_images/trail4/labels"
ACTION_FOLDER = "training_images/trail4/actions"
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(DISTANCE_FOLDER, exist_ok=True)
os.makedirs(ACTION_FOLDER, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--env-name", default="Duckietown-small_loop-v0")
parser.add_argument("--map-name", default="small_loop")
parser.add_argument("--distortion", default=False, action="store_true")
parser.add_argument("--camera_rand", default=False, action="store_true")
parser.add_argument("--draw-curve", action="store_true", help="draw the lane following curve")
parser.add_argument("--draw-bbox", action="store_true", help="draw collision detection bounding boxes")
parser.add_argument("--domain-rand", action="store_true", help="enable domain randomization")
parser.add_argument("--dynamics_rand", action="store_true", help="enable dynamics randomization")
parser.add_argument("--frame-skip", default=1, type=int, help="number of frames to skip")
parser.add_argument("--seed", default=1, type=int, help="seed")
args = parser.parse_args()

if args.env_name and args.env_name.find("Duckietown") != -1:
    env = DuckietownEnv(
        seed=args.seed,
        map_name=args.map_name,
        draw_curve=args.draw_curve,
        draw_bbox=args.draw_bbox,
        domain_rand=args.domain_rand,
        frame_skip=args.frame_skip,
        distortion=args.distortion,
        camera_rand=args.camera_rand,
        dynamics_rand=args.dynamics_rand,
    )
else:
    env = gym.make("Duckietown-small_loop-v0") 

env.reset()
env.render()

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
        
    return model



def toggle_capturing():
    global capturing_images
    capturing_images = not capturing_images
    if capturing_images:
        print("Started capturing images and distances.")
    else:
        print("Stopped capturing images and distances.")
    time.sleep(1)



def save_image_distance_action(obs, distance, action_taken, image_id):
    # Save the image
    im = Image.fromarray(obs)
    image_path = os.path.join(IMAGE_FOLDER, f"{image_id}.png")
    im.save(image_path)

    # Save the distance in a text file
    distance_path = os.path.join(DISTANCE_FOLDER, f"{image_id}.txt")
    with open(distance_path, "w") as f:
        f.write(str(distance))

    # Save the action in a separate text file
    action_path = os.path.join(ACTION_FOLDER, f"{image_id}.txt")
    with open(action_path, "w") as f:
        f.write(f"{action_taken[0]} {action_taken[1]}")



@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """
    global pid_enabled
    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print("RESET")
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)
    elif symbol == key.S:
        toggle_capturing()

    # Adjust PID parameters
    elif symbol == key.O:  # Increase kp
        pid.kp += 10
        # write_pid_to_file()
    elif symbol == key.P:  # Decrease kp
        pid.kp = max(0, pid.kp - 10)
        # write_pid_to_file()
    elif symbol == key.K:  # Increase kd
        pid.kd += 10
        # write_pid_to_file()
    elif symbol == key.L:  # Decrease kd
        pid.kd = max(0, pid.kd - 10)
        # write_pid_to_file()

    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0

    elif key_handler[key.SPACE]:
        pid_enabled = not pid_enabled
        print("pid engaged!" if pid_enabled else "pid disabled.")

    # increase/decrease speed on pid mode
    elif symbol == key.N:  # Decrease kd
        last_action[0] +=0.05
    elif symbol == key.B:  # Decrease kd
        last_action[0] -=0.05







# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0

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



def update(dt):
    global pid_enabled
    global old_dist
    global model, device
    wheel_distance = 0.102
    min_rad = 0.08
    global old_dist, step_counter, image_id, capturing_images

    action = np.array([0.0, 0.0])

    if key_handler[key.UP]:
        action += np.array([0.5, 0.0])
    if key_handler[key.DOWN]:
        action -= np.array([1, 0])
    if key_handler[key.LEFT]:
        action += np.array([0, 2])
    if key_handler[key.RIGHT]:
        action -= np.array([0, 2])
    
    
    if pid_enabled:
        action = last_action #np.array([0, 0])


    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    # obs, reward, done, info = env.step(action)
    # print(f"step_count = {env.unwrapped.step_count}, reward={reward}")
    
    obs, dist = env.go(action)

    if dist == None:
        dist = old_dist
    else:
        # dist *= 100
        old_dist = dist

    im = np.array(Image.fromarray(obs))

    image_tensor = torch.from_numpy(np.transpose(im, (2, 0, 1))).float() / 255.0

    image_tensor = image_tensor.unsqueeze(0)
 
    with torch.no_grad():
        prediction = model(image_tensor)  # Predict the distance

    predicted_distance = prediction.item()
    
    

    #  step: {env.unwrapped.step_count},
    if capturing_images:
        if env.unwrapped.step_count % 27 == 0:
            save_image_distance_action(obs, dist, action, image_id)
            image_id += 1


    if pid_enabled:  
        error = -dist  #  predicted_distance # 
        pid_output = pid.compute(error, dt)
        turn = pid_output
        last_action[1] = -pid_output # np.clip(pid_output, -1.0, 1.0)
    else:
        turn = None


    if key_handler[key.Z]:

        im = Image.fromarray(obs)

        im.save("example.png")
        print("image saved !")


    print(f"dist_real = {dist}, dt = {dt}, pid_output {turn}")
    print(f"pred_real = {predicted_distance}")
    # if done:
    #     print("done!")
    #     env.reset()
    #     env.render()

    env.render()

# Create a Tkinter window
# root = Tk()
# root.title("Live PNG Animation")
# canvas = Canvas(root, width=900, height=900)  # Adjust the size to your image dimensions
# canvas.pack()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# pid
pid = PIDController(kp=50, ki=0, kd=50)

# last action
last_action = np.array([0.1, 0.0])

##
old_dist = 0
model = LaneDetectionCNN((3, 480, 640))

# Load the model weights
model.load_state_dict(torch.load("lane_detection_model.pth"))
model.eval()  # Set to evaluation mode
print("Model weights loaded from 'lane_detection_model.pth'")

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
