#!/usr/bin/env python
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
from learning.RNN_distance.RNN import *
from gym_duckietown.envs import DuckietownEnv
from stuff import *


env = DuckietownEnv(
        seed=1,
        map_name="small_loop",
        draw_curve=False,
        draw_bbox=False,
        domain_rand=False,
        frame_skip=1,
        distortion=False,
        camera_rand=False,
        dynamics_rand=False)

env.reset()
env.render()

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    global capturing_images
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
        save_image.toggle_capturing()
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
        pid.pid_enabled = not pid.pid_enabled
        print("pid engaged!" if pid.pid_enabled else "pid disengaged!")
    # increase/decrease speed on pid mode
    elif symbol == key.N:
        pid.pid_action[0] +=0.05
    elif symbol == key.B:
        pid.pid_action[0] -=0.05


def update(dt):
    global model, device
    global old_dist, hidden_state

    action = np.array([0.0, 0.0])

    if key_handler[key.UP]:
        action += np.array([0.5, 0.0])
    if key_handler[key.DOWN]:
        action -= np.array([1, 0])
    if key_handler[key.LEFT]:
        action += np.array([0, 2])
    if key_handler[key.RIGHT]:
        action -= np.array([0, 2])
    if pid.pid_enabled:
        action = pid.pid_action

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    obs, dist = env.go(action)
    if dist == None:
        dist = old_dist
    else:
        old_dist = dist

    im = np.array(Image.fromarray(obs))

    # Predict distance and update hidden state
    # previous action is given to the model
    predicted_distance, hidden_state = predict_dist(model, im, action, hidden_state, device) 
    
    
    if save_image.save_image_enabled():
        if env.unwrapped.step_count % 27 == 0:
            save_image.save_image_distance_action(obs, dist, action)

    if pid.pid_enabled:  
        pid.get_action(predicted_distance,dt) #predicted_distance  #  dist # 

    if key_handler[key.Z]:
        im = Image.fromarray(obs)
        im.save("example.png")
        print("image saved !")

    # print(f"dist_real = {dist}, pred_real = {predicted_distance}, dt = {dt}")
    # print(f"pred_real = {predicted_distance}")

    env.render()

# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# pid
pid = PIDController(kp=50, ki=0, kd=50)

# initialize save image class
save_image = SaveImages()

# initialize the action of pid control

old_dist = 0 # will contain previous distance value for times when agent exits the lane
hidden_state = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# # Load the trained model
input_shape = (3, 480, 640)  # Input shape of images (channels, height, width)
model = load_RNN_model("learning/RNN_distance/models/lane_detection_rnn5.pth", input_shape, device)

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()