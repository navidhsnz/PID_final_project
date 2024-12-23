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
from gym_duckietown.envs import DuckietownEnv
from stuff import *
from torchvision.transforms.functional import crop
import argparse
import time

# note: before running the code, make sure you import the right version. 
# For example, for CNN1:
import learning.CNN.CNN1 as CNN1

# For example, for CNN2:
import learning.CNN.CNN2 as CNN2

# For example, for RNN1:
import learning.RNN.RNN1 as RNN1

# For example, for RNN2:
import learning.RNN.RNN2 as RNN2

# For example, for RNN3:
import learning.RNN.RNN3 as RNN3

# For example, for RNN4:
import learning.RNN.RNN4 as RNN4


parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="CNN2")

parser.add_argument("--show_preprocessed", default=True)
parser.add_argument("--averaging", default=True)
args = parser.parse_args()



if args.model_type == "CNN1":
    model_weights_path = "learning/CNN/models/lane_detection_CNN1.pth"
elif args.model_type == "CNN2":
    model_weights_path = "learning/CNN/models/lane_detection_CNN2.pth"
elif args.model_type == "RNN1":
    model_weights_path = "learning/RNN/models/lane_detection_rnn_new_RNN1.pth"
elif args.model_type == "RNN2":
    model_weights_path = "learning/RNN/models/lane_detection_rnn_new_RNN2.pth"
elif args.model_type == "RNN3":
    model_weights_path = "learning/RNN/models/lane_detection_rnn_new_RNN3.pth"
elif args.model_type == "RNN4":
    model_weights_path = "learning/RNN/models/lane_detection_rnn_new_RNN4.pth"

print(f"Model type: {args.model_type}")
print(f"Model weights: {model_weights_path}")




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
    global model, device, old_predicted_distance
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
        action = pid.pid_action.copy()

    # Speed boost
    if key_handler[key.LSHIFT]:
        action[0] *= 2


    img, dist = env.go(action)

    if dist == None:
        dist = old_dist
    else:
        old_dist = dist
    
    # use model to predict distance from the middle of the lane
    if args.model_type=="CNN1":  
        predicted_distance = CNN1.predict_dist(model, img, device)
    elif args.model_type=="CNN2":  
        predicted_distance = CNN2.predict_dist(model, img, device) 
    elif args.model_type=="RNN1":  
        predicted_distance, hidden_state = RNN1.predict_dist(model, img, action, hidden_state, device)
    elif args.model_type=="RNN2":  
        predicted_distance, hidden_state = RNN2.predict_dist(model, img, action, hidden_state, device)
    elif args.model_type=="RNN3":  
        predicted_distance, hidden_state = RNN3.predict_dist(model, img, action, hidden_state, device)
    elif args.model_type=="RNN4":  
        predicted_distance, hidden_state = RNN4.predict_dist(model, img, action, hidden_state, device)
    
    # show pre_procecessed image
    if args.model_type=="CNN1" and args.show_preprocessed:
        CNN1.show_image(img,f"True Distance:      {dist}",f"Predicted Distance: {predicted_distance}",args.model_type)
    elif args.model_type=="CNN2" and args.show_preprocessed:
        CNN2.show_image(img,f"True Distance:      {dist}",f"Predicted Distance: {predicted_distance}",args.model_type)
    elif args.model_type=="RNN1" and args.show_preprocessed:
        RNN1.show_image(img,f"True Distance:      {dist}",f"Predicted Distance: {predicted_distance}",args.model_type)
    elif args.model_type=="RNN2" and args.show_preprocessed:
        RNN2.show_image(img,f"True Distance:      {dist}",f"Predicted Distance: {predicted_distance}",args.model_type)
    elif args.model_type=="RNN3" and args.show_preprocessed:
        RNN3.show_image(img,f"True Distance:      {dist}",f"Predicted Distance: {predicted_distance}",args.model_type)
    elif args.model_type=="RNN4" and args.show_preprocessed:
        RNN4.show_image(img,f"True Distance:      {dist}",f"Predicted Distance: {predicted_distance}","")


    ratio = 0.9
    if args.averaging:
        predicted_distance = predicted_distance*ratio + old_predicted_distance*(1-ratio)
    
    old_predicted_distance = predicted_distance
    
    if save_image.save_image_enabled():
        if env.unwrapped.step_count % 8 == 0:
            save_image.save_image_distance_action(img, dist, action)

    if pid.pid_enabled:  
        pid.set_action(predicted_distance,dt) #predicted_distance  #  dist # 

    if key_handler[key.Z]:
        im = Image.fromarray(img)
        im.save("example.png")
        print("image saved !")

    print(f"dist_real = {dist}, dt = {dt}")
    print(f"pred_real = {predicted_distance}")

    env.render()


# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# pid
pid = PIDController(kp=50, ki=0, kd=50)

# initialize save image class
save_image = SaveImages("benchmark")

# initialize the action of pid control

old_dist = 0 # will contain previous distance value for times when agent exits the lane
old_predicted_distance = 0 # if averaging is enabled, this is the pred dist taken in step before. averaging is to smooth out the changes in dist.
hidden_state = None


# # Load the trained model
input_shape = (3, 480, 640)  # Input shape of images (channels, height, width)
if args.model_type=="CNN1":
    model = CNN1.load_CNN_model(model_weights_path, input_shape, device)
if args.model_type=="CNN2":
    model = CNN2.load_CNN_model(model_weights_path, input_shape, device)
elif args.model_type=="RNN1":
    model = RNN1.load_RNN_model(model_weights_path, input_shape, device)
elif args.model_type=="RNN2":
    model = RNN2.load_RNN_model(model_weights_path, input_shape, device)
elif args.model_type=="RNN3":
    model = RNN3.load_RNN_model(model_weights_path, input_shape, device)
elif args.model_type=="RNN4":
    model = RNN4.load_RNN_model(model_weights_path, input_shape, device)

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()