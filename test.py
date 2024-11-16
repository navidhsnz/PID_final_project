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
from PIL import Image, ImageTk

from gym_duckietown.envs import DuckietownEnv

# from experiments.utils import save_img

parser = argparse.ArgumentParser()
parser.add_argument("--env-name", default="Duckietown-zigzag_dists-v0")
parser.add_argument("--map-name", default="zigzag_dists")
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

def write_pid_to_file():
    """
    Writes the current PID parameters to a file.
    Overwrites the file each time the parameters change.
    """
    with open("pid_parameters.txt", "w") as f:
        f.write(f"kp: {pid.kp}\n")
        f.write(f"ki: {pid.ki}\n")
        f.write(f"kd: {pid.kd}\n")



@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print("RESET")
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

    # Adjust PID parameters
    elif symbol == key.O:  # Increase kp
        pid.kp += 10
        write_pid_to_file()
    elif symbol == key.P:  # Decrease kp
        pid.kp = max(0, pid.kp - 10)
        write_pid_to_file()
    elif symbol == key.K:  # Increase kd
        pid.kd += 10
        write_pid_to_file()
    elif symbol == key.L:  # Decrease kd
        pid.kd = max(0, pid.kd - 10)
        write_pid_to_file()

    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0






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
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    global old_dist
    wheel_distance = 0.102
    min_rad = 0.08

    action = np.array([0.0, 0.0])
    

    if key_handler[key.UP]:
        action += np.array([1, 0.0])
    if key_handler[key.DOWN]:
        action -= np.array([1, 0])
    if key_handler[key.LEFT]:
        action += np.array([0, 2])
    if key_handler[key.RIGHT]:
        action -= np.array([0, 2])
    if key_handler[key.SPACE]:
        action = last_action #np.array([0, 0])


    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    # obs, reward, done, info = env.step(action)
    # print(f"step_count = {env.unwrapped.step_count}, reward={reward}")
    
    obs, dist = env.go(action)

    print(old_dist)
    if dist == None:
        dist = old_dist
    else:
        # dist *= 100
        old_dist = dist

    if key_handler[key.SPACE]:  
        error = -dist
        pid_output = pid.compute(error, dt)
        turn = pid_output
        last_action[1] = -pid_output # np.clip(pid_output, -1.0, 1.0)
    else:
        turn = None

    print(f"dist = {dist}, dt = {dt}, pid_output {turn}")



    if key_handler[key.RETURN]:

        im = Image.fromarray(obs)

        im.save("example.png")

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

# pid
pid = PIDController(kp=50, ki=0, kd=50)

# last action
last_action = np.array([0.1, 0.0])

##
old_dist = 0

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
