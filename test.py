#!/usr/bin/env python
# manual

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
parser.add_argument("--env-name", default="Duckietown-straight_road-v0")
parser.add_argument("--map-name", default="straight_road")
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
    env = gym.make("Duckietown-straight_road-v0") 

env.reset()
env.render()


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

    # Take a screenshot
    # UNCOMMENT IF NEEDED - Skimage dependency
    # elif symbol == key.RETURN:
    #     print('saving screenshot')
    #     img = env.render('rgb_array')
    #     save_img('screenshot.png', img)


# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)



def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
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
        action = np.array([0, 0])


    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    # obs, reward, done, info = env.step(action)
    # print(f"step_count = {env.unwrapped.step_count}, reward={reward}")
    
    obs, dist = env.go(action)
    print(f"dist = {dist}")


    # im = Image.fromarray(obs)

    # image = np.array(im)
    # analyze_image(im)
    # #################################################
 
    # img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # cv2.imshow("Image", img)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # sobelx = cv2.Sobel(img,cv2.CV_64F,1,0)
    # sobely = cv2.Sobel(img,cv2.CV_64F,0,1)

    # sigma = 4.4

    # # Smooth the image using a Gaussian kernel
    # img_gaussian_filter = cv2.GaussianBlur(img,(0,0), sigma)

    # sobelx = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,1,0)
    # sobely = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,0,1)

    # # Compute the magnitude of the gradients
    # Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)

    # # Compute the orientation of the gradients
    # Gdir = cv2.phase(np.array(sobelx, np.float32), np.array(sobely, dtype=np.float32), angleInDegrees=True)

    # threshold = 48

    # mask_mag = (Gmag > threshold)

    # image_masekd = mask_mag*Gmag

    # image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    # white_lower_hsv = np.array([0, 0, 205])         # CHANGE ME
    # white_upper_hsv = np.array([228, 69, 255])   # CHANGE ME
    # yellow_lower_hsv = np.array([15, 30, 100])        # CHANGE ME
    # yellow_upper_hsv = np.array([35, 254, 255])  # CHANGE ME

    # mask_white = cv2.inRange(image_hsv, white_lower_hsv, white_upper_hsv)
    # mask_yellow = cv2.inRange(image_hsv, yellow_lower_hsv, yellow_upper_hsv)

    # width = img.shape[1]
    # mask_left = np.ones(sobelx.shape)
    # mask_left[:,int(np.floor(width/2)):width + 1] = 0
    # mask_right = np.ones(sobelx.shape)
    # mask_right[:,0:int(np.floor(width/2))] = 0

    # mask_sobelx_pos = (sobelx > 0)
    # mask_sobelx_neg = (sobelx < 0)
    # mask_sobely_pos = (sobely > 0)
    # mask_sobely_neg = (sobely < 0)

    # mask_left_edge =  mask_left * mask_mag * mask_sobelx_neg * mask_sobely_neg * mask_yellow
    # mask_right_edge =  mask_right * mask_mag * mask_sobelx_pos * mask_sobely_neg * mask_white


    # # Convert the masks to uint8 (values between 0 and 255) for display
    # mask_left_edge = np.uint8(mask_left_edge * 255)
    # mask_right_edge = np.uint8(mask_right_edge * 255)

    # # Convert NumPy arrays to PIL Image format
    # # pil_mask_left_edge = Image.fromarray(mask_left_edge)
    # # pil_mask_right_edge = Image.fromarray(mask_right_edge)

    #     #################################################


    # tk_img = ImageTk.PhotoImage(pil_mask_left_edge)
    
    # canvas.create_image(0, 0, anchor="nw", image=tk_img)
    # root.update_idletasks()  # This ensures the window updates immediately
    # root.update()  # This triggers the event loop to redraw the canvas


    # # Convert the PIL Image to Tkinter-compatible format
    # tk_img = ImageTk.PhotoImage(im)

    # # Update the canvas with the new image
    # canvas.create_image(0, 0, anchor="nw", image=tk_img)
    # root.update_idletasks()  # This ensures the window updates immediately
    # root.update()  # This triggers the event loop to redraw the canvas


    if key_handler[key.RETURN]:

        im = Image.fromarray(obs)

        im.save("example.png")

    # if done:
    #     print("done!")
    #     env.reset()
    #     env.render()

    env.render()

# Create a Tkinter window
root = Tk()
root.title("Live PNG Animation")
canvas = Canvas(root, width=900, height=900)  # Adjust the size to your image dimensions
canvas.pack()



pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
