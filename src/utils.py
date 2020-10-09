"""
This provides the tools for
1. record the images for model training
2. rename the images for model training 
3. simulate mario run

@Author: Yun Wu
@Email: yunwustc@gmail.com
"""
import numpy as np
from nes_py.wrappers import JoypadSpace
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from time import sleep
import cv2
import os


def record_video():
    """
    record the video for model training
    """
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 420)
    
    i = 0
    while i < 1000:
        
        key = cv2.waitKey(10)
        if key == 27: # ESC key value = 27
            break
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        cv2.imshow('Detection', frame)
        cv2.imwrite('./img/'+str(i)+'.jpg', frame)
        i += 1

    cap.release()
    cv2.destroyAllWindows()

def rename_files(path, start_index, end_index, name, move, output_folder='Train'):
    files = os.listdir(path) 

    output_path = os.path.join(path, output_folder)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    for i in range(start_index, end_index + 1):
        file = os.path.join(path, str(i) + '.jpg')
        output_file = os.path.join(output_path, move + '-' + name + str(i) + '.jpg')
        os.rename(file, output_file)

def mario(v, lock):
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    done = True
    while True:
        if done:
            env.reset()
            with lock:
                v.value = 0
        with lock:
            u = v.value
        _, _, done, _ = env.step(u)
        env.render()
        sleep(0.02)

