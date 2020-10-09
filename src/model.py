import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from random import shuffle
from time import sleep

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

MOVES = {'Run_Left': [1, 0, 0, 0, 0, 0],
        'Run_Right': [0, 1, 0, 0, 0, 0],
        'Jump_Left': [0, 0, 1, 0, 0, 0],
        'Jump_Right': [0, 0, 0, 1, 0, 0],
        'Jump': [0, 0, 0, 0, 1, 0],
        'Stay': [0, 0, 0, 0, 0, 1]}

IMG_SIZE = 300

def get_size_stat(DIR):
    heights = []
    widths = []
    for img in os.listdir(DIR):
        path = os.path.join(DIR, img)
        data = np.array(Image.open(path))
        heights.append(data.shape[0])
        widths.append(data.shape[1])
    avg_height = sum(heights) / len(heights)
    avg_width = sum(widths) / len(widths)
    
    print('Ave H: ' + str(avg_height))
    print('Max H: ' + str(max(heights)))
    print('Min H: ' + str(min(heights)))
    print('\n')
    print('Ave W: ' + str(avg_width))
    
def label_img(name):
    word_label = name.split('-')[0]
    if word_label in MOVES.keys():
        return MOVES[word_label]
    else:
        return [0 for _ in range(len(MOVES))]

def load_data(DIR):
    data = []
    for img in os.listdir(DIR):
        label = label_img(img)
        path = os.path.join(DIR, img)
        img = Image.open(path)
        img = img.convert('L')
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
        data.append([np.array(img), label])
    shuffle(data)
    return data
        

def build_model(train_data):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(96, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(96, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(len(MOVES), activation='softmax'))
    
    trainImages = np.array([data[0] for data in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    trainLabels = np.array([data[1] for data in train_data])
    
    model.compile(loss='binary_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])
    
    model.fit(trainImages, trainLabels, batch_size=20, epochs=5, verbose=1)
    
    return model

def load_model(model_path='Yun_model'):
    model = keras.models.load_model(model_path)
    return model

def predict(model, frame):
    img = Image.fromarray(frame)
    img = img.convert('L')
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
    data = np.array(img).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    result = model.predict(data)
    key = get_move(result)
    return key

def get_move(result):
    res = [0 for _ in range(len(MOVES))]
    ind = np.argmax(result) 
    res[ind] = 1 
    for key, value in MOVES.items():
        if res == value:
            return key
    
