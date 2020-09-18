import pandas as pd
import cv2
import numpy as np
#AVX2 Erorr
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
def nothing(x):
    pass

#Add Dataset Path
dataset_path = 'fer2013/fer2013/fer2013.csv'
screen_size=(48, 48)

#dataset train
def load_fer2013():
        data = pd.read_csv(dataset_path)
        pixels = data['pixels'].tolist()
        width, height = 48, 48
        faces = []
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(width, height)
            face = cv2.resize(face.astype('uint8'), screen_size)
            faces.append(face.astype('float32'))
        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)
        emotions = pd.get_dummies(data['emotion']).as_matrix()
        return faces, emotions

def preprocess_input(value, v2=True):
    value = value.astype('float32')
    value = value / 255.0
    if v2:
        value = value - 0.5
        value = value * 2.0
    return value