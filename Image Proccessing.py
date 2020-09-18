# İmage-Proccessing and Emotion Analysis
import pandas as pd
import cv2
import numpy as np
#AVX2 Error
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
def nothing(x):
    pass

# Update Dataset
dataset_path = 'fer2013/fer2013/fer2013.csv'
# Set Screen Size
ekran_ebatı=(48, 48)

def load_fer2013():

        data = pd.read_csv(dataset_path)
        pixels = data['pixels'].tolist()
	#Hight and Width Screen
        width, height = 48, 48
        faces = []
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(width, height)
            face = cv2.resize(face.astype('uint8'), ekran_ebatı)
            faces.append(face.astype('float32'))
        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)
        duygular = pd.get_dummies(data['emotion']).as_matrix()
        return faces, duygular

def preprocess_input(value, v2=True):