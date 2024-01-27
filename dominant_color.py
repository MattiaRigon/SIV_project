import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import show_image

QS = 3
QV = 3
QLN = 72

def quantization_level(h,s,v):

    H,S,V = 0,0,0
    
    if h < 10 or h > 170:
        H = 0
    elif h < 20 and h > 10:
        H = 1
    elif h < 33 and h > 20:
        H = 2
    elif h < 78 and h > 33:
        H = 3
    elif h < 95 and h > 78:
        H = 4
    elif h < 135 and h > 95:
        H = 5
    elif h < 150 and h > 135:
        H = 6
    else:
        H = 7

    if s < 51:
        S = 0
    elif s < 179 and s > 51:
        S = 1
    else:
        S = 2

    if v < 51:
        V = 0
    elif v < 179 and v > 51:
        V = 1
    else:
        V = 2

    return [H,S,V]

def quantified_value(h,s,v):
    global QS,QV
    return (h*QS*QV)+(QS*s)+v

def dominant_colors_extraction(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    rows, cols, _ = image.shape
    quantified_values = np.zeros((rows, cols), dtype=int)

    for i in range(rows):
        for j in range(cols):
            h, s, v = hsv_image[i, j]
            quantization_hsv = quantization_level(h, s, v)
            quantified_values[i, j] = quantified_value(quantization_hsv[0], quantization_hsv[1], quantization_hsv[2])

    show_image(hsv_image)
    show_image(quantified_values)
    # histogram = np.zeros(quantified_values, dtype=float)

    # for i in range(rows):
    #     for j in range(cols):
    #         histogram[quantified_values[i, j]] += 1

    # histogram /= np.sum(histogram)  # Normalizzazione dell'istogramma

    # return histogram

# Carica l'immagine
image = cv2.imread('campo.png')  # Inserisci il percorso della tua immagine

dominant_colors_extraction(image)

