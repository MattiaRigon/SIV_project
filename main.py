import cv2
import numpy as np
from utils import *

if __name__ == "__main__":
    # image_path = 'progetto/cagliariphoto.jpeg'  # Sostituisci con il percorso del qtuo file immagine
    # image = cv2.imread(image_path)
    # campo = trovaCampoDaGioco(image=image)
    vidcap = cv2.VideoCapture('PlayerDetection/cutvideo.mp4')
    success,image = vidcap.read()
    count = 0
    success = True
    idx = 0

    #Read the video frame by frame
    while success:
        campo = trovaCampoDaGioco(image=image)
        findPlayers(campo,image)
        success,image = vidcap.read()
