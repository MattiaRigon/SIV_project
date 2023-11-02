import time
import cv2
import numpy as np
from utils import *
from trovaCampo import find_lines

if __name__ == "__main__":
    # image_path = 'progetto/cagliariphoto.jpeg'  # Sostituisci con il percorso del qtuo file immagine
    # image = cv2.imread(image_path)
    # campo = trovaCampoDaGioco(image=image)
    vidcap = cv2.VideoCapture("PlayerDetection/cutvideo.mp4")
    success, image = vidcap.read()
    count = 0
    success = True
    idx = 0

    # Read the video frame by frame
    while success:
        start_time = time.time()

        campo = trovaCampoDaGioco(image=image)
        linee = find_lines(campo)
        # show_image(campo)
        # show_image(linee)


        findPlayers(campo, image)
        end_time = time.time()
        print(f"Execution time of findPlayers: {end_time - start_time} seconds")
        success, image = vidcap.read()
