import time
import cv2
import numpy as np
from utils import *

if __name__ == "__main__":
    vidcap = cv2.VideoCapture("progetto/AZMOUN e LUKAKU ribaltano tutto nel recupero Roma-Lecce 2-1 Serie A TIM DAZN Highlights.mp4")
    success, image = vidcap.read()
    count = 0
    success = True
    idx = 0

    # Read the video frame by frame
    while success:
        start_time =  time.time()
        campo = trovaCampoDaGioco(image=image)
        linee = find_lines(campo,image)
        # show_image(campo)
        # show_image(linee)
        show_image(image)
        # findPlayers(campo, image)
        end_time = time.time()
        print(f"Execution time of findPlayers: {end_time - start_time} seconds")
        success, image = vidcap.read()
