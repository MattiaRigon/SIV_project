import time
import cv2
import numpy as np
from utils import *

if __name__ == "__main__":
    vidcap = cv2.VideoCapture("2h-left-5min.avi")
    success, image = vidcap.read()
    count = 0
    success = True
    idx = 0

    # Read the video frame by frame
    while success:
        nuova_dimensione = (2048, 1080)
        image = cv2.resize(image, nuova_dimensione)
        pts = np.array([[257, 540], [1018, 263], [1736, 246], [1738, 643]], np.int32)
        pts = pts.reshape((-1, 1, 2))

        # Crea la maschera
        mask = np.zeros((1080, 2048), dtype=np.uint8)  # Creazione di un'immagine vuota per la maschera
        cv2.fillPoly(mask, [pts], 255)  # Disegna il poligono sulla maschera

        # Applica la maschera all'immagine originale
        image = cv2.bitwise_and(image, image, mask=mask)
        start_time = time.time()
        campo = trovaCampoDaGioco(image=image)
        # linee = find_lines(campo, image)
        # show_image(campo)
        # # show_image(linee)
        # show_image(campo)
        findPlayers(campo, image)
        
        end_time = time.time()
        fps = 1 / (end_time-start_time)

        print(f"{fps:.2f} FPS")        
        success, image = vidcap.read()
