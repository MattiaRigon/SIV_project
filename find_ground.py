import time
import cv2
import numpy as np

from utils import show_image

selected_points = False
points = []

# Funzione di callback per gestire i clic del mouse
def click_event(event, x, y, flags, param):
    global selected_points,points,image
    if event == cv2.EVENT_LBUTTONDOWN:
        # Aggiungi le coordinate del clic a una lista
        points.append((x, y))
        

        # Disegna un punto rosso sul frame
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        
        # Se abbiamo cliccato su almeno 4 punti, disegna il poligono
        if len(points) == 4:
            print(points)
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))

            new_dimension = (2048, 1080)
            image = cv2.resize(image, new_dimension)

            pts = np.array([[257, 540], [1018, 263], [1731, 246], [1732, 643]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            mask = np.zeros((1080, 2048), dtype=np.uint8)  # Creazione di un'immagine vuota per la maschera
            cv2.fillPoly(mask, [pts], 255)  # Disegna il poligono sulla maschera

            image = cv2.bitwise_and(image, image, mask=mask)
            selected_points = True


if __name__ == "__main__":

    vidcap = cv2.VideoCapture("video_input/chievo-juve/1h-left.avi")

    success, image = vidcap.read()
    new_dimension = (2048, 1080)
    image = cv2.resize(image, new_dimension)

    # Crea un'immagine vuota
    # Lista per memorizzare i punti cliccati

    cv2.imshow("Image", image)

    # Collega la funzione di callback al clic del mouse
    cv2.setMouseCallback("Image", click_event)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ASCII per 'Esc'
            break

    print(points)