import cv2
import numpy as np
from show_image import show_image
import copy
import time


def trovaRigheCampo(image):

    # Converti l'immagine da BGR a HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Definisci i range HSV per il verde (sfondo)
    #white range
    lower_white = np.array([0,0,0])
    upper_white = np.array([0,0,255])

    # Crea la maschera per il verde (sfondo)
    mask_white = cv2.inRange(hsv_image, lower_white, upper_white)

    show_image(image)

    # Applica la maschera verde all'immagine originale
    image_green = cv2.bitwise_and(image, image, mask=mask_white)

    # Converte l'immagine verde in scala di grigi
    gray_image = cv2.cvtColor(image_green, cv2.COLOR_BGR2GRAY)

    # Applica una soglia, dove il verde diventa 0 e il resto diventa 255
    _, thresholded_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY_INV)

    return thresholded_image


def trovaCampoDaGioco(image):

    # Converti l'immagine da BGR a HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Definisci i range HSV per il verde (sfondo)
    lower_green = np.array([35, 100, 100])  # Range verde
    upper_green = np.array([85, 255, 255])

    # Crea la maschera per il verde (sfondo)
    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)

    # Applica la maschera verde all'immagine originale
    image_green = cv2.bitwise_and(image, image, mask=mask_green)

    # Converte l'immagine verde in scala di grigi
    gray_image = cv2.cvtColor(image_green, cv2.COLOR_BGR2GRAY)

    # Applica una soglia, dove il verde diventa 0 e il resto diventa 255
    _, thresholded_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY_INV)

    # Trova i contorni degli oggetti nell'immagine
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Cicla sui contorni e disegna solo quelli che sono "isolati" nel bianco
    cleaned_image = np.ones_like(image,dtype=np.uint8) * 255  # Inizializza un'immagine completamente bianca

    # Cicla sui contorni e disegna solo quelli che sono "isolati" nel bianco
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10 :  # Definisci un valore soglia per l'area
            cv2.drawContours(cleaned_image, [contour], -1, (0), thickness=cv2.FILLED)  # Aggiungi il contorno alla maschera

    cleaned_image = 255 - cleaned_image
    cleaned_image = cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2GRAY)
    result =  copy.deepcopy(thresholded_image)
    result[cleaned_image == 255] = 255
    difference = result - thresholded_image
    contours, _ = cv2.findContours(difference, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Calcola l'area del contorno
        area = cv2.contourArea(contour)
        if area > 600 :  # Definisci un valore soglia per l'area
            cv2.drawContours(result, [contour], -1, (0), thickness=cv2.FILLED)  # Aggiungi il contorno alla maschera
    show_image(result)
    return result

def findPlayers(campo,image):

    contours,hierarchy = cv2.findContours(campo,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if(h>=(1)*w):
            if(w>15 and h>= 15):
                cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),3)

    # show_image(image)


if __name__ == "__main__":
    # Carica l'immagine del campo da calcio
    image_path = 'progetto/belgio-francia.png'  # Sostituisci con il percorso del tuo file immagine
    image = cv2.imread(image_path)
    # Misura il tempo di esecuzione
    # campo = trovaCampoDaGioco(image=image)
    # findPlayers(campo=campo,image=image)
    righe = trovaRigheCampo(image)
    show_image(righe)

