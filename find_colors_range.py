import cv2
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.use('TkAgg') 

def leggi_immagini_cartella(path_cartella):
    elenco_immagini = []
    if not os.path.exists(path_cartella):
        print(f"Il percorso '{path_cartella}' non esiste.")
        return elenco_immagini
    
    files = os.listdir(path_cartella)
    immagini = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
    for img_file in immagini:
        img_path = os.path.join(path_cartella, img_file)
        img = cv2.imread(img_path)
        
        if img is not None:
            # hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Converti l'immagine in HSV
            elenco_immagini.append(img)
        else:
            print(f"Impossibile leggere l'immagine: {img_path}")
    
    return elenco_immagini

def extract_top_colors(image):
    # Carica l'immagine

    # Converte l'immagine da BGR a RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Ridimensiona l'immagine per facilitare il calcolo dei colori
    resized_image = cv2.resize(image, (100, 100))  # Puoi regolare le dimensioni a seconda delle tue esigenze

    # Appiattisci l'immagine per ottenere una lista di colori
    pixels = resized_image.reshape((-1, 3))
    
    # Calcola i 5 colori più presenti nell'immagine
    color_counts = Counter(tuple(pixel) for pixel in pixels)
    top_colors = color_counts.most_common(6)
    if top_colors[0][0] == (0,0,0):
        top_colors = top_colors[1:6]
    return top_colors


path_test = "labeled_photo/squad1"
images_s1 = leggi_immagini_cartella(path_test)

for image in images_s1:
    top_colors = extract_top_colors(image)
    print(top_colors)
    # plt.figure(figsize=(8, 2))
    for i, color in enumerate(top_colors):
        if color[0] == (0,0,0):
            continue
        plt.subplot(1, len(top_colors), i+1)
        plt.imshow([[color[0]]], extent=[0, 1, 0, 1])
        plt.axis('off')
    plt.show()
    plt.clf()
# # Percorso dell'immagine di input
# input_image_path = 'labeled_photo/squad2/Screenshot from 2023-11-28 16-59-41.png'  # Inserisci il percorso della tua immagine

# # Estrai i 5 colori più presenti dall'immagine
# top_colors = extract_top_colors(input_image_path)



# plt.show()
# # Stampa i 5 colori più presenti nell'immagine
# print("I 5 colori più presenti nell'immagine sono:")
# for color, count in top_colors:
#     print(f"Colore: {color}, Frequenza: {count}")
