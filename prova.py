from collections import Counter
import time
import numpy as np
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import PIL
import matplotlib

from sklearn.cluster import DBSCAN, KMeans
from utils import show_image
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
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Converti l'immagine in HSV
        # Riduci il numero di valori di H da 180 a 8 usando il modulo
        quantized_h = np.uint8((hsv_img[:,:,0] ) % 8)

        # Aggiorna il valore di H nell'immagine HSV
        hsv_img[:,:,0] = quantized_h 

        # Converti l'immagine HSV in RGB
        new_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
        show_img_compar(img, new_img)
        
        if img is not None:
            # hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Converti l'immagine in HSV
            elenco_immagini.append(img)
        else:
            print(f"Impossibile leggere l'immagine: {img_path}")
    
    return elenco_immagini

def show_img_compar(img_1, img_2 ):
    f, ax = plt.subplots(1, 2, figsize=(10,10))
    ax[0].imshow(img_1)
    ax[1].imshow(img_2)
    ax[0].axis('off') #hide the axis
    ax[1].axis('off')
    f.tight_layout()
    plt.show()

def palette_perc(k_cluster):
    width = 300
    palette = np.zeros((50, width, 3), np.uint8)
    
    n_pixels = len(k_cluster.labels_)
    counter = Counter(k_cluster.labels_) # count how many pixels per cluster
    perc = {}
    for i in counter:
        perc[i] = np.round(counter[i]/n_pixels, 2)
    perc = dict(sorted(perc.items()))
    
    #for logging purposes
    print(perc)
    print(k_cluster.cluster_centers_)
    
    step = 0
    
    for idx, centers in enumerate(k_cluster.cluster_centers_): 
        palette[:, step:int(step + perc[idx]*width+1), :] = centers
        step += int(perc[idx]*width+1)
        
    return palette

def palette_perc_dbscan(dbscan_result, data):
    width = 300
    palette = np.zeros((50, width, 3), np.uint8)
    
    n_pixels = len(dbscan_result.labels_)
    counter = Counter(dbscan_result.labels_)  # Conta quanti pixel per cluster
    perc = {}
    for label, count in counter.items():
        perc[label] = np.round(count / n_pixels, 2)
    perc = dict(sorted(perc.items()))
    
    # Per scopi di log
    print(perc)
    
    step = 0
    colors = np.random.randint(0, 256, size=(len(perc), 3))  # Genera colori casuali per ogni cluster
    
    for idx, label in enumerate(sorted(perc.keys())): 
        mask = dbscan_result.labels_ == label
        cluster_indices = np.where(mask)[0]
        cluster_pixels = data[cluster_indices]  # Estrai i punti di dati originali usando gli indici del cluster
        cluster_color = np.mean(cluster_pixels, axis=0)  # Calcola il colore medio dei punti di dati nel cluster
        colors[idx] = cluster_color
        
        cluster_width = int(perc[label] * width + 1)
        palette[:, step:step + cluster_width, :] = colors[idx]
        step += cluster_width
        
    return palette

# for img in images_s1:


# Leggi l'immagine
img = cv2.imread('campo.png')

# Trasforma l'immagine in un array di pixel
pixels = img.reshape((-1, 3))

# Definisci il numero di cluster
n_clusters = 3
clt = KMeans(n_clusters=n_clusters)
clt.fit(pixels)

# # Trova l'etichetta del cluster più frequente
# labels = clt.labels_
# unique, counts = np.unique(labels, return_counts=True)
# dominant_label = unique[np.argmax(counts)]

# # Crea una maschera per selezionare solo la classe più frequente
# mask = (labels == dominant_label).reshape(img.shape[:2])

# # Applica la maschera all'immagine originale
# masked_image = np.copy(img)
# masked_image[~mask] = 0


# show_image(masked_image)


show_img_compar(img, palette_perc(clt))

# start_time = time.time()
# dbscan = DBSCAN(eps=100, min_samples=5)
# clt = dbscan.fit(img.reshape(-1, 3))#dbscan.fit(np.array([h, s,v]).T).labels_
# print(img.reshape(-1, 3).shape)
# end_time = time.time()
# print(f"Tempo di esecuzione: {end_time - start_time} secondi")
# #colors = np.array(["red", "green", "blue", "gray"])[labels]
# show_img_compar(img, palette_perc_dbscan(clt,img.reshape(-1, 3)))

# # # clt_2 = clt.fit(img_2.reshape(-1, 3))
# show_img_compar(img_2, palette_perc(clt_2))