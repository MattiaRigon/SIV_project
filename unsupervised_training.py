import os
import cv2
from sklearn.cluster import DBSCAN, KMeans
import numpy as np
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt

def leggi_immagini_cartella(path_cartella):
    elenco_hist = []
    elenco_immagini = []
    if not os.path.exists(path_cartella):
        print(f"Il percorso '{path_cartella}' non esiste.")
        return elenco_hist
    
    files = os.listdir(path_cartella)
    immagini = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
    for img_file in immagini:
        img_path = os.path.join(path_cartella, img_file)
        img = cv2.imread(img_path)
        elenco_immagini.append(img)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv_img], [0], None, [179], [1, 180])  # Calcola l'istogramma per il canale H (tonalità)
        cv2.normalize(h_hist, h_hist, 0, 255, cv2.NORM_MINMAX)
        
        if img is not None:
            # hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Converti l'immagine in HSV
            elenco_hist.append(h_hist.flatten())
        else:
            print(f"Impossibile leggere l'immagine: {img_path}")

    return elenco_immagini,elenco_hist

def unsupervised_clustering(nome_file):
    images,histograms = leggi_immagini_cartella(f"datasets/{nome_file.replace('.avi','')}")
    histograms = np.array(histograms)
    num_clusters = 2
    # Definizione del modello DBSCAN
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)

    # Applicazione di K-Means agli istogrammi
    labels = kmeans.fit_predict(histograms)

    # Numero di cluster trovati
    num_clusters_found = len(set(labels))
    print(f"Numero di cluster trovati: {num_clusters_found}")

    # Stampa delle etichette assegnate da K-Means
    print("\nEtichette assegnate da K-Means:")
    for label in np.unique(labels):
        print(f"Cluster {label}")


    for i in range(num_clusters_found):
        directory = f"datasets/{nome_file.replace('.avi','')}/cluster{i}"
        if not os.path.exists(directory):
            os.makedirs(directory)

    for i,label in enumerate(labels):
        cv2.imwrite(f"datasets/{nome_file.replace('.avi','')}/cluster{label}/"+str(i)+".png",images[i])


if __name__ == "__main__":
    nome_file = "2h-left-5min.avi"
    images,histograms = leggi_immagini_cartella(f"datasets/{nome_file.replace('.avi','')}")
    histograms = np.array(histograms)
    num_clusters = 3
    # Definizione del modello DBSCAN
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)

    # Applicazione di K-Means agli istogrammi
    labels = kmeans.fit_predict(histograms)

    # Numero di cluster trovati
    num_clusters_found = len(set(labels))
    print(f"Numero di cluster trovati: {num_clusters_found}")

    # Stampa delle etichette assegnate da K-Means
    print("\nEtichette assegnate da K-Means:")
    for label in np.unique(labels):
        print(f"Cluster {label}")


    for i in range(num_clusters_found):
        directory = f"datasets/{nome_file.replace('.avi','')}/cluster{i}"
        if not os.path.exists(directory):
            os.makedirs(directory)

    for i,label in enumerate(labels):
        cv2.imwrite(f"datasets/{nome_file.replace('.avi','')}/cluster{label}/"+str(i)+".png",images[i])

    centroids = kmeans.cluster_centers_

    # Calcola le distanze dei campioni dai rispettivi centroidi
    for i in range(num_clusters):
        distances = np.linalg.norm(histograms[labels == i] - centroids[i], axis=1)

        # Stampa le distanze per i campioni nel cluster i
        print(f"Distanze nel cluster {i}:")
        print(distances)
