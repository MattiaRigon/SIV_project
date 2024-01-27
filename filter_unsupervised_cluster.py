from unsupervised_training import leggi_immagini_cartella
import numpy as np
from utils import show_image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2

def filter_unsupervised_cluster(images, histograms):
    avarage_histogram = np.mean(histograms, axis=0)
    std_histogram = np.std(histograms, axis=0)
    i = 0
    difference = []
    for histogram in histograms:
        show_image(images[i])
        plt.plot(histogram)
        plt.show()
        plt.clf()
        count = 0
        for value,avarage,std in zip(histogram,avarage_histogram,std_histogram):
            # if np.abs(value - avarage) > std:
            count += np.abs(value - avarage) 
                # count += 1
        # if count < 20:
        #     show_image(images[i])
        print(count)
        i += 1
        difference.append(count)
    
    max_indices = np.argsort(difference)[-30:]
    print(max_indices)

    for i in max_indices:
        print(difference[i])
        show_image(images[i])

    
    


if __name__ == "__main__":

    path = "datasets/2h-left-5min/cluster0"

    images,histograms = leggi_immagini_cartella(path)
    histograms = np.array(histograms)
    num_clusters = 3
    # Definizione del modello DBSCAN
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)

    # Applicazione di K-Means agli istogrammi
    labels = kmeans.fit_predict(histograms)
    for i,label in enumerate(labels):
        cv2.imwrite(f"cluster{label}/"+str(i)+".png",images[i])

            
    # filter_unsupervised_cluster(images,hists)