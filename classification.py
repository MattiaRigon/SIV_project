from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import *

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


def calcola_istogrammi(images):
    hist_features = []
    for img in images:

        lower = np.array([100, 0, 0])
        upper = np.array([180, 255, 255])
        mask_hist = cv2.inRange(img, lower, upper)
        
        inverse_mask_hist = cv2.bitwise_not(mask_hist)  # Create inverse mask
        inverse_image_hist = cv2.bitwise_and(img, img, mask=inverse_mask_hist)
        # show_image(inverse_image_hist)
        hsv_img = cv2.cvtColor(inverse_image_hist, cv2.COLOR_BGR2HSV)
        # show_image(hsv_img)
        # show_image(inverse_image_hist)
        h_hist = cv2.calcHist([hsv_img], [0], None, [179], [1, 180])  # Calcola l'istogramma per il canale H (tonalità)
        # s_hist = cv2.calcHist([hsv_img], [1], None, [256], [0, 256])  # Calcola l'istogramma per il canale S (saturazione)
        # v_hist = cv2.calcHist([hsv_img], [2], None, [256], [0, 256])  # Calcola l'istogramma per il canale V (valore)
        # Concatena gli istogrammi in un unico vettore di feature per l'immagine
        hist_features.append(h_hist.flatten())#np.concatenate((h_hist, s_hist, v_hist)).flatten())
    
    return np.array(hist_features)

def predict(svm_classifier,image):
    lower = np.array([100, 0, 0])
    upper = np.array([180, 255, 255])
    mask_hist = cv2.inRange(image, lower, upper)
    inverse_mask_hist = cv2.bitwise_not(mask_hist)  # Create inverse mask
    inverse_image_hist = cv2.bitwise_and(image, image, mask=inverse_mask_hist)
    hsv_img = cv2.cvtColor(inverse_image_hist, cv2.COLOR_BGR2HSV)

    # show_image(hsv_img)
    h_hist = cv2.calcHist([hsv_img], [0], None, [179], [1, 180])  # Calcola l'istogramma per il canale H (tonalità)
    predicted_label = svm_classifier.predict([h_hist.flatten()])
    return predicted_label
    # print(predicted_label)




if __name__ == "__main__":
    # Combina le immagini delle due cartelle
    path_s1 = "test/squad1"
    path_s2 = "test/squad2"
    path_test = "test"
    images_s1 = leggi_immagini_cartella(path_s1)
    images_s2 = leggi_immagini_cartella(path_s2)
    images_test = leggi_immagini_cartella(path_test)


    dataset_images = images_s1 + images_s2
    labels = [0] * len(images_s1) + [1] * len(images_s2)

    # Genera gli istogrammi HSV dalle immagini
    histogram_features = calcola_istogrammi(dataset_images)

    # Dividi il dataset in set di addestramento e test
    X_train, X_test, y_train, y_test = train_test_split(histogram_features, labels, test_size=0.2, random_state=42)

    # Addestramento del classificatore SVM
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train, y_train)

    # Valutazione delle prestazioni del modello
    y_pred = svm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)


    print("Accuracy:", accuracy)

    # for image in images_s1:
    #     predict(svm_classifier,image)
    # for image in images_s2:
    #     predict(svm_classifier,image)

    for image in images_test:
        predict(svm_classifier,image)
