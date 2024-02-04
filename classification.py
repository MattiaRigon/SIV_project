from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from scipy.signal import savgol_filter
import matplotlib

matplotlib.use('TkAgg')

def leggi_immagini_cartella(path_cartella):
    """
    Legge le immagini presenti nella cartella specificata e restituisce un elenco di immagini.

    Args:
        path_cartella (str): Il percorso della cartella contenente le immagini.

    Returns:
        list: Un elenco di immagini lette dalla cartella.

    """
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
    """
    Calculate histograms for a list of images.

    Parameters:
    images (list): A list of input images.

    Returns:
    H (ndarray): Array of H channel histograms.
    S (ndarray): Array of S channel histograms.
    V (ndarray): Array of V channel histograms.
    """

    H, S, V = [], [], []
    for img in images:

        lower = np.array([100, 0, 0])
        upper = np.array([180, 255, 255])
        mask_hist = cv2.inRange(img, lower, upper)
        
        inverse_mask_hist = cv2.bitwise_not(mask_hist)  # Create inverse mask
        inverse_image_hist = cv2.bitwise_and(img, img, mask=inverse_mask_hist)
        hsv_img = cv2.cvtColor(inverse_image_hist, cv2.COLOR_BGR2HSV)

        h_hist = cv2.calcHist([hsv_img], [0], None, [255], [1, 257])  # Calcola l'istogramma per il canale H (tonalità)
        s_hist = cv2.calcHist([hsv_img], [1], None, [255], [0, 256])  # Calcola l'istogramma per il canale S (saturazione)
        v_hist = cv2.calcHist([hsv_img], [2], None, [255], [0, 256])  # Calcola l'istogramma per il canale V (valore)

        cv2.normalize(h_hist, h_hist, 0, 255, cv2.NORM_MINMAX)
        cv2.normalize(s_hist, s_hist, 0, 255, cv2.NORM_MINMAX)
        cv2.normalize(v_hist, v_hist, 0, 255, cv2.NORM_MINMAX)

        H.append(h_hist.flatten())
        S.append(s_hist.flatten())
        V.append(v_hist.flatten())
 
    return np.array(H), np.array(S), np.array(V)

def predict(svm_classifier, image):
    """
    Predicts the label of an image using a support vector machine classifier.

    Args:
        svm_classifier (object): The trained support vector machine classifier.
        image (numpy.ndarray): The input image.

    Returns:
        int: The predicted label for the image.
    """
    lower = np.array([100, 0, 0])
    upper = np.array([180, 255, 255])
    mask_hist = cv2.inRange(image, lower, upper)
    inverse_mask_hist = cv2.bitwise_not(mask_hist)  # Create inverse mask
    inverse_image_hist = cv2.bitwise_and(image, image, mask=inverse_mask_hist)
    hsv_img = cv2.cvtColor(inverse_image_hist, cv2.COLOR_BGR2HSV)

    h_hist = cv2.calcHist([hsv_img], [0], None, [255], [1, 257])  # Calcola l'istogramma per il canale H (tonalità)
    s_hist = cv2.calcHist([hsv_img], [1], None, [255], [0, 256])  # Calcola l'istogramma per il canale S (saturazione)
    v_hist = cv2.calcHist([hsv_img], [2], None, [255], [0, 256]) 

    cv2.normalize(h_hist, h_hist, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(s_hist, s_hist, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(v_hist, v_hist, 0, 255, cv2.NORM_MINMAX)

    hist = np.hstack((h_hist.flatten(), s_hist.flatten(), v_hist.flatten()))

    predicted_label = svm_classifier.predict([hist])
    return predicted_label


if __name__ == "__main__":

    path = "datasets/2h-left-5min"
    dataset_images = []
    labels = []
    cartelle = [nome for nome in os.listdir(path) if os.path.isdir(os.path.join(path, nome))]

    for idx, cartella in enumerate(cartelle, start=1):
        path_cartella = os.path.join(path, cartella)
        images = leggi_immagini_cartella(path_cartella)
        dataset_images += images
        labels += [idx] * len(images)

    H, S, V = calcola_istogrammi(dataset_images)

    weight_H = 2  # Peso assegnato a X1_train
    weight_S = 1  # Peso assegnato a X2_train
    weight_V = 1  # Peso assegnato a X3_train

    # Moltiplicazione delle caratteristiche per i pesi assegnati
    weighted_X1_train = H * weight_H
    weighted_X2_train = S * weight_S
    weighted_X3_train = V * weight_V

    # Combinazione delle caratteristiche pesate
    combined_weighted_features_train = np.hstack((weighted_X1_train, weighted_X2_train, weighted_X3_train))

    # Dividi il dataset in set di addestramento e test
    X_train, X_test, y_train, y_test = train_test_split(combined_weighted_features_train, labels, test_size=0.2, random_state=42)

    # Addestramento del classificatore SVM
    svm_classifier = SVC(kernel='linear',C=0.5)
    svm_classifier.fit(X_train, y_train)

    # Valutazione delle prestazioni del modello
    y_pred = svm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)


    print("Accuracy:", accuracy)
