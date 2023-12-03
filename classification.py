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

def trova_intervalli_non_consecutivi(array):
    intervals = []
    start = array[0]
    end = array[0]

    for i in range(1, len(array)):
        if array[i] - array[i - 1] != 1:
            end = array[i - 1]
            if start != end:
                intervals.append((start, end))
            start = array[i]
    
    if start != array[-1]:
        intervals.append((start, array[-1]))

    return intervals



def plot_major_colors(histogram_h, histogram_s, histogram_v):
    # Trova i valori H più alti nell'istogramma
    num_colors = 5  # Numero di colori da plottare
    major_indices = np.argsort(histogram_h)[::-1][:num_colors]  # Ottieni gli indici dei valori H più alti
    colors = []  # Lista per salvare i colori corrispondenti ai valori HSV

    # Converte gli indici H, S, V in colori RGB per il plot
    for index in major_indices:
        hue_value = index * (180 / len(histogram_h))  # Converti l'indice H in valore di tonalità (H)
        sat_value = histogram_s[index] #/ max(histogram_s)  # Normalizza la saturazione (S)
        val_value = histogram_v[index] #/ max(histogram_v)  # Normalizza il valore (V)
        
        rgb_color = hsv_to_rgb(hue_value, sat_value, val_value)  # Converti HSV in RGB
        print(hue_value, sat_value, val_value, rgb_color)
        colors.append(rgb_color)

    # Plot dei colori ottenuti
    plt.figure(figsize=(8, 2))
    for i, color in enumerate(colors):
        plt.subplot(1, num_colors, i+1)
        plt.imshow([[color]], extent=[0, 1, 0, 1])
        plt.axis('off')

    plt.show()

def hsv_to_rgb(h, s, v):
    """
    Converti da spazio colore HSV a spazio colore RGB utilizzando OpenCV.

    Args:
    - h (float): Valore di tonalità compreso tra 0 e 360.
    - s (float): Valore di saturazione compreso tra 0 e 1.
    - v (float): Valore di luminosità (o valore) compreso tra 0 e 1.

    Returns:
    - tuple: Tupla contenente i valori RGB compresi tra 0 e 255.
    """

    # Creazione di un'immagine 1x1 con valore HSV specificato
    hsv_color = np.array([[[h, s , v ]]], dtype=np.uint8)
    # Conversione da HSV a RGB
    rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)
    # Estrai i valori RGB
    r, g, b = rgb_color[0, 0, :]
    return int(r), int(g), int(b)

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

def calculate_approximation(h_hist):

    x = np.arange(len(h_hist))
    degree = 10  # Grado del polinomio per la regressione
    window_length = 100 # Lunghezza della finestra per il filtro Savitzky-Golay

    # Applicazione del filtro Savitzky-Golay per la smoothing dei dati
    smoothed = savgol_filter(h_hist, window_length, 3)  # '3' è il grado del polinomio

    # Fit del polinomio ai dati smoothed
    coefficients = np.polyfit(x, smoothed, degree)
    polynomial = np.poly1d(coefficients)

    # Generazione della stima usando il polinomio
    approximation = polynomial(x)

    # Trovare i massimi della funzione stimata
    # Calcolare la derivata prima e seconda della funzione stimata
    first_derivative = np.gradient(approximation)
    second_derivative = np.gradient(first_derivative)

    # Trovare gli indici in cui la derivata prima si annulla e la derivata seconda è negativa
    maxima_indices = np.where((first_derivative[:-1] > 0) & (first_derivative[1:] < 0) & (second_derivative[1:] < 0))[0] + 1
    maxima_indices =  np.append(maxima_indices, len(approximation) - 1)
    maxima_indices = np.append(maxima_indices, 0)

    maxima_values = approximation[maxima_indices]
    
    max = np.max(maxima_values)
    threshold = 0.3 * max
    above_threshold = np.where(approximation > threshold)[0]

    intervals = trova_intervalli_non_consecutivi(above_threshold)

    print(f"intervals :{intervals}")
    # print(above_threshold)
    # print(f"max: {maxima_indices}")
    # print(maxima_values)
    plt.plot(approximation)
    plt.scatter(above_threshold, approximation[above_threshold], color='red', marker='o', label='Sopra soglia')
    plt.scatter(maxima_indices, maxima_values, color='green', marker='o', label='Massimi')
    plt.axhline(y=threshold, color='red', linestyle='--', label=f'Retta orizzontale a {threshold}')  # Traccia la retta orizzontale
    plt.plot(h_hist.flatten())
    plt.show()
        # Trova gli indici dei massimi locali
        
    return approximation


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
        s_hist = cv2.calcHist([hsv_img], [1], None, [255], [0, 256])  # Calcola l'istogramma per il canale S (saturazione)
        v_hist = cv2.calcHist([hsv_img], [2], None, [255], [0, 256])  # Calcola l'istogramma per il canale V (valore)
        # Concatena gli istogrammi in un unico vettore di feature per l'immagine
        hist_features.append(h_hist.flatten())#np.concatenate((h_hist, s_hist, v_hist)).flatten())
        # start_time = time.time()
        # f = calculate_approximation(h_hist.flatten())
        # hist_features.append(f)
        # end_time = time.time()
        # print(f"Tempo di esecuzione: {end_time - start_time}")

        # plot_major_colors(h_hist.flatten(),s_hist.flatten(),v_hist.flatten())
    
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
    # path_s1 = "img_players/cluster0"
    path = "img_players"
    dataset_images = []
    labels = []
    cartelle = [nome for nome in os.listdir(path) if os.path.isdir(os.path.join(path, nome))]

    for idx, cartella in enumerate(cartelle, start=1):
        path_cartella = os.path.join(path, cartella)
        images = leggi_immagini_cartella(path_cartella)
        dataset_images += images
        labels += [idx] * len(images)

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

    # for image in images_test:
    #     predict(svm_classifier,image)
