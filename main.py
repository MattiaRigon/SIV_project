import time
import cv2
import numpy as np
from utils import *

from classification import *


if __name__ == "__main__":
    vidcap = cv2.VideoCapture("2h-left-5min.avi")
    success, image = vidcap.read()
    count = 0
    success = True
    idx = 0
    path_s1 = "test/squad1"
    path_s2 = "test/squad2"
    images_s1 = leggi_immagini_cartella(path_s1)
    images_s2 = leggi_immagini_cartella(path_s2)

    dataset_images = images_s1 + images_s2
    labels = [0] * len(images_s1) + [1] * len(images_s2)

    # Genera gli istogrammi HSV dalle immagini
    histogram_features = calcola_istogrammi(dataset_images)

    # Dividi il dataset in set di addestramento e test
    X_train, X_test, y_train, y_test = train_test_split(histogram_features, labels, test_size=0.2, random_state=42)

    # Addestramento del classificatore SVM
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train, y_train)


    # Read the video frame by frame
    while success:
        nuova_dimensione = (2048, 1080)
        image = cv2.resize(image, nuova_dimensione)
        pts = np.array([[257, 540], [1018, 263], [1731, 246], [1732, 643]], np.int32)
        pts = pts.reshape((-1, 1, 2))

        # Crea la maschera
        mask = np.zeros((1080, 2048), dtype=np.uint8)  # Creazione di un'immagine vuota per la maschera
        cv2.fillPoly(mask, [pts], 255)  # Disegna il poligono sulla maschera

        # Applica la maschera all'immagine originale
        image = cv2.bitwise_and(image, image, mask=mask)
        start_time = time.time()
        campo = trovaCampoDaGioco(image=image)
        masked_image = cv2.bitwise_and(image, image, mask=campo)
        hist_hue = cv2.calcHist([masked_image], [0], None, [180], [0, 180])
        hist_saturation = cv2.calcHist([masked_image], [1], None, [256], [0, 256])
        hist_value = cv2.calcHist([masked_image], [2], None, [256], [0, 256])
        
        hist_hue = hist_hue[1:]  # Remove the first element from hist_hue
        hist_saturation = hist_saturation[1:-1]  # Remove the first and last element from hist_saturation
        hist_value = hist_value[1:-1]  # Remove the first and last element from hist_value

        lower = np.array([100, 0, 0])
        upper = np.array([180, 255, 255])
        mask_hist = cv2.inRange(masked_image, lower, upper)
        
        image_hist = cv2.bitwise_and(masked_image, masked_image, mask=mask_hist)
        inverse_mask_hist = cv2.bitwise_not(mask_hist)  # Create inverse mask
        inverse_image_hist = cv2.bitwise_and(masked_image, masked_image, mask=inverse_mask_hist)
        gray_image = cv2.cvtColor(inverse_image_hist, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
        if binary_image.dtype != np.uint8:
            binary_image = binary_image.astype(np.uint8)

        findPlayers(binary_image, image,svm_classifier)

        end_time = time.time()
        fps = 1 / (end_time-start_time)

        print(f"{fps:.2f} FPS")        
        success, image = vidcap.read()
