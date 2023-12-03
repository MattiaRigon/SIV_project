import time
import cv2
import numpy as np
from utils import *

from classification import *


if __name__ == "__main__":
    vidcap = cv2.VideoCapture("2h-left-5min.avi")
    # generate_photo_dataset(vidcap,50)
    success, image = vidcap.read()
    count = 0
    success = True
    idx = 0
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

    # Read the video frame by frame
    while success:

        start_time = time.time()

        eroded_image, image = preprocess(image)
        
        findPlayers(eroded_image, image,svm_classifier)

        altezza, larghezza = image.shape[:2]

        # Specifica i 4 punti per deformare l'immagine (in alto a sinistra, in alto a destra, in basso a sinistra, in basso a destra)
        pts = [[1018, 263], [1731, 246],[257, 540], [1732, 643]]
        # i =0 
        # for pt in pts:
        #     cv2.circle(image, tuple(pt), 5, (0, 0, 255), -1)
        #     cv2.putText(image, str(i), (pt[0] + 10, pt[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #     i = i+1

        show_image(image)

        # Deforma l'immagine basata sui 4 punti specificati

        # img_deformata = deforma_immagine(image, pts)

        # show_image(img_deformata)


        end_time = time.time()
        fps = 1 / (end_time-start_time)

        print(f"{fps:.2f} FPS")        
        success, image = vidcap.read()

