import time
import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
from unsupervised_training import unsupervised_clustering
from utils import *

from classification import *
import os


if __name__ == "__main__":

    nome_file = "/cagliari-chievo/2h-left-5min.avi"
    dataset_directory = f"datasets/{nome_file.replace('.avi','')}"

    if not os.path.exists(dataset_directory):
        os.makedirs(dataset_directory)
        generate_photo_dataset(nome_file,50)
        unsupervised_clustering(nome_file)
    vidcap = cv2.VideoCapture(f"video_input/{nome_file}")
    success, image = vidcap.read()
    count = 0
    success = True
    idx = 0
    path = dataset_directory
    dataset_images = []
    labels = []
    cartelle = [nome for nome in os.listdir(dataset_directory) if os.path.isdir(os.path.join(dataset_directory, nome))]

    for idx, cartella in enumerate(cartelle, start=1):
        path_cartella = os.path.join(dataset_directory, cartella)
        images = leggi_immagini_cartella(path_cartella)
        dataset_images += images
        labels += [idx] * len(images)

    H, S, V = calcola_istogrammi(dataset_images)

    # data = []

    # # for i in range(len(H)):
    # #     data.append([H[i], S[i], V[i]])
    # data = np.array([H, S, V])
    # # # Assegnazione dei pesi alle caratteristiche
    weight_H = 1  # Peso assegnato a X1_train
    weight_S = 1  # Peso assegnato a X2_train
    weight_V = 1  # Peso assegnato a X3_train

    # Moltiplicazione delle caratteristiche per i pesi assegnati
    weighted_X1_train = H * weight_H
    weighted_X2_train = S * weight_S
    weighted_X3_train = V * weight_V

    # # # Combinazione delle caratteristiche pesate
    combined_weighted_features_train = np.hstack((weighted_X1_train, weighted_X2_train, weighted_X3_train))

    # # data = [[H, S, V], ...]  # inserire i nuovi dati in un formato simile a quello di addestramento
    # Genera gli istogrammi HSV dalle immagini

    # Dividi il dataset in set di addestramento e test
    X_train, X_test, y_train, y_test = train_test_split(combined_weighted_features_train, labels, test_size=0.2, random_state=42)

    # Addestramento del classificatore SVM
    svm_classifier = SVC(kernel='linear',C=1/len(dataset_images), random_state=42)
    svm_classifier.fit(X_train, y_train)

    # plot_decision_surface(X_train, y_train, svm_classifier)

    # Read the video frame by frame
    while success:

        start_time = time.time()

        eroded_image, image = preprocess(image)

        # show_image(image)
        
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

