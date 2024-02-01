import time
import cv2
import numpy as np
from populate_soccer_field import populate_soccer_field
from soccer_field import SoccerField
from unsupervised_training import unsupervised_clustering
from utils import *
from classification import *
import os
from settings import *


if __name__ == "__main__":
    
    soccer_field = SoccerField(cv2.imread("soccer-field.jpg"))
    # show_image(soccer_field_img)

    nome_file = "/cagliari-chievo/2h-right-5min.avi"
    dataset_directory = f"datasets/{nome_file.replace('.avi','')}"

    if 'left' in nome_file:
        isLeft = True
        pts_cut = POINTS_LEFT_CUT
        pts_transformed = POINTS_LEFT_TRANSFORMATION
    else:
        isLeft = False
        pts_cut = POINTS_RIGHT_CUT
        pts_transformed = POINTS_RIGHT_TRANSFORMATION

    if not os.path.exists(dataset_directory):
        os.makedirs(dataset_directory)
        fixed_points = generate_photo_dataset(nome_file,50,pts_transformed)
        unsupervised_clustering(nome_file)
    else:
        fixed_points = cv2.imread(f"{dataset_directory}/fixed_points.png")
        fixed_points_gray = cv2.cvtColor(fixed_points, cv2.COLOR_BGR2GRAY)
        _, fixed_points = cv2.threshold(fixed_points_gray, 0, 255, cv2.THRESH_BINARY)
    
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

    weight_H = 1  # Peso assegnato a X1_train
    weight_S = 1  # Peso assegnato a X2_train
    weight_V = 1  # Peso assegnato a X3_train

    # Moltiplicazione delle caratteristiche per i pesi assegnati
    weighted_X1_train = H * weight_H
    weighted_X2_train = S * weight_S
    weighted_X3_train = V * weight_V

    # # # Combinazione delle caratteristiche pesate
    combined_weighted_features_train = np.hstack((weighted_X1_train, weighted_X2_train, weighted_X3_train))

    # Dividi il dataset in set di addestramento e test
    X_train, X_test, y_train, y_test = train_test_split(combined_weighted_features_train, labels, test_size=0.2, random_state=42)

    # Addestramento del classificatore SVM
    svm_classifier = SVC(kernel='linear',C=1/len(dataset_images), random_state=42)
    svm_classifier.fit(X_train, y_train)

    # Read the video frame by frame
    while success:

        start_time = time.time()

        eroded_image, image = preprocess(image,pts_cut,fixed_points)

        # show_image(eroded_image)
        
        soccer_players = findPlayers(eroded_image, image,svm_classifier)

        # # Specifica i 4 punti per deformare l'immagine (in alto a sinistra, in alto a destra, in basso a sinistra, in basso a destra)


        show_image(image)
        # # Deforma l'immagine basata sui 4 punti specificati

        transformed_img = deforma_immagine(image, pts_transformed, soccer_players)
        # show_image(transformed_img)
        # soccer_field_offset = { "x" : 96 , "y": 49 }

        soccer_field_populated = populate_soccer_field(soccer_field, transformed_img,soccer_players, SOCCER_FIELD_OFFSET,isLeft)

        # # show_image(transformed_img)
        # show_image(soccer_field_populated)
        # heatmaps = soccer_field.generate_heatmaps()


        end_time = time.time()
        fps = 1 / (end_time-start_time)

        print(f"{fps:.2f} FPS")        
        success, image = vidcap.read()

