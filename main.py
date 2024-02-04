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
        fixed_points = generate_photo_dataset(nome_file,50,pts_cut)
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

    combined_weighted_features_train = np.hstack((H, S, V))

    X_train, X_test, y_train, y_test = train_test_split(combined_weighted_features_train, labels, test_size=0.2, random_state=42)

    svm_classifier = SVC(kernel='linear',C=1/len(dataset_images), random_state=42)
    svm_classifier.fit(X_train, y_train)

    # Read the video frame by frame
    while success:

        start_time = time.time()
        eroded_image, image = preprocess(image,pts_cut,fixed_points)        
        soccer_players = findPlayers(eroded_image, image,svm_classifier)
        show_image(image)
        transformed_img = deforma_immagine(image, pts_transformed, soccer_players)
        soccer_field_populated = populate_soccer_field(soccer_field, transformed_img,soccer_players, SOCCER_FIELD_OFFSET,isLeft)

        show_image(soccer_field_populated)


        end_time = time.time()
        fps = 1 / (end_time-start_time)

        print(f"{fps:.2f} FPS")        
        success, image = vidcap.read()
