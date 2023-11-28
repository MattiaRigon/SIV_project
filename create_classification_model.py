import cv2
import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np
from utils import *
from enum import Enum

matplotlib.use('tkagg')  # Imposta il backend su tkagg


class Squad(Enum):
    SQUAD1 = 1
    SQUAD2 = 2


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
        img = filter_image_green(img)
        if img is not None:
            elenco_immagini.append(img)
        else:
            print(f"Impossibile leggere l'immagine: {img_path}")
    
    return elenco_immagini

def filter_image_green(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
    inverted_mask = cv2.bitwise_not(mask_green)
    image_green = cv2.bitwise_and(image, image, mask=inverted_mask)

    lower = np.array([100, 0, 0])
    upper = np.array([180, 255, 255])
    mask_hist = cv2.inRange(image_green, lower, upper)
    
    inverse_mask_hist = cv2.bitwise_not(mask_hist)  # Create inverse mask
    inverse_image_hist = cv2.bitwise_and(image_green, image_green, mask=inverse_mask_hist)
    hsv_inverse_image_hist = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # show_image(inverse_image_hist)

    return hsv_inverse_image_hist

class CreateClassificationModel():
    def __init__(self,squad1 : list[object] ,squad2 : list[object]) -> None:
        self.squad1 = squad1
        self.squad2 = squad2
        pass
    
    def fit(self):
        H,S,V = [],[],[]

        for image in self.squad1:
            h, s, v = cv2.split(image)
            H.extend(h.flatten())
            S.extend(s.flatten())
            V.extend(v.flatten())
            print(get_hsv_range_from_image(image))

        H, S, V = np.array(H), np.array(S), np.array(V)

        mean_H, mean_S, mean_V = np.mean(H), np.mean(S), np.mean(V)
        std_H, std_S, std_V = np.std(H), np.std(S), np.std(V)

        # Filter values that are within 2 standard deviations from the mean
        filtered_indices = (
            # (np.abs(H - mean_H) < 2 * std_H) &
            # (np.abs(S - mean_S) < 2 * std_S) &
            # (np.abs(V - mean_V) < 2 * std_V) &
            (H != 0)  # Aggiunta della condizione per H diverso da zero
        )

        filtered_H = H[filtered_indices]
        filtered_S = S[filtered_indices]
        filtered_V = V[filtered_indices]

        # Calculate centroid of the filtered points
        centroid_H = np.mean(filtered_H)
        centroid_S = np.mean(filtered_S)
        centroid_V = np.mean(filtered_V)

        centroid1 = [centroid_H, centroid_S, centroid_V]
        self.centroid1 = np.array(centroid1)

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # # ax.scatter(filtered_H, filtered_S, filtered_V, c='red', marker='o')
        # ax.set_xlim(0, 255)  # Imposta i limiti per l'asse x
        # ax.set_ylim(0, 255)  # Imposta i limiti per l'asse y
        # ax.set_zlim(0, 255)  # Imposta i limiti per l'asse z

        
        # ax.set_xlabel('H')
        # ax.set_ylabel('S')
        # ax.set_zlabel('V')

        # # Plot the centroid
        # ax.scatter(centroid_H, centroid_S, centroid_V, c='blue', marker='^', s=100, label='Centroid')
        # ax.legend()

        # # plt.show()

        # plt.savefig("graphs/hsv1_filtered.png")
        # # plt.clf()

        ######################################################################################################

        H,S,V = [],[],[]

        for image in self.squad2:
            h, s, v = cv2.split(image)
            H.extend(h.flatten())
            S.extend(s.flatten())
            V.extend(v.flatten())   
            print(get_hsv_range_from_image(image))
     

        # Convert to numpy arrays for easier calculations
        H, S, V = np.array(H), np.array(S), np.array(V)

        # Calculate mean and standard deviation
        mean_H, mean_S, mean_V = np.mean(H), np.mean(S), np.mean(V)
        std_H, std_S, std_V = np.std(H), np.std(S), np.std(V)

        # Filter values that are within 2 standard deviations from the mean
        filtered_indices = (
            # (np.abs(H - mean_H) < 2 * std_H) &
            # (np.abs(S - mean_S) < 2 * std_S) &
            # (np.abs(V - mean_V) < 2 * std_V) &
            (H != 0)  # Aggiunta della condizione per H diverso da zero
        )

        filtered_H = H[filtered_indices]
        filtered_S = S[filtered_indices]
        filtered_V = V[filtered_indices]

        # Calculate centroid of the filtered points
        centroid_H = np.mean(filtered_H)
        centroid_S = np.mean(filtered_S)
        centroid_V = np.mean(filtered_V)

        centroid2 = [centroid_H, centroid_S, centroid_V]
        self.centroid2 = np.array(centroid2)
        # print(centroid2)

        # # # Plot the filtered points
        # # fig = plt.figure()
        # # ax = fig.add_subplot(111, projection='3d')
        # # ax.scatter(filtered_H, filtered_S, filtered_V, c='blue', marker='o')
        # ax.set_xlabel('H')
        # ax.set_ylabel('S')
        # ax.set_zlabel('V')
        # ax.set_xlim(0, 180)  # Imposta i limiti per l'asse x
        # ax.set_ylim(0, 255)  # Imposta i limiti per l'asse y
        # ax.set_zlim(0, 255)  # Imposta i limiti per l'asse z
        # # Plot the centroid
        # ax.scatter(centroid_H, centroid_S, centroid_V, c='red', marker='^', s=100, label='Centroid')
        # ax.legend()

        # plt.savefig("graphs/hsv2_filtered.png")
        # plt.clf()

    def test_classification(self):

        result : list[bool] = []
        
        for image in self.squad1:

            if self.predict(image) == Squad.SQUAD1:
                result.append(True)
                # ax.scatter(centroid_H, centroid_S,centroid_V, c='green', marker='o', s=100, label='Centroid')
            else :
                result.append(False)
                # ax.scatter(centroid_H, centroid_S,centroid_V, c='red', marker='o', s=100, label='Centroid')

            # print(centroid_H, centroid_S,centroid_V)

        for image in self.squad2:

            if self.predict(image) == Squad.SQUAD2:
                result.append(True)
                # ax.scatter(centroid_H, centroid_S,centroid_V, c='green', marker='o', s=100, label='Centroid')
            else :
                result.append(False)
                # ax.scatter(centroid_H, centroid_S,centroid_V, c='red', marker='o', s=100, label='Centroid'))

        return result.count(True) / len(result)

        
    def predict(self,image) -> Squad:

        h, s, v = cv2.split(image) 
        h = h.flatten()
        s = s.flatten()
        v = v.flatten()
        filtered_indices = ((h != 0))
        filtered_H = h[filtered_indices]
        filtered_S = s[filtered_indices]
        filtered_V = v[filtered_indices]
        centroid_H = np.mean(filtered_H)
        centroid_S = np.mean(filtered_S)
        centroid_V = np.mean(filtered_V)
        centroid = [centroid_H, centroid_S, centroid_V]
        centroid = np.array(centroid)
        distance1 = np.linalg.norm(self.centroid1 - centroid)
        distance2 = np.linalg.norm(self.centroid2 - centroid)

        if distance1 < distance2:
            return Squad.SQUAD1
        else :
            return Squad.SQUAD2
    
    def hog(self):

        # Carica l'immagine
        hog = cv2.HOGDescriptor()


        for image in self.squad1:

            # Converte l'immagine in scala di grigi
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


            # Calcola l'HOG sull'immagine in scala di grigi
            hog_result = hog.compute(gray)

            # Mostra l'HOG result (vettore di caratteristiche HOG)
            print("Shape dell'HOG result:", hog_result.shape)
            print("Primi 10 valori dell'HOG result:", hog_result[:10])

        for image in self.squad2:

            # Converte l'immagine in scala di grigi
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Calcola l'HOG sull'immagine in scala di grigi
            hog_result = hog.compute(gray)

            # Mostra l'HOG result (vettore di caratteristiche HOG)
            print("Shape dell'HOG result:", hog_result.shape)
            print("Primi 10 valori dell'HOG result:", hog_result[:10])


if __name__ == "__main__":

    path_s1 = "labeled_photo/squad1"
    path_s2 = "labeled_photo/squad2"
    images_s1 = leggi_immagini_cartella(path_s1)
    images_s2 = leggi_immagini_cartella(path_s2)
    model = CreateClassificationModel(images_s1,images_s2)
    model.fit()
    print(model.hog())