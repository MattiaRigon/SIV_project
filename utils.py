import time
import cv2
import numpy as np
from show_image import show_image
import copy
import matplotlib

lower_blue = np.array([90, 50, 50])  # Valori minimi per Hue, Saturation e Value
upper_blue = np.array([150, 255, 255])  # Valori massimi per Hue, Saturation e Value

# lower_red = np.array([0, 50, 50])      # Valori minimi per Hue, Saturation e Value
# upper_red = np.array([30, 255, 255]) 

lower_red = np.array([0, 100, 100])    # Valori minimi per Hue, Saturation e Value
upper_red = np.array([10, 255, 255])    # Valori massimi per Hue, Saturation e Value


font = cv2.FONT_HERSHEY_SIMPLEX

def trovaRigheCampo(image):


    # Converti l'immagine in scala di grigi
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Applica la rilevazione dei bordi (ad esempio, Canny)
    edges = cv2.Canny(gray, 70, 140)

    show_image(edges)

    # Utilizza HoughLinesP per rilevare linee
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 10, 50)

    # Disegna le linee rilevate sull'immagine originale
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return image



def trovaCampoDaGioco(image):

    # Converti l'immagine da BGR a HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Definisci i range HSV per il verde (sfondo)
    lower_green = np.array([35, 100, 100])  # Range verde
    upper_green = np.array([85, 255, 255])

    # Crea la maschera per il verde (sfondo)
    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)

    # Applica la maschera verde all'immagine originale
    image_green = cv2.bitwise_and(image, image, mask=mask_green)

    # Converte l'immagine verde in scala di grigi
    gray_image = cv2.cvtColor(image_green, cv2.COLOR_BGR2GRAY)

    # Applica una soglia, dove il verde diventa 0 e il resto diventa 255
    _, thresholded_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY_INV)

    # Trova i contorni degli oggetti nell'immagine
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Cicla sui contorni e disegna solo quelli che sono "isolati" nel bianco
    cleaned_image = np.ones_like(image,dtype=np.uint8) * 255  # Inizializza un'immagine completamente bianca

    # Cicla sui contorni e disegna solo quelli che sono "isolati" nel bianco
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10 :  # Definisci un valore soglia per l'area
            cv2.drawContours(cleaned_image, [contour], -1, (0), thickness=cv2.FILLED)  # Aggiungi il contorno alla maschera

    cleaned_image = 255 - cleaned_image
    cleaned_image = cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2GRAY)
    result =  copy.deepcopy(thresholded_image)
    result[cleaned_image == 255] = 255
    difference = result - thresholded_image
    contours, _ = cv2.findContours(difference, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Calcola l'area del contorno
        area = cv2.contourArea(contour)
        if area > 600 :  # Definisci un valore soglia per l'area
            cv2.drawContours(result, [contour], -1, (0), thickness=cv2.FILLED)  # Aggiungi il contorno alla maschera
    # show_image(result)
    return result

def filter_and_find_players(player_image,binary_player_image):

    contours,hierarchy = cv2.findContours(binary_player_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    squad1 = np.zeros_like(binary_player_image)
    squad2 = np.zeros_like(binary_player_image)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 200  :
            mask = np.zeros_like(binary_player_image)
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)            
            player_image = cv2.bitwise_and(player_image, player_image,mask=mask)
            player_hsv = cv2.cvtColor(player_image,cv2.COLOR_BGR2HSV)

            mask1 = cv2.inRange(player_hsv, lower_blue, upper_blue)
            res1_img = cv2.bitwise_and(player_image, player_image, mask=mask1)
            res1 = cv2.cvtColor(res1_img,cv2.COLOR_HSV2BGR)
            res1 = cv2.cvtColor(res1,cv2.COLOR_BGR2GRAY)
            nzCount1 = cv2.countNonZero(res1)
            if nzCount1 > 50 :

                _, thresholded_squad1 = cv2.threshold(res1, 1, 255, cv2.THRESH_BINARY)
                squad1 = cv2.add(squad1, thresholded_squad1)                
                # show_image(squad1)

            mask2 = cv2.inRange(player_hsv, lower_red, upper_red)
            res2_img = cv2.bitwise_and(player_image, player_image, mask=mask2)
            res2 = cv2.cvtColor(res2_img,cv2.COLOR_HSV2BGR)
            res2 = cv2.cvtColor(res2,cv2.COLOR_BGR2GRAY)
            nzCount2 = cv2.countNonZero(res2)
            if nzCount2 > 50 :
                _, thresholded_squad2 = cv2.threshold(res2, 1, 255, cv2.THRESH_BINARY)
                squad2 = cv2.add(squad2, thresholded_squad2)                
    return squad1,squad2

def bounding_box_player(image,binary,squad):
    contours,_ = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if squad == 1:
        color = (255,0,0)
    else:
        color = (0,0,255)
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if(w>15 and h>= 15 ):
            cv2.rectangle(image,(x,y),(x+w,y+h),color,3)
        
def findPlayers(campo,image):

    contours,hierarchy = cv2.findContours(campo,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    squad1 = np.zeros_like(campo)
    squad2 = np.zeros_like(campo)
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        # area = cv2.contourArea(c)

        # if(w>15 and h>= 15 and area > 200):
        player_img = image[y:y+h,x:x+w]
        binary_player_img = campo[y:y+h,x:x+w]

        mask = np.zeros_like(binary_player_img)
        cv2.drawContours(mask, [c], -1, (255, 255, 255), thickness=cv2.FILLED)            
        s1,s2 = filter_and_find_players(player_img,binary_player_img)
        squad1[y:y+h,x:x+w] = cv2.add(squad1[y:y+h,x:x+w], s1)                
        squad2[y:y+h,x:x+w] = cv2.add(squad2[y:y+h,x:x+w], s2)    

    # s1 = cv2.bitwise_and(image, image,mask=squad1)
    # show_image(s1)
    # s2 = cv2.bitwise_and(image, image,mask=squad2)
    # show_image(s2)

    bounding_box_player(image,squad1,1)
    bounding_box_player(image,squad2,0)


            # if nzCount1 > 50:
            #     cv2.putText(image, 'Squadra1', (x-2, y-2), font, 0.8, (255,0,0), 2, cv2.LINE_AA)
            #     cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),3)



            # if nzCount2 > 50:

            #     cv2.putText(image, 'Squadra2', (x-2, y-2), font, 0.8, (0,0,255), 2, cv2.LINE_AA)
            #     cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),3)
            # #If player has red jersy
            # mask2 = cv2.inRange(player_hsv, lower_red, upper_red)
            # res2 = cv2.bitwise_and(player_img, player_img, mask=mask2)
            # res2 = cv2.cvtColor(res2,cv2.COLOR_HSV2BGR)
            # res2 = cv2.cvtColor(res2,cv2.COLOR_BGR2GRAY)
            # nzCount2 = cv2.countNonZero(res2)
    show_image(image)





if __name__ == "__main__":
    # Carica l'immagine del campo da calcio
    image_path = 'progetto/belgio-francia.png'  # Sostituisci con il percorso del tuo file immagine
    image = cv2.imread(image_path)
    # Misura il tempo di esecuzione
    # campo = trovaCampoDaGioco(image=image)
    # findPlayers(campo=campo,image=image)
    righe = trovaRigheCampo(image)
    show_image(righe)

