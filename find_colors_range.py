import cv2
import numpy as np
from show_image import show_image

# Carica l'immagine
image = cv2.imread("progetto/SIV_project/maglia-calcio-francia-2018-blu.jpg")

# Converte l'immagine in HSV (Hue, Saturation, Value)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_white = np.array([0, 0, 200])  # Range bianco
upper_white = np.array([255, 30, 255])

# Crea la maschera per il verde (sfondo)
mask_white = cv2.inRange(hsv_image, lower_white, upper_white)

inverted_mask = cv2.bitwise_not(mask_white)

# Applica la maschera verde all'immagine originale
image_not_white = cv2.bitwise_and(image, image, mask=inverted_mask)


# Definisci il riquadro (ROI) intorno alla maglietta
# Questa regione deve includere principalmente la maglietta di interesse
roi = image_not_white

# Calcola l'istogramma nell'intervallo di Hue (H)
hist = cv2.calcHist([roi], [0], None, [256], [0, 256])

# Calcola l'istogramma cumulativo
cdf = hist.cumsum()
cdf_normalized = cdf / cdf[-1]

# Trova i limiti (lower e upper) usando una soglia specifica
# Ad esempio, scegliamo di catturare l'80% dei pixel
threshold = 0.95
lower_value = np.argmax(cdf_normalized > (1 - threshold))
upper_value = np.argmax(cdf_normalized > threshold)

# Applica i valori lower e upper per la selezione del colore
lower_color = np.array([lower_value, 0, 0])
upper_color = np.array([upper_value, 255, 255])

# Filtra l'immagine per isolare il colore di interesse
mask = cv2.inRange(image_not_white, lower_color, upper_color)
result = cv2.bitwise_and(image_not_white, image_not_white, mask=mask)



# Visualizza l'immagine risultante
cv2.imshow("Immagine con filtro colore", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
