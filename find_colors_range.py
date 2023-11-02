import cv2
import numpy as np

# Carica l'immagine
image = cv2.imread("tua_immagine.jpg")

# Converte l'immagine in HSV (Hue, Saturation, Value)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Definisci il riquadro (ROI) intorno alla maglietta
# Questa regione deve includere principalmente la maglietta di interesse
roi = hsv_image[y1:y2, x1:x2]

# Calcola l'istogramma nell'intervallo di Hue (H)
hist = cv2.calcHist([roi], [0], None, [256], [0, 256])

# Calcola l'istogramma cumulativo
cdf = hist.cumsum()
cdf_normalized = cdf / cdf[-1]

# Trova i limiti (lower e upper) usando una soglia specifica
# Ad esempio, scegliamo di catturare l'80% dei pixel
threshold = 0.8
lower_value = np.argmax(cdf_normalized > (1 - threshold))
upper_value = np.argmax(cdf_normalized > threshold)

# Applica i valori lower e upper per la selezione del colore
lower_color = np.array([lower_value, 0, 0])
upper_color = np.array([upper_value, 255, 255])

# Filtra l'immagine per isolare il colore di interesse
mask = cv2.inRange(hsv_image, lower_color, upper_color)
result = cv2.bitwise_and(image, image, mask=mask)

# Visualizza l'immagine risultante
cv2.imshow("Immagine con filtro colore", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
