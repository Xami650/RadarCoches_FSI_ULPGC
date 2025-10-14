import cv2
import numpy as np

"""
CÓDIGO PARA SACAR EL BACKGROUND


cap = cv2.VideoCapture("C:\\Users\\Usuario\\OneDrive - Colegio Sagrado Corazón de Tafira\\Escritorio\\OpenCV-Perception\\OpenCV-Perception\\trafico01.mp4")

# Variables para acumular frames
ret, frame = cap.read()
avg_frame = np.zeros_like(frame, np.float32)  # acumulador en float para que no se desborde cuando supere 255

# Volvemos al inicio
cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Empezar desde el primer frame

# Sumar todos los frames
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Convertir a float y acumular
    avg_frame += frame.astype(np.float32)
    count += 1

cap.release() # libera los recursos del vídeo

# Dividir entre el número de frames -> promedio
background = avg_frame / count

# Convertimos de nuevo a uint8 para verlo
background = cv2.convertScaleAbs(background) # convertimos a uint8 (0-255)

# Mostrar y guardar
cv2.imshow("Background", background)
cv2.imwrite("background.jpg", background)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

cap = cv2.VideoCapture("OpenCV-Perception/trafico01.mp4")
background = cv2.imread("OpenCV-Perception/background.jpg")

ret, frame = cap.read()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    diff = cv2.absdiff(frame, background)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    ret, umbralizado = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY) # Los pixeles por debajo de 50 se ponen a 0, los demas a 255
    cv2.imshow("Video normal", frame)
    cv2.imshow("Coches", diff)
    cv2.imshow("Umbralizado", umbralizado)
    
    



    if cv2.waitKey(34) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
    