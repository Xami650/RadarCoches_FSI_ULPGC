import cv2
import numpy as np


def obtener_fondo(video):
    cap = cv2.VideoCapture(video)

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

def obtener_coches(video, fondo):
    cap = cv2.VideoCapture(video)
    background = cv2.imread(fondo)

    ret, frame = cap.read()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        diff = cv2.absdiff(frame, background)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        ret, umbralizado = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY) # Los pixeles por debajo de 50 se ponen a 0, los demas a 255
        cv2.imshow("Diferencia sin umbral", diff)
        cv2.imshow("Umbralizado", umbralizado)
    
    



        if cv2.waitKey(34) & 0xFF == ord('q'):
            break
        

    cap.release()
    cv2.destroyAllWindows()
        
def main():
        
    obtener_coches('OpenCV-Perception/trafico01.mp4','OpenCV-Perception/background.jpg' )
    
if __name__ == '__main__':
    main()
    
    