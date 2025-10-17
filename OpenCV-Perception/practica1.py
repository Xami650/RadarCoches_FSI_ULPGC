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
    scale = 0.5  # o 0.6, 0.7... ajusta según tu pantalla
    cv2.imshow("Background", cv2.resize(background, (0,0), fx=scale, fy=scale))
    cv2.imwrite("background.jpg", cv2.resize(background, (0,0), fx=scale, fy=scale))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def obtener_coches(video, fondo):
    
    
    cap = cv2.VideoCapture(video)
    
    background = cv2.imread(fondo)

    min_area = 100# Ajusta según necesites
    max_area = 1500

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calcular diferencia absoluta
        diff = cv2.absdiff(frame, background)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Umbralización
        _, umbralizado = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        umbralizado = cv2.morphologyEx(umbralizado, cv2.MORPH_CLOSE, kernel_close)
        
        # Apertura opcional: elimina ruido pequeño residual
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        umbralizado = cv2.morphologyEx(umbralizado, cv2.MORPH_OPEN, kernel_open)

        # Mostrar resultados intermedios (opcional, para depuración)
        scale = 0.5  # o 0.6, 0.7... ajusta según tu pantalla
        cv2.imshow("Umbralizado", cv2.resize(umbralizado, (0,0), fx=scale, fy=scale))
        cv2.imshow("Diferencia sin umbral", cv2.resize(diff, (0,0), fx=scale, fy=scale))
        
        # Encontrar contornos
        contours, _ = cv2.findContours(umbralizado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output_frame = frame.copy()
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                
       
        cv2.imshow("Blops", cv2.resize(output_frame, (0,0), fx=scale, fy=scale))            
        

        if cv2.waitKey(34) & 0xFF == ord('q'):
            break
        

    cap.release()
    cv2.destroyAllWindows()
        
def main():
        
    obtener_coches('OpenCV-Perception/trafico01.mp4','OpenCV-Perception/background.jpg' )
    obtener_fondo('OpenCV-Perception/trafico01.mp4')
    
if __name__ == '__main__':
    main()
    
    