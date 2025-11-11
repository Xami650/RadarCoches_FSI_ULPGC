import cv2
import numpy as np
import math


class Vehiculo:
    def __init__(self, id, x, y, w, h):
        self.id = id
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.centroid = (int(x + w / 2), int(y + h / 2))
        self.lost_frames = 0  # Para manejar desapariciones
        self.tracked_frames = 1 # Contar los frames en los que se ha visto
        self.contado = False # Para evitar duplicados
        

    def actualizar(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.centroid = (int(x + w / 2), int(y + h / 2))
        self.lost_frames = 0
        self.tracked_frames += 1



def distancia(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def extraer_fondo(video):
    cap = cv2.VideoCapture(video)
    
    ret, frame = cap.read()
    if not ret:
        print("Error: no se pudo leer el video.")
        return
    
    avg_frame = np.zeros_like(frame, np.float32)
    
    # set se utiliza para establecer un valor de propiedad del video.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Nos aseguramos de empezar desde el principio, nos movemos al frame 0.
    count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        avg_frame += frame.astype(np.float32)
        count += 1

    cap.release()
    
    background = avg_frame / count
    background = cv2.convertScaleAbs(background) # Convertir de float32 a uint8 en valor absoluto.
    
    cv2.imshow("Background", background)
    cv2.imwrite("background.jpg", background)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detectar_vehiculos(video, background, min_area, dist_thresh, frames_thresh, max_lost):
    # Cargar fondo y convertir a escala de grises
    fondo = cv2.imread(background)
    fondo_gray = cv2.cvtColor(fondo, cv2.COLOR_BGR2GRAY)
    
    cap = cv2.VideoCapture(video)
    vehiculos = [] # Lista para almacenar los objetos de la clase Vehículo
    id_counter = 0
    total_contados = 0 # Contador de vehículos totales contados.

        
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convertimos el frame a gris, más fácil para procesar.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Restamos el fondo
        diff = cv2.absdiff(gray, fondo_gray)
        
        # Suavizamos un poco para reducir ruido (Gaussian Blur)
        blur = cv2.GaussianBlur(diff, (5, 5), 0)
        
        # Umbralizamos: todo lo diferente al fondo se vuelve blanco
        # Umbralizado estableciendo el umbral manualmente:
            # _, umbralizado = cv2.threshold(blur, 42, 255, cv2.THRESH_BINARY)
        
        # Esto elige automáticamente el mejor umbral según el histograma del frame:
        _, umbralizado = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


        # Operaciones morfológicas para limpiar ruido    
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)) 
        # Crea una máscara 7 x 7 que se mueve por toda la imagen binaria y analiza como son los 
        # píxeles en esa region 7 x 7
        
        # Dilatar para rellenar huecos
        dilatado = cv2.dilate(umbralizado, kernel, iterations=2)
        # Se miran los pixeles a 1 y se ve si estos están rodeados de 1 o 0,
        # en dilatación si el píxel permanece a 1 si al menos uno de los que le rodea
        # está a 1 (255), en erosión es lo contrario.
        
         # Cerrar para unir regiones cercanas, dilatación -> erosión, primero expande
         # para cerrar regiones cercanas y luego erosiona para volver al tamaño original.
        cerrado = cv2.morphologyEx(dilatado, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Abrir para eliminar ruido suelto, erosión -> dilatación, primero erosiona y limpia
        # ruido pequeño y luego dilata para agrandar los objetos que quedaron.
        umbralizado = cv2.morphologyEx(cerrado, cv2.MORPH_OPEN, kernel, iterations=1)
        

        
        mask = np.zeros_like(umbralizado, dtype=np.uint8)

        # Ejemplo: solo analizar desde Y=320 hasta el final (ignorar el fondo lejano)
        # y excluir los últimos 120 px de la esquina inferior derecha donde están los números
        alto, ancho = umbralizado.shape
        cv2.rectangle(mask, (0, 320), (ancho, alto - 120), 255, -1) # -1 rellena el rectángulo de blanco.

        # Aplicamos la máscara
        umbralizado = cv2.bitwise_and(umbralizado, mask) # Operación AND entre la imagen umbralizada y la máscara.

        # (Opcional) Visualizar la ROI
        # frame_masked = cv2.bitwise_and(frame, frame, mask=mask) # Apllocamos la máscara al frame original.
        # Hace la operacion AND entre frame y frame, dejando solo la máscara.
        # cv2.imshow("ROI", frame_masked)
        
        # Analiza imagen binaria para encontrar los bordes de todos los blops.
        contornos, _ = cv2.findContours(umbralizado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # En _ se guarda la jerarquía, describe la relación entre contornos, no interesa aquí.
        # Se recuperan solo los contornos externos (RETR_EXTERNAL)
        # Guarda todos los puntos de los contornos, pero simplificados (CHAIN_APPROX_SIMPLE).
        blobs = []
        
        for contorno in contornos:
            area = cv2.contourArea(contorno) # Calcula el mejor rectángulo que encierra el contorno y devuelve x, y, w, h.
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contorno)
                blobs.append((x, y, w, h))                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # El  parámetro 2 es el grosor del rectángulo.               
                
        vistos = set()   # Para rastrear qué vehículos han sido vistos en este frame.
        usados = set()   # Para evitar ids duplicados en un mismo frame. 
        # Se utiliza usados por si un vehículo es el mejor match para dos blobs diferentes en el mismo frame.       
        for (x, y, w, h) in blobs:
            centro = (int(x + w / 2), int(y + h / 2))
            min_d = float('inf')
            match = None
            for vehiculo in vehiculos:
                if vehiculo in usados:
                    continue    # Salta al siguiente vehículo si ya ha sido usado en este frame.
                d = distancia(vehiculo.centroid, centro)
                if d < min_d and d < dist_thresh:
                    min_d = d
                    match = vehiculo
            if match:
                match.actualizar(x, y, w, h)
                vistos.add(match)
                usados.add(match)   
                if not match.contado and match.tracked_frames >= frames_thresh:
                    total_contados += 1
                    match.contado = True
                    
                cv2.putText(frame, f"id: {match.id}", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                
            else:
                id_counter += 1
                vehiculo = Vehiculo(id_counter, x, y, w, h)
                vehiculos.append(vehiculo)
                vistos.add(vehiculo)
                
                cv2.putText(frame, f"id: {vehiculo.id}", (x, y - 10), # Esquina inferior izquierda de la imagen.
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2) # 0.6 es la escala de la fuente (por defecto es 1), 2 es el grosor.

        for vehiculo in vehiculos:
            if vehiculo not in vistos:
                vehiculo.lost_frames += 1
        vehiculos = [v for v in vehiculos if v.lost_frames <= max_lost]
        
        
        # Implementar algo para la fusion de blops
        # Implementar logica para que los coches que parpadean no reciban otra id distinta
        
        

        total_vehiculos = len(vehiculos)
        cv2.putText(frame, f"Vehiculos activos: {len(vehiculos)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Vehiculos contados: {total_contados}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)    

                        
        
        # Mostrar resultado
        umbral_resized = cv2.resize(umbralizado, (1280, 720))
        frame_resized = cv2.resize(frame, (1280, 720))

        cv2.imshow("Detección de coches", umbral_resized)
        cv2.imshow("Blops marcados", frame_resized)
        
        # Salir con 'q'
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    # extraer_fondo("App/trafico01.mp4")
    detectar_vehiculos('App/trafico01.mp4', 'background.jpg', 500, 75, 10, 15)
    

if __name__ == '__main__':
    main()
    
    