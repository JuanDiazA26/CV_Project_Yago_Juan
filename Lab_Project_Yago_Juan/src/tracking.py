import cv2
import numpy as np
import time

class KalmanTracker:
    def __init__(self):
        # Inicializar el Filtro de Kalman
        # 4 variables de estado (x, y, velocidad_x, velocidad_y)
        # 2 variables de medición (x, y) - Lo que nos da la cámara
        self.kf = cv2.KalmanFilter(4, 2)
        
        # --- 1. Matriz de Transición (A) ---
        # Define cómo cambia el estado de un frame a otro (Física del movimiento)
        # x_nuevo = x_viejo + vx * dt
        # y_nuevo = y_viejo + vy * dt
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0], # x + vx
            [0, 1, 0, 1], # y + vy
            [0, 0, 1, 0], # vx constante (inercia)
            [0, 0, 0, 1]  # vy constante (inercia)
        ], np.float32)

        # --- 2. Matriz de Medición (H) ---
        # Relaciona el estado interno con lo que medimos.
        # Solo medimos posición (x, y), no velocidad.
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)

        # --- 3. Ruido del Proceso (Q) - AJUSTE CLAVE ---
        # Cuánto confiamos en nuestro modelo físico. 
        # En baloncesto, la pelota acelera (gravedad) y rebota. Como nuestro modelo
        # asume velocidad constante, necesitamos dar flexibilidad (ruido) aquí.
        # Valores bajos = movimiento muy suave/rectilíneo. 
        # Valores altos = permite cambios bruscos de dirección.
        self.kf.processNoiseCov = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 5, 0], # Más ruido en velocidad para permitir aceleración
            [0, 0, 0, 5]
        ], np.float32) * 1 # Con 0.5 perfecto

        # --- 4. Ruido de la Medición (R) - AJUSTE CLAVE ---
        # Cuánto confiamos en la detección visual (la "mancha" de color).
        # Si tu detección por color tiembla mucho, aumenta este valor.
        # Esto hará que el filtro ignore el jitter y prefiera la suavidad.
        self.kf.measurementNoiseCov = np.array([
            [1, 0],
            [0, 1]
        ], np.float32) * 1  # Valor medio. Subir si la detección es muy mala.

        # --- 5. Matriz de Error (P) ---
        # Incertidumbre inicial (no sabemos dónde empieza la pelota)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1

    def predict(self):
        """
        Paso 1: Predicción física.
        Calcula dónde DEBERÍA estar la pelota basándose en su velocidad anterior.
        Útil cuando la pelota desaparece o sale del plano.
        """
        prediction = self.kf.predict()
        # Devolvemos solo (x, y)
        return int(prediction[0]), int(prediction[1])

    def update(self, x, y):
        """
        Paso 2: Corrección visual.
        Si detectamos la pelota con color, usamos ese dato para corregir al Kalman.
        """
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measurement)

def interseccion_segmento_parabola(a, b, c, p1, p2):
    x_coords = np.array([p1[0], p2[0]])
    y_coords = np.array([p1[1], p2[1]])
    m, n = np.polyfit(x_coords, y_coords, 1)

    A = a
    B = b - m
    C = c - n

    discriminante = B**2 - 4*A*C
    
    if discriminante < 0:
        return False

    raices_x = [(-B + np.sqrt(discriminante)) / (2*A), 
                (-B - np.sqrt(discriminante)) / (2*A)]

    x_min, x_max = min(p1[0], p2[0]), max(p1[0], p2[0])
    y_min, y_max = min(p1[1], p2[1]), max(p1[1], p2[1])

    for x_int in raices_x:
        y_int = a*x_int**2 + b*x_int + c
        
        if x_min <= x_int <= x_max and y_min <= y_int <= y_max:
            return True

def kalman_tracking():
    tracker = KalmanTracker()
    cap = cv2.VideoCapture(0)

    points_prediction = []
    points_real = []

    max_found = False
    fitted = False
    fitted_real = False
    resultado_predict = False

    tiempo_anterior = 0
    tiempo_actual = 0
    aciertos = 0
    fallos = 0
    pred_correcta = 0
    pred_error = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        # --- A. PARTE DE VISIÓN (Tu código de color) ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Rango naranja aproximado (ajustar a tu pelota)
        mask = cv2.inRange(hsv, (7, 90, 70), (14, 225, 230)) # 5, 65, 100; 15, 255, 255
        # 5,65,100;10,200,200
        # Con 11, 65, 100; 15, 200, 200 más o menos va
        # (7, 140, 50), (12, 225, 200)
        
        # Limpieza morfológica para mejorar la "mancha"
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detectado = False
        cx, cy = 0, 0

        if contours:
            pelota_contour = max(contours, key=cv2.contourArea)
            # Solo consideramos si es lo suficientemente grande
            if cv2.contourArea(pelota_contour) > 500:
                x, y, w, h = cv2.boundingRect(pelota_contour)
                cx = (x+x+w)//2
                cy = (y+y+h)//2
                detectado = True

                if cx < 1200 and cy < 360: # and len(points_cuadratic) < 5
                    points_prediction.append((cx, cy))
                    points_real.append((cx, cy))
                    if len(points_prediction) > 2 and points_prediction[-1][1] > points_prediction[-2][1] and points_prediction[-2][1] < points_prediction[-3][1]:
                        max_found = True

                if len(points_prediction) > 2: #max_found and not fitted: 
                    puntos_x = [punto[0] for punto in points_prediction]
                    puntos_y = [punto[1] for punto in points_prediction]
                    a, b, c = np.polyfit(puntos_x, puntos_y, 2)
                    fitted = True
                    if max_found:
                        a_def, b_def, c_def = a, b, c
                        if interseccion_segmento_parabola(a_def, b_def, c_def, (40, 235), (130, 285)):
                            resultado_predict = True
                        else:
                            resultado_predict = False

                if fitted and max_found:
                    for x_alt in range(0, 1000): # max_x
                        y_alt = int(a * (x_alt**2) + b * x_alt + c)
                        
                        if 0 <= y_alt < 720:
                            cv2.circle(frame, (x_alt, y_alt), 2, (0, 255, 255), -1)

                if (x < 130 or y > 300) and len(points_real) > 5 and not fitted_real: # 130
                    puntos_real_x = [punto[0] for punto in points_real]
                    puntos_real_y = [punto[1] for punto in points_real]
                    a_real, b_real, c_real = np.polyfit(puntos_real_x, puntos_real_y, 2)
                    fitted_real = True

                    if interseccion_segmento_parabola(a_real, b_real, c_real, (40, 235), (130, 285)):
                        resultado_real = True
                        aciertos += 1
                    else:
                        resultado_real = False
                        fallos += 1

                    if resultado_real == resultado_predict:
                        pred_correcta += 1
                    else:
                        pred_error += 1

                if fitted_real:
                    for x_real in range(0, 900):
                        y_real = int(a_real * (x_real**2) + b_real * x_real + c_real)
                        
                        if 0 <= y_real < 720:
                            cv2.circle(frame, (x_real, y_real), 2, (255, 255, 0), -1)

                for point in points_real:
                    cv2.circle(frame, (point[0], point[1]), 10, (255, 255, 0), -1)
                    
                    # Dibujar lo que ve la cámara (Punto ROJO - Crudo)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        else:
            points_prediction = []
            points_real = []
            fitted = False
            fitted_real = False
            max_found = False

        pred_x, pred_y = tracker.predict()
        
        if detectado:
            tracker.update(cx, cy)
            punto_final = (pred_x, pred_y) 
        else:
            punto_final = (pred_x, pred_y)
            cv2.putText(frame, "Prediccion (Sin Vision)", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            points_prediction = []
            points_real = []
            fitted = False
            fitted_real = False
            max_found = False

        cv2.circle(frame, punto_final, 7, (0, 255, 0), 2)
        
        cv2.rectangle(frame, (pred_x-30, pred_y-30), (pred_x+30, pred_y+30), (255, 255, 0), 1)

        cv2.rectangle(frame, (0,0), (1200, 360), (0, 0, 255), 2)

        cv2.rectangle(frame, (0,0), (130, 300), (0, 255, 0), 2) 

        cv2.putText(frame, f"Aciertos: {aciertos}", (750, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"Fallos: {fallos}", (750, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(frame, f"Predicciones bien: {pred_correcta}", (750, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"Predicciones mal: {pred_error}", (750, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.line(frame, (40, 235), (130, 285), (0, 0, 255), 2, cv2.LINE_AA)

        tiempo_actual = time.time()

        duracion = tiempo_actual - tiempo_anterior
        if duracion > 0:
            fps = 1 / duracion
        else:
            fps = 0
            
        tiempo_anterior = tiempo_actual

        fps_texto = f"FPS: {int(fps)}"

        cv2.putText(frame, fps_texto, (750, 250), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.5, (0, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow("Baloncesto con Kalman", frame)
        
        if cv2.waitKey(30) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()