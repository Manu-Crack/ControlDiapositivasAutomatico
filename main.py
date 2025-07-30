import cv2
import mediapipe as mp
import pyautogui
import time

# Inicializa MediaPipe
mp_hands = mp.solutions.hands
mp_dibujo = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1)
tiempo_ultimo = 0
tiempo_espera = 1.0
#tamaño de la zona activa
# ZONA_ANCHO y ZONA_ALTO definen el tamaño de la zona activa en la parte inferior de la pantalla
ZONA_ANCHO = 165
ZONA_ALTO = 124

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as manos:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        altura, ancho, _ = frame.shape

        zona_x1 = 0
        zona_y1 = altura - ZONA_ALTO
        zona_x2 = ZONA_ANCHO
        zona_y2 = altura

        imagen_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = manos.process(imagen_rgb)
        frame = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2BGR)

        # Dibuja zona activa
        cv2.rectangle(frame, (zona_x1, zona_y1), (zona_x2, zona_y2), (0, 0, 255), 2)

        if resultado.multi_hand_landmarks:
            for hand_landmarks in resultado.multi_hand_landmarks:
                # Dibuja todos los puntos y conexiones de la mano
                mp_dibujo.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_dibujo.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_dibujo.DrawingSpec(color=(255, 255, 255), thickness=2))

                # Punto índice (landmark 8)
                punto_indice = hand_landmarks.landmark[8]
                x = int(punto_indice.x * ancho)
                y = int(punto_indice.y * altura)

                # Dibuja círculo más grande en la punta del índice
                cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)

                # Verifica si está dentro de la zona activa
                if zona_x1 <= x <= zona_x2 and zona_y1 <= y <= zona_y2:
                    rel_x = x - zona_x1
                    rel_y = y - zona_y1

                    tiempo_actual = time.time()
                    if tiempo_actual - tiempo_ultimo > tiempo_espera:
                        if rel_y <= 10:
                            pyautogui.press('f5')
                            tiempo_ultimo = tiempo_actual
                            cv2.putText(frame, "Iniciar", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                        elif rel_y >= ZONA_ALTO - 10:
                            pyautogui.press('esc')
                            tiempo_ultimo = tiempo_actual
                            cv2.putText(frame, "Salir", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        elif rel_x <= 10:
                            pyautogui.press('left')
                            tiempo_ultimo = tiempo_actual
                            cv2.putText(frame, "Anterior", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        elif rel_x >= ZONA_ANCHO - 10:
                            pyautogui.press('right')
                            tiempo_ultimo = tiempo_actual
                            cv2.putText(frame, "Siguiente", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Detección completa de ambas manos", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()