import cv2
import mediapipe as mp
import numpy as np
import pygame
import os

# Inicializar pygame para sonidos
pygame.mixer.init()

def load_sound(filename):
    return pygame.mixer.Sound(filename) if os.path.exists(filename) else None

pong_sound = load_sound("pong.mpeg")
win_sound = load_sound("win.mp3")

# Configuración de la ventana
WIDTH = 640
HEIGHT = 480

# Tamaño de la bola
BALL_SIZE = 10

# Velocidad de la bola
ballSpeedX = 5
ballSpeedY = 5

# Posición inicial de la bola
ballPosition = [WIDTH // 2, HEIGHT // 2]

# Puntuación
left_score = 0
right_score = 0
last_touched = None  # 1: izquierda, 2: derecha

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.1)
=======
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.3)
>>>>>>> Stashed changes
=======
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.3)
>>>>>>> Stashed changes
=======
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.3)
>>>>>>> Stashed changes
mp_drawing = mp.solutions.drawing_utils

# Variable global para almacenar los rectángulos de las manos
hand_rects = []

def get_hand_rect(hand_landmarks):
    x_coords = [lm.x * WIDTH for lm in hand_landmarks.landmark]
    y_coords = [lm.y * HEIGHT for lm in hand_landmarks.landmark]
    min_x, max_x = int(min(x_coords)), int(max(x_coords))
    min_y, max_y = int(min(y_coords)), int(max(y_coords))
    return (min_x, min_y), (max_x, max_y)

def process_hands(results):
    hand_data = []
    if results.multi_hand_landmarks:
        # Ordenar manos por posición X (izquierda a derecha)
        hands_sorted = sorted(
            zip(results.multi_hand_landmarks, results.multi_handedness),
            key=lambda h: h[0].landmark[mp_hands.HandLandmark.WRIST].x
        )
        
        for idx, (landmarks, handedness) in enumerate(hands_sorted, 1):
            label = idx  # 1: izquierda, 2: derecha
            rect = get_hand_rect(landmarks)
            hand_data.append((label, rect))
    return hand_data

def draw_ball(frame):
    cv2.circle(frame, tuple(ballPosition), BALL_SIZE, (255, 255, 255), -1)

def draw_middle_line(frame):
    for y in range(0, HEIGHT, 20):  # Dibujar segmentos de 10 píxeles con espacio de 10 píxeles
        if (y // 10) % 2 == 0:
            cv2.line(frame, (WIDTH // 2, y), (WIDTH // 2, y + 10), (255, 255, 255), 2)

def update_ball_position(hand_rects):
    global ballPosition, ballSpeedX, ballSpeedY, left_score, right_score, last_touched

    next_x = ballPosition[0] + ballSpeedX
    next_y = ballPosition[1] + ballSpeedY

    # Rebotar en bordes superior/inferior
    if next_y <= 0 or next_y >= HEIGHT:
        ballSpeedY = -ballSpeedY

    if len(hand_rects) < 2:
        return  # No mover la pelota si hay menos de 2 jugadores

    # Verificar colisiones con manos
    collision = False
    for label, ((min_x, min_y), (max_x, max_y)) in hand_data:
        if (min_x <= next_x <= max_x) and (min_y <= next_y <= max_y):
            if last_touched != label:
                ballSpeedX = -ballSpeedX
                last_touched = label
                collision = True
                if pong_sound:
                    pong_sound.play()
                break

    # Actualizar posición solo si no hay colisión no válida
    if not collision:
        ballPosition[0] = next_x
        ballPosition[1] = next_y

    # Asignar puntos según el borde alcanzado
    if ballPosition[0] <= 0:
        right_score += 1  # Punto para la derecha
        last_touched = None
        if win_sound:
            win_sound.play()
        ballPosition = [WIDTH // 2, HEIGHT // 2]  # Reiniciar la bola
    elif ballPosition[0] >= WIDTH:
        left_score += 1  # Punto para la izquierda
        last_touched = None
        if win_sound:
            win_sound.play()
        ballPosition = [WIDTH // 2, HEIGHT // 2]

def draw_score(frame):
    cv2.putText(frame, f"Left: {left_score}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Right: {right_score}", (WIDTH - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Procesar manos en cada frame
    results = hands.process(rgb_frame)
    hand_data = process_hands(results)
    
    # Dibujar elementos del juego
    draw_ball(frame)
    draw_middle_line(frame)
    draw_score(frame)
    update_ball_position(hand_data)
    
    # Dibujar rectángulos y etiquetas de manos
    for label, ((min_x, min_y), (max_x, max_y)) in hand_data:
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
        cv2.putText(frame, str(label), (min_x + 5, min_y + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Mostrar mensaje si faltan jugadores
    if len(hand_data) < 2:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (WIDTH, HEIGHT), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, f"Waiting for {2 - len(hand_data)} player(s)...", 
                    (WIDTH//4, HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Pong AR - Turn-Based", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()