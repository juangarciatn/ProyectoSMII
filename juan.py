import cv2
import mediapipe as mp
import numpy as np
import pygame
import os
import time

# Inicializar pygame para sonidos
pygame.mixer.init()

def load_sound(filename):
    return pygame.mixer.Sound(filename) if os.path.exists(filename) else None

pong_sound = load_sound("pong.mpeg")
win_sound = load_sound("win.mp3")
countdown_sound = load_sound("countdown.mp3")

# Tiempo de countdown
COUNTDOWN = 3
countdown_active = False
countdown_start_time = 0
current_countdown = COUNTDOWN
game_started = False
waiting_for_players = True  # Nueva variable para controlar el estado inicial

# Configuración de la posición en el eje coordinal
POS_HORIZONTAL_IZQUIERDA = 30
POS_HORIZONTAL_DERECHA = 610

# Configuración de los rectángulos
RECTANGULO_WIDTH = 30
RECTANGULO_HEIGHT = 90

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
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils

def get_hand_rect(hand_landmarks):
    x_coords = [lm.x * WIDTH for lm in hand_landmarks.landmark]
    y_coords = [lm.y * HEIGHT for lm in hand_landmarks.landmark]
    min_x, max_x = int(min(x_coords)), int(max(x_coords))
    min_y, max_y = int(min(y_coords)), int(max(y_coords))
    return (min_x, min_y), (max_x, max_y)

# Variables globales para la posición de los rectángulos
left_paddle_y = HEIGHT // 2 - RECTANGULO_HEIGHT // 2
right_paddle_y = HEIGHT // 2 - RECTANGULO_HEIGHT // 2

def process_hands(results):
    global left_paddle_y, right_paddle_y
    
    hand_data = []

    if results.multi_hand_landmarks:
        hands_sorted = sorted(
            zip(results.multi_hand_landmarks, results.multi_handedness),
            key=lambda h: h[0].landmark[mp_hands.HandLandmark.WRIST].x
        )

        for idx, (landmarks, handedness) in enumerate(hands_sorted, 1):
            label = idx  # 1: izquierda, 2: derecha
            rect = get_hand_rect(landmarks)
            hand_data.append((label, rect))

            # Actualizar la posición del rectángulo
            center_y = (rect[0][1] + rect[1][1]) // 2
            if label == 1:  # Mano izquierda
                left_paddle_y = max(0, min(HEIGHT - RECTANGULO_HEIGHT, center_y - RECTANGULO_HEIGHT // 2))
            elif label == 2:  # Mano derecha
                right_paddle_y = max(0, min(HEIGHT - RECTANGULO_HEIGHT, center_y - RECTANGULO_HEIGHT // 2))
    
    return hand_data

def draw_ball(frame):
    cv2.circle(frame, tuple(ballPosition), BALL_SIZE, (255, 255, 255), -1)

def draw_middle_line(frame):
    for y in range(0, HEIGHT, 20):
        if (y // 10) % 2 == 0:
            cv2.line(frame, (WIDTH // 2, y), (WIDTH // 2, y + 10), (255, 255, 255), 2)

def update_ball_position(hand_data):
    global ballPosition, ballSpeedX, ballSpeedY, left_score, right_score, last_touched, countdown_active, current_countdown, game_started, waiting_for_players

    # No mover la pelota si el juego no está activo
    if not game_started or countdown_active:
        return

    next_x = ballPosition[0] + ballSpeedX
    next_y = ballPosition[1] + ballSpeedY

    # Rebotar en bordes superior/inferior
    if next_y <= 0 or next_y >= HEIGHT:
        ballSpeedY = -ballSpeedY

    # Detectar colisión con los rectángulos
    if (POS_HORIZONTAL_IZQUIERDA - RECTANGULO_WIDTH // 2 <= next_x <= POS_HORIZONTAL_IZQUIERDA + RECTANGULO_WIDTH // 2 and 
        left_paddle_y <= next_y <= left_paddle_y + RECTANGULO_HEIGHT):
        if last_touched != 1:
            ballSpeedX = -ballSpeedX
            last_touched = 1
            if pong_sound:
                pong_sound.play()

    elif (POS_HORIZONTAL_DERECHA - RECTANGULO_WIDTH // 2 <= next_x <= POS_HORIZONTAL_DERECHA + RECTANGULO_WIDTH // 2 and 
          right_paddle_y <= next_y <= right_paddle_y + RECTANGULO_HEIGHT):
        if last_touched != 2:
            ballSpeedX = -ballSpeedX
            last_touched = 2
            if pong_sound:
                pong_sound.play()

    # Actualizar la posición de la bola
    ballPosition[0] = next_x
    ballPosition[1] = next_y

    # Puntos cuando la bola cruza los límites
    if ballPosition[0] <= 0:
        right_score += 1
        last_touched = None
        if win_sound:
            win_sound.play()
        ballPosition = [WIDTH // 2, HEIGHT // 2]
        # Activar cuenta regresiva después de un punto
        countdown_active = True
        countdown_start_time = time.time()
        current_countdown = COUNTDOWN
    elif ballPosition[0] >= WIDTH:
        left_score += 1
        last_touched = None
        if win_sound:
            win_sound.play()
        ballPosition = [WIDTH // 2, HEIGHT // 2]
        # Activar cuenta regresiva después de un punto
        countdown_active = True
        countdown_start_time = time.time()
        current_countdown = COUNTDOWN

def draw_paddles(frame):
    cv2.rectangle(frame, (POS_HORIZONTAL_IZQUIERDA - RECTANGULO_WIDTH // 2, left_paddle_y),
                  (POS_HORIZONTAL_IZQUIERDA + RECTANGULO_WIDTH // 2, left_paddle_y + RECTANGULO_HEIGHT),
                  (255, 255, 255), -1)

    cv2.rectangle(frame, (POS_HORIZONTAL_DERECHA - RECTANGULO_WIDTH // 2, right_paddle_y),
                  (POS_HORIZONTAL_DERECHA + RECTANGULO_WIDTH // 2, right_paddle_y + RECTANGULO_HEIGHT),
                  (255, 255, 255), -1)

def draw_score(frame):
    cv2.putText(frame, f"Left: {left_score}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Right: {right_score}", (WIDTH - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def draw_game_status(frame, hand_data):
    global countdown_active, game_started, waiting_for_players, current_countdown, countdown_start_time
    
    # Mostrar mensaje de espera si no hay suficientes jugadores
    if len(hand_data) < 2 and waiting_for_players:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (WIDTH, HEIGHT), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, f"Waiting for {2 - len(hand_data)} player(s)...", 
                    (WIDTH//4, HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return
    
    # Mostrar countdown si está activo
    if countdown_active:
        elapsed = time.time() - countdown_start_time
        remaining = max(0, COUNTDOWN - int(elapsed))
        
        if remaining != current_countdown:
            if countdown_sound and remaining > 0:
                countdown_sound.play()
            current_countdown = remaining
        
        if remaining > 0:
            text = str(remaining)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 5, 10)[0]
            text_x = (WIDTH - text_size[0]) // 2
            text_y = (HEIGHT + text_size[1]) // 2
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10)
        else:
            text = "GO!"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 3, 10)[0]
            text_x = (WIDTH - text_size[0]) // 2
            text_y = (HEIGHT + text_size[1]) // 2
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 8)
            
            if elapsed > COUNTDOWN + 0.5:
                countdown_active = False
                game_started = True
                waiting_for_players = False

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)
    hand_data = process_hands(results)
    
    # Activar countdown solo al inicio cuando se detectan dos jugadores por primera vez
    if len(hand_data) == 2 and waiting_for_players and not countdown_active:
        countdown_active = True
        countdown_start_time = time.time()
        current_countdown = COUNTDOWN  # Asegurar que empieza desde COUNTDOWN
        if countdown_sound:
            countdown_sound.play()
    
    # Si los jugadores se van durante el juego
    if len(hand_data) < 2 and game_started:
        game_started = False
        waiting_for_players = True
        countdown_active = False
        current_countdown = COUNTDOWN  # Reiniciar el countdown
    
    draw_ball(frame)
    draw_middle_line(frame)
    draw_score(frame)
    draw_paddles(frame)
    update_ball_position(hand_data)    
    draw_game_status(frame, hand_data)
    
    cv2.imshow("Pong AR - Turn-Based", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()