import cv2
import mediapipe as mp
import numpy as np
import pygame
import os
import time
import subprocess

# Inicializar pygame para sonidos (2 canales estéreo)
pygame.mixer.init(frequency=44100, size=-16, channels=2)

def scape():
     print("Saliendo del pong retro...")
     cv2.destroyAllWindows()
     subprocess.Popen(["python", "menu.py"])

def load_sound(filename):
    if not os.path.exists(filename):
        return None
    sound = pygame.mixer.Sound(filename)
    sound.set_volume(1.0)  # Volumen máximo por defecto
    return sound

pong_sound = load_sound("pong.mpeg")
win_sound = load_sound("win.mp3")
countdown_sound = load_sound("countdown.mp3")

# Configuración del tiempo de espera
COUNT_DOWN = 3

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

# Velocidad original de la bola
MIN_SPEED = 5

# Máxima velocidad de la bola
MAX_SPEED = 30

# Velocidad de la bola
ballSpeedX = MIN_SPEED
ballSpeedY = MIN_SPEED

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
    try:
        x, y = int(ballPosition[0]), int(ballPosition[1])
        cv2.circle(frame, (x, y), BALL_SIZE, (255, 255, 255), -1)
    except Exception as e:
        print(f"Error dibujando bola: {e}")
        print(f"ballPosition: {ballPosition}, tipo: {type(ballPosition[0])}, {type(ballPosition[1])}")

def draw_middle_line(frame):
    for y in range(0, HEIGHT, 20):
        if (y // 10) % 2 == 0:
            cv2.line(frame, (WIDTH // 2, y), (WIDTH // 2, y + 10), (255, 255, 255), 2)

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

def draw_shadow(frame):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (WIDTH, HEIGHT), (0, 0, 0), -1)
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def draw_stop(frame, hand_data):
    global ballSpeedX, ballSpeedY
    if not hasattr(draw_stop, "prev_players"):
        draw_stop.prev_players = 0
        draw_stop.countdown_active = False
        draw_stop.saved_speed = None  # Para guardar la velocidad original
    
    num_players = len(hand_data)

    while num_players < 2:
        if not draw_stop.countdown_active:
            draw_shadow(frame)
        
        # Guardar la velocidad actual si no lo hemos hecho ya
        if draw_stop.saved_speed is None:
            draw_stop.saved_speed = (ballSpeedX, ballSpeedY)
        
        ballSpeedX = 0
        ballSpeedY = 0
    
        message = f"Jugadores detectados: {num_players}/2"
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (WIDTH - text_size[0]) // 2
        text_y = (HEIGHT + text_size[1]) // 2
    
        cv2.putText(frame, message, (text_x, text_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        draw_stop.prev_players = num_players
        draw_stop.countdown_active = False
        break

    else:
        if draw_stop.prev_players < 2 and num_players == 2:
            draw_stop.countdown_active = True
            
            original_frame = frame.copy()
            
            for i in range(COUNT_DOWN, 0, -1):
                frame[:] = original_frame[:]
                
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (WIDTH, HEIGHT), (0, 0, 0), -1)
                alpha = 0.7
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                
                count_text = str(i)
                text_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 3, 5)[0]
                text_x = (WIDTH - text_size[0]) // 2
                text_y = (HEIGHT + text_size[1]) // 2
                
                cv2.putText(frame, count_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
                
                cv2.imshow("Pong AR - Turn-Based", frame)
                if countdown_sound:
                    countdown_sound.play()
                cv2.waitKey(1000)
            
            draw_stop.countdown_active = False
            # Restaurar la velocidad guardada o usar la velocidad por defecto
            if draw_stop.saved_speed is not None:
                ballSpeedX, ballSpeedY = draw_stop.saved_speed
                draw_stop.saved_speed = None
            else:
                ballSpeedX = -MIN_SPEED if last_touched == 2 else MIN_SPEED
                ballSpeedY = -MIN_SPEED
        
        draw_stop.prev_players = num_players

def speed_up():
    global ballSpeedX, ballSpeedY
    
    SPEED_INCREASE_FACTOR = 1.1
    
    ballSpeedX = abs(ballSpeedX) * SPEED_INCREASE_FACTOR * (1 if ballSpeedX > 0 else -1)
    ballSpeedY = abs(ballSpeedY) * SPEED_INCREASE_FACTOR * (1 if ballSpeedY > 0 else -1)
    
    if abs(ballSpeedX) > MAX_SPEED:
        ballSpeedX = MAX_SPEED * (1 if ballSpeedX > 0 else -1)
    
    if abs(ballSpeedY) > MAX_SPEED:
        ballSpeedY = MAX_SPEED * (1 if ballSpeedY > 0 else -1)

def update_ball_position(hand_data):
    global ballPosition, ballSpeedX, ballSpeedY, left_score, right_score, last_touched

    next_x = ballPosition[0] + ballSpeedX
    next_y = ballPosition[1] + ballSpeedY

    if next_y <= 0 or next_y >= HEIGHT:
        ballSpeedY = -ballSpeedY

    if (POS_HORIZONTAL_IZQUIERDA - RECTANGULO_WIDTH // 2 <= next_x <= POS_HORIZONTAL_IZQUIERDA + RECTANGULO_WIDTH // 2 and 
        left_paddle_y <= next_y <= left_paddle_y + RECTANGULO_HEIGHT):
        if last_touched != 1:
            ballSpeedX = -ballSpeedX
            last_touched = 1
            speed_up()
            if pong_sound:
                channel = pygame.mixer.Channel(0)
                channel.set_volume(1.0, 0.0)
                channel.play(pong_sound)

    elif (POS_HORIZONTAL_DERECHA - RECTANGULO_WIDTH // 2 <= next_x <= POS_HORIZONTAL_DERECHA + RECTANGULO_WIDTH // 2 and 
          right_paddle_y <= next_y <= right_paddle_y + RECTANGULO_HEIGHT):
        if last_touched != 2:
            ballSpeedX = -ballSpeedX
            last_touched = 2
            speed_up()
            if pong_sound:
                channel = pygame.mixer.Channel(1)
                channel.set_volume(0.0, 1.0)
                channel.play(pong_sound)

    ballPosition[0] = next_x
    ballPosition[1] = next_y

    if ballPosition[0] <= 0:
        right_score += 1
        last_touched = None
        ballSpeedX = MIN_SPEED
        ballSpeedY = MIN_SPEED
        if win_sound:
            # Usamos el canal 2 para el sonido de victoria (derecho)
            channel = pygame.mixer.Channel(2)
            channel.set_volume(0.0, 1.0)  # Solo derecho
            channel.play(win_sound)
        ballPosition = [WIDTH // 2, HEIGHT // 2]
    elif ballPosition[0] >= WIDTH:
        left_score += 1
        last_touched = None
        ballSpeedX = MIN_SPEED
        ballSpeedY = MIN_SPEED
        if win_sound:
            # Usamos el canal 3 para el sonido de victoria (izquierdo)
            channel = pygame.mixer.Channel(3)
            channel.set_volume(1.0, 0.0)  # Solo izquierdo
            channel.play(win_sound)
        ballPosition = [WIDTH // 2, HEIGHT // 2]

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
    
    draw_shadow(frame)
    draw_ball(frame)
    draw_middle_line(frame)
    draw_score(frame)
    draw_paddles(frame)
    update_ball_position(hand_data)
    draw_stop(frame, hand_data)
    
    cv2.imshow("Pong AR - Turn-Based", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        scape()
        break

cap.release()
cv2.destroyAllWindows()
