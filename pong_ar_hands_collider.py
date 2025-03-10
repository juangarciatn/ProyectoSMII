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

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.9)
mp_drawing = mp.solutions.drawing_utils

# Variable global para almacenar los rectángulos de las manos
hand_rects = []

def get_hand_rect(hand_landmarks):
    x_coords = [lm.x * WIDTH for lm in hand_landmarks.landmark]
    y_coords = [lm.y * HEIGHT for lm in hand_landmarks.landmark]
    min_x, max_x = int(min(x_coords)), int(max(x_coords))
    min_y, max_y = int(min(y_coords)), int(max(y_coords))
    return (min_x, min_y), (max_x, max_y)

def draw_ball(frame):
    cv2.circle(frame, tuple(ballPosition), BALL_SIZE, (255, 255, 255), -1)

def update_ball_position(hand_rects):
    global ballPosition, ballSpeedX, ballSpeedY, left_score, right_score

    ballPosition[0] += ballSpeedX
    ballPosition[1] += ballSpeedY

    # Rebotar en los bordes superior e inferior
    if ballPosition[1] <= 0 or ballPosition[1] >= HEIGHT:
        ballSpeedY = -ballSpeedY

    # Rebotar en las manos
    for rect in hand_rects:
        (min_x, min_y), (max_x, max_y) = rect
        if (min_x <= ballPosition[0] <= max_x) and (min_y <= ballPosition[1] <= max_y):
            ballSpeedX = -ballSpeedX
            if pong_sound:
                pong_sound.play()
            break

    # Asignar puntos según el borde alcanzado
    if ballPosition[0] <= 0:
        right_score += 1  # Punto para la derecha
        if win_sound:
            win_sound.play()
        ballPosition = [WIDTH // 2, HEIGHT // 2]  # Reiniciar la bola
    elif ballPosition[0] >= WIDTH:
        left_score += 1  # Punto para la izquierda
        if win_sound:
            win_sound.play()
        ballPosition = [WIDTH // 2, HEIGHT // 2]

def draw_score(frame):
    cv2.putText(frame, f"Left: {left_score}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Right: {right_score}", (WIDTH - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

frame_counter = 0
detection_interval = 2

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if frame_counter % detection_interval == 0:
        results = hands.process(rgb_frame)
        hand_rects.clear()
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                rect = get_hand_rect(hand_landmarks)
                hand_rects.append(rect)

    for rect in hand_rects:
        cv2.rectangle(frame, rect[0], rect[1], (0, 255, 0), 2)

    frame_counter += 1
    draw_ball(frame)
    update_ball_position(hand_rects)
    draw_score(frame)
    cv2.imshow("Pong AR", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
