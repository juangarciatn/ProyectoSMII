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

# Función para dibujar las líneas verticales
def draw_vertical_lines(frame):
    cv2.line(frame, (POS_HORIZONTAL_IZQUIERDA, 0), (POS_HORIZONTAL_IZQUIERDA, HEIGHT), (0, 0, 255), 2)
    cv2.line(frame, (POS_HORIZONTAL_DERECHA, 0), (POS_HORIZONTAL_DERECHA, HEIGHT), (0, 0, 255), 2)

# Configuración de la posición en el eje coordinal
POS_HORIZONTAL_IZQUIERDA = 30
POS_HORIZONTAL_DERECHA = 610

# Configuración de los rectángulos
RECTANGULO_WIDTH  = 30
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

# Variable global para almacenar los rectángulos de las manos
hand_rects = []

def get_hand_rect(hand_landmarks):
    x_coords = [lm.x * WIDTH for lm in hand_landmarks.landmark]
    y_coords = [lm.y * HEIGHT for lm in hand_landmarks.landmark]
    min_x, max_x = int(min(x_coords)), int(max(x_coords))
    min_y, max_y = int(min(y_coords)), int(max(y_coords))
    return (min_x, min_y), (max_x, max_y)

# Variables globales para la posición de los rectángulos azules
left_paddle_y = HEIGHT // 2 - RECTANGULO_HEIGHT // 2
right_paddle_y = HEIGHT // 2 - RECTANGULO_HEIGHT // 2

def process_hands(results):
    global left_paddle_y, right_paddle_y  # Necesitamos modificar estas variables globales
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

            # Actualizar la posición del rectángulo en función de la mano detectada
            center_y = (rect[0][1] + rect[1][1]) // 2  # Calcular el centro de la mano
            if label == 1:  # Mano izquierda
                left_paddle_y = max(0, min(HEIGHT - RECTANGULO_HEIGHT, center_y - RECTANGULO_HEIGHT // 2))
            elif label == 2:  # Mano derecha
                right_paddle_y = max(0, min(HEIGHT - RECTANGULO_HEIGHT, center_y - RECTANGULO_HEIGHT // 2))

    return hand_data

def draw_ball(frame):
    cv2.circle(frame, tuple(ballPosition), BALL_SIZE, (255, 255, 255), -1)

def draw_middle_line(frame):
    for y in range(0, HEIGHT, 20):  # Dibujar segmentos de 10 píxeles con espacio de 10 píxeles
        if (y // 10) % 2 == 0:
            cv2.line(frame, (WIDTH // 2, y), (WIDTH // 2, y + 10), (255, 255, 255), 2)

def update_ball_position(hand_data):
    global ballPosition, ballSpeedX, ballSpeedY, left_score, right_score, last_touched

    # No mover la pelota si hay menos de 2 jugadores
    if len(hand_data) < 2:
        return

    next_x = ballPosition[0] + ballSpeedX
    next_y = ballPosition[1] + ballSpeedY

    # Rebotar en bordes superior/inferior
    if next_y <= 0 or next_y >= HEIGHT:
        ballSpeedY = -ballSpeedY

    # Detectar colisión con los rectángulos azules
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
    elif ballPosition[0] >= WIDTH:
        left_score += 1
        last_touched = None
        if win_sound:
            win_sound.play()
        ballPosition = [WIDTH // 2, HEIGHT // 2]

def draw_paddles(frame):
    # Cambiar el color de los rectángulos a blanco (255, 255, 255)
    cv2.rectangle(frame, (POS_HORIZONTAL_IZQUIERDA - RECTANGULO_WIDTH // 2, left_paddle_y),
                  (POS_HORIZONTAL_IZQUIERDA + RECTANGULO_WIDTH // 2, left_paddle_y + RECTANGULO_HEIGHT),
                  (255, 255, 255), -1)  # Blanco

    cv2.rectangle(frame, (POS_HORIZONTAL_DERECHA - RECTANGULO_WIDTH // 2, right_paddle_y),
                  (POS_HORIZONTAL_DERECHA + RECTANGULO_WIDTH // 2, right_paddle_y + RECTANGULO_HEIGHT),
                  (255, 255, 255), -1)  # Blanco

def draw_score(frame):
    cv2.putText(frame, f"Left: {left_score}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Right: {right_score}", (WIDTH - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Crear una segunda ventana con fondo negro
def create_second_window():
    second_window = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    return second_window

# Dibujar elementos en la segunda ventana
def draw_second_window(second_window, hand_data):
    # Dibujar la línea discontinua blanca
    draw_middle_line(second_window)

    # Dibujar los marcadores
    draw_score(second_window)

    # Dibujar los rectángulos blancos
    draw_paddles(second_window)

    # Dibujar la pelota en la segunda ventana
    draw_ball(second_window)

    # Mostrar mensaje si faltan jugadores
    if len(hand_data) < 2:
        overlay = second_window.copy()
        cv2.rectangle(overlay, (0, 0), (WIDTH, HEIGHT), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, second_window, 0.3, 0, second_window)
        cv2.putText(second_window, f"Waiting for {2 - len(hand_data)} player(s)...",
                    (WIDTH//4, HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# Crear la segunda ventana
second_window = create_second_window()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar manos en cada frame
    results = hands.process(rgb_frame)
    hand_data = process_hands(results)

    # Dibujar elementos del juego en la ventana principal
    draw_ball(frame)
    draw_middle_line(frame)
    draw_score(frame)
    draw_paddles(frame)
    update_ball_position(hand_data)
    draw_vertical_lines(frame)

    # Dibujar rectángulos y etiquetas de manos
    for label, ((min_x, min_y), (max_x, max_y)) in hand_data:
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

        # Dibujar la "X" en la posición de la mano
        cv2.line(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)  # Línea diagonal \
        cv2.line(frame, (max_x, min_y), (min_x, max_y), (0, 255, 0), 2)  # Línea diagonal /

        # Calcular el centro de la "X"
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2

        # Determinar la línea horizontal según la mano (izquierda o derecha)
        if label == 1:  # Mano izquierda
            cv2.line(frame, (center_x, center_y), (POS_HORIZONTAL_IZQUIERDA, center_y), (0, 0, 255), 2)
        elif label == 2:  # Mano derecha
            cv2.line(frame, (center_x, center_y), (POS_HORIZONTAL_DERECHA, center_y), (0, 0, 255), 2)

        cv2.putText(frame, str(label), (min_x + 5, min_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Mostrar mensaje si faltan jugadores
    if len(hand_data) < 2:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (WIDTH, HEIGHT), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, f"Waiting for {2 - len(hand_data)} player(s)...",
                    (WIDTH//4, HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Actualizar la segunda ventana
    second_window = create_second_window()
    draw_second_window(second_window, hand_data)

    # Mostrar ambas ventanas
    cv2.imshow("Pong AR - Turn-Based", frame)
    cv2.imshow("Pong AR - Second Window", second_window)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
