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

# Configuración de la ventana
WIDTH = 640
HEIGHT = 480

# Tamaño de la bola
BALL_SIZE = 10

# Configuración de velocidad de la bola
INITIAL_BALL_SPEED = 5
MAX_BALL_SPEED = 15
HAND_SPEED_MULTIPLIER = 0.5

# Posición inicial de la bola (asegurando que sean enteros)
ballPosition = [int(WIDTH // 2), int(HEIGHT // 2)]
ballSpeedX = INITIAL_BALL_SPEED
ballSpeedY = INITIAL_BALL_SPEED

# Puntuación
left_score = 0
right_score = 0
last_touched = None

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils

# Variables para el seguimiento
last_detected_time = [0, 0]
warning_shown = [False, False]
game_active = False
game_paused = False

# Variables para velocidad de manos
hand_positions = [[None, None], [None, None]]
hand_speeds = [[0, 0], [0, 0]]

# Tiempos en segundos
WARNING_TIME = 1.0
PAUSE_TIME = 3.0

def get_hand_rect(hand_landmarks):
    x_coords = [lm.x * WIDTH for lm in hand_landmarks.landmark]
    y_coords = [lm.y * HEIGHT for lm in hand_landmarks.landmark]
    min_x, max_x = int(min(x_coords)), int(max(x_coords))
    min_y, max_y = int(min(y_coords)), int(max(y_coords))
    return (min_x, min_y), (max_x, max_y)

def calculate_hand_speed(hand_index, current_pos):
    global hand_positions, hand_speeds

    if hand_positions[hand_index][0] is not None:
        prev_pos, prev_time = hand_positions[hand_index]
        time_elapsed = time.time() - prev_time

        if time_elapsed > 0:
            current_center = ((current_pos[0][0] + current_pos[1][0]) / 2, (current_pos[0][1] + current_pos[1][1]) / 2)
            prev_center = ((prev_pos[0][0] + prev_pos[1][0]) / 2, (prev_pos[0][1] + prev_pos[1][1]) / 2)

            dx = current_center[0] - prev_center[0]
            dy = current_center[1] - prev_center[1]

            hand_speeds[hand_index][0] = dx / time_elapsed
            hand_speeds[hand_index][1] = dy / time_elapsed

    hand_positions[hand_index] = (current_pos, time.time())

def process_hands(results):
    global last_detected_time, warning_shown, game_active, game_paused

    hand_data = []
    current_time = time.time()
    hands_detected = [False, False]

    if results.multi_hand_landmarks:
        hands_sorted = sorted(
            zip(results.multi_hand_landmarks, results.multi_handedness),
            key=lambda h: h[0].landmark[mp_hands.HandLandmark.WRIST].x
        )

        for landmarks, handedness in hands_sorted:
            label = 0 if handedness.classification[0].label == "Left" else 1
            rect = get_hand_rect(landmarks)
            hand_data.append((label + 1, rect))

            calculate_hand_speed(label, rect)

            hands_detected[label] = True
            last_detected_time[label] = current_time

            if warning_shown[label]:
                warning_shown[label] = False

    for i in range(2):
        time_since_last_detection = current_time - last_detected_time[i]

        if not hands_detected[i] and last_detected_time[i] > 0:
            if time_since_last_detection > WARNING_TIME and not warning_shown[i]:
                warning_shown[i] = True

            if time_since_last_detection > PAUSE_TIME:
                game_paused = True

    if all(hands_detected) and (not game_active or game_paused):
        game_active = True
        game_paused = False
        for i in range(2):
            warning_shown[i] = False

    return hand_data

def draw_ball(frame):
    # Asegurar que las coordenadas sean enteros válidos
    center = (int(round(ballPosition[0])), int(round(ballPosition[1])))
    cv2.circle(frame, center, BALL_SIZE, (255, 255, 255), -1)

def draw_middle_line(frame):
    for y in range(0, HEIGHT, 20):
        if (y // 10) % 2 == 0:
            cv2.line(frame, (WIDTH // 2, y), (WIDTH // 2, y + 10), (255, 255, 255), 2)

def update_ball_position(hand_data):
    global ballPosition, ballSpeedX, ballSpeedY, left_score, right_score, last_touched

    if game_paused or not game_active:
        return

    # Asegurar que la posición sea numérica antes de calcular
    next_x = float(ballPosition[0]) + ballSpeedX
    next_y = float(ballPosition[1]) + ballSpeedY

    # Rebotar en bordes
    if next_y <= 0 or next_y >= HEIGHT:
        ballSpeedY = -ballSpeedY

    # Colisiones con manos
    collision = False
    for label, ((min_x, min_y), (max_x, max_y)) in hand_data:
        center_x = (min_x + max_x) // 2
        if ((center_x-10) <= next_x <= (center_x+10)) and (min_y <= next_y <= max_y):
            if last_touched != label:
                hand_index = label - 1

                new_speed_x = -ballSpeedX + (hand_speeds[hand_index][0] * HAND_SPEED_MULTIPLIER)
                new_speed_y = ballSpeedY + (hand_speeds[hand_index][1] * HAND_SPEED_MULTIPLIER)

                speed_magnitude = np.sqrt(new_speed_x**2 + new_speed_y**2)
                if speed_magnitude > MAX_BALL_SPEED:
                    scale_factor = MAX_BALL_SPEED / speed_magnitude
                    new_speed_x *= scale_factor
                    new_speed_y *= scale_factor

                ballSpeedX = new_speed_x
                ballSpeedY = new_speed_y

                last_touched = label
                collision = True
                if pong_sound:
                    pong_sound.play()
                break

    if not collision:
        # Asegurar que las posiciones sean números válidos
        ballPosition[0] = int(round(next_x))
        ballPosition[1] = int(round(next_y))

    # Puntos
    if ballPosition[0] <= 0:
        right_score += 1
        last_touched = None
        if win_sound:
            win_sound.play()
        reset_ball()
    elif ballPosition[0] >= WIDTH:
        left_score += 1
        last_touched = None
        if win_sound:
            win_sound.play()
        reset_ball()

def reset_ball():
    global ballPosition, ballSpeedX, ballSpeedY
    ballPosition = [int(WIDTH // 2), int(HEIGHT // 2)]
    ballSpeedX = INITIAL_BALL_SPEED if np.random.rand() > 0.5 else -INITIAL_BALL_SPEED
    ballSpeedY = INITIAL_BALL_SPEED

def draw_score(frame):
    cv2.putText(frame, f"Left: {left_score}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Right: {right_score}", (WIDTH - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def draw_warnings(frame):
    for i in range(2):
        if warning_shown[i]:
            hand_name = "Left" if i == 0 else "Right"
            cv2.putText(frame, f"{hand_name} hand not detected!",
                        (20, 100 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

def draw_game_status(frame):
    if not game_active:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (WIDTH, HEIGHT), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, "Waiting for 2 players...",
                    (WIDTH//4, HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    elif game_paused:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (WIDTH, HEIGHT), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, "Game Paused - Show both hands to resume",
                    (WIDTH//6, HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

def draw_debug_info(frame):
    # Mostrar información de debug en la esquina inferior izquierda
    debug_texts = [
        f"Ball Position: ({ballPosition[0]}, {ballPosition[1]})",
        f"Ball Speed: ({ballSpeedX:.1f}, {ballSpeedY:.1f})",
        f"Left Hand Speed: ({hand_speeds[0][0]:.1f}, {hand_speeds[0][1]:.1f})",
        f"Right Hand Speed: ({hand_speeds[1][0]:.1f}, {hand_speeds[1][1]:.1f})",
        f"Game State: {'Active' if game_active else 'Inactive'} {'(Paused)' if game_paused else ''}",
        f"Last Touched: {'Left' if last_touched == 1 else 'Right' if last_touched == 2 else 'None'}"
    ]

    for i, text in enumerate(debug_texts):
        cv2.putText(frame, text, (10, HEIGHT - 30 - (i * 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

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

    draw_ball(frame)
    draw_middle_line(frame)
    draw_score(frame)
    update_ball_position(hand_data)
    draw_warnings(frame)
    draw_game_status(frame)
    draw_debug_info(frame)  # <-- Método de debug reimplementado

    if game_active or not game_paused:
        for label, ((min_x, min_y), (max_x, max_y)) in hand_data:
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
            cv2.putText(frame, str(label), (min_x + 5, min_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Pong AR - Turn-Based", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
