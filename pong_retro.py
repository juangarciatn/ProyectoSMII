import cv2
import mediapipe as mp
import numpy as np
import pygame
import os
import time
import subprocess
import random

# Inicializar pygame para sonidos
pygame.mixer.init(frequency=44100, size=-16, channels=2)

# Constantes del juego
COUNT_DOWN_LEFT_HAND = 2
COUNT_DOWN_RIGHT_HAND = 2
POS_HORIZONTAL_IZQUIERDA = 30
POS_HORIZONTAL_DERECHA = 610
RECTANGULO_WIDTH = 30
RECTANGULO_HEIGHT = 90
WIDTH = 640
HEIGHT = 480
BALL_SIZE = 10
MIN_SPEED = 5
MAX_SPEED = 30

def scape():
    print("Saliendo del pong retro...")
    cv2.destroyAllWindows()
    subprocess.Popen(["python", "menu.py"])

def load_sound(filename):
    if not os.path.exists(filename):
        return None
    sound = pygame.mixer.Sound(filename)
    sound.set_volume(1.0)
    return sound

def reset_ball(direction):
    global ballPosition, ballSpeedX, ballSpeedY, last_touched
    
    ballPosition = [WIDTH // 2, HEIGHT // 2]
    ballSpeedX = MIN_SPEED * direction
    ballSpeedY = MIN_SPEED * random.choice([-1, 1])
    last_touched = None
    time.sleep(1)

def initialize_game():
    global pong_sound, win_sound, countdown_sound
    global left_paddle_y, right_paddle_y
    global ballPosition, ballSpeedX, ballSpeedY
    global left_score, right_score, last_touched
    global hands, mp_hands, mp_drawing
    
    pong_sound = load_sound("pong.mpeg")
    win_sound = load_sound("win.mp3")
    countdown_sound = load_sound("countdown.mp3")
    
    left_paddle_y = HEIGHT // 2 - RECTANGULO_HEIGHT // 2
    right_paddle_y = HEIGHT // 2 - RECTANGULO_HEIGHT // 2
    
    reset_ball(direction=random.choice([-1, 1]))
    
    left_score = 0
    right_score = 0
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.3)
    mp_drawing = mp.solutions.drawing_utils

def get_hand_rect(hand_landmarks):
    x_coords = [lm.x * WIDTH for lm in hand_landmarks.landmark]
    y_coords = [lm.y * HEIGHT for lm in hand_landmarks.landmark]
    min_x, max_x = int(min(x_coords)), int(max(x_coords))
    min_y, max_y = int(min(y_coords)), int(max(y_coords))
    return (min_x, min_y), (max_x, max_y)

def process_hands(results):
    global left_paddle_y, right_paddle_y
    
    hand_data = []
    left_hand_detected = False
    right_hand_detected = False

    if results.multi_hand_landmarks:
        for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label  # 'Left' o 'Right'
            rect = get_hand_rect(landmarks)
            hand_data.append((hand_label, rect))

            center_y = (rect[0][1] + rect[1][1]) // 2
            if hand_label == 'Left':
                left_hand_detected = True
                left_paddle_y = max(0, min(HEIGHT - RECTANGULO_HEIGHT, center_y - RECTANGULO_HEIGHT // 2))
            elif hand_label == 'Right':
                right_hand_detected = True
                right_paddle_y = max(0, min(HEIGHT - RECTANGULO_HEIGHT, center_y - RECTANGULO_HEIGHT // 2))
    
    return hand_data, left_hand_detected, right_hand_detected

def draw_ball(frame):
    try:
        x, y = int(ballPosition[0]), int(ballPosition[1])
        cv2.circle(frame, (x, y), BALL_SIZE, (255, 255, 255), -1)
    except Exception as e:
        print(f"Error dibujando bola: {e}")

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

def draw_stop(frame, hand_data, left_hand_detected, right_hand_detected):
    global COUNT_DOWN_LEFT_HAND, COUNT_DOWN_RIGHT_HAND
    
    # Contador para mano izquierda
    if not left_hand_detected:
        text = "Left hand not detected!"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = (WIDTH // 4) - (text_size[0] // 2)  # Centrado en el lado izquierdo
        text_y = HEIGHT // 4  # Más arriba (1/4 de la pantalla)
        cv2.putText(frame, text, 
                   (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        COUNT_DOWN_LEFT_HAND -= 1/30
    else:
        COUNT_DOWN_LEFT_HAND = 2
    
    # Contador para mano derecha
    if not right_hand_detected:
        text = "Right hand not detected!"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = (3 * WIDTH // 4) - (text_size[0] // 2)  # Centrado en el lado derecho
        text_y = HEIGHT // 4  # Más arriba (1/4 de la pantalla)
        cv2.putText(frame, text, 
                   (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        COUNT_DOWN_RIGHT_HAND -= 1/30
    else:
        COUNT_DOWN_RIGHT_HAND = 2
    
    # Mensaje de pausa
    if COUNT_DOWN_LEFT_HAND <= 0 or COUNT_DOWN_RIGHT_HAND <= 0:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (WIDTH, HEIGHT), (0, 0, 0), -1)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        pause_text = "JUEGO EN PAUSA"
        pause_size = cv2.getTextSize(pause_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)[0]
        cv2.putText(frame, pause_text, 
                   (WIDTH // 2 - pause_size[0] // 2, HEIGHT // 2 - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        continue_text = "Muestre ambas manos para continuar"
        continue_size = cv2.getTextSize(continue_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.putText(frame, continue_text, 
                   (WIDTH // 2 - continue_size[0] // 2, HEIGHT // 2 + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return True
    
    return False

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

    if (POS_HORIZONTAL_IZQUIERDA - RECTANGULO_WIDTH//2 <= next_x <= POS_HORIZONTAL_IZQUIERDA + RECTANGULO_WIDTH//2 and 
        left_paddle_y <= next_y <= left_paddle_y + RECTANGULO_HEIGHT):
        if last_touched != 'Left':
            ballSpeedX = -ballSpeedX
            last_touched = 'Left'
            speed_up()
            if pong_sound:
                channel = pygame.mixer.Channel(0)
                channel.set_volume(1.0, 0.0)
                channel.play(pong_sound)

    elif (POS_HORIZONTAL_DERECHA - RECTANGULO_WIDTH//2 <= next_x <= POS_HORIZONTAL_DERECHA + RECTANGULO_WIDTH//2 and 
          right_paddle_y <= next_y <= right_paddle_y + RECTANGULO_HEIGHT):
        if last_touched != 'Right':
            ballSpeedX = -ballSpeedX
            last_touched = 'Right'
            speed_up()
            if pong_sound:
                channel = pygame.mixer.Channel(1)
                channel.set_volume(0.0, 1.0)
                channel.play(pong_sound)

    ballPosition[0] = next_x
    ballPosition[1] = next_y

    # Reinicio cuando la pelota sale por izquierda (punto para derecha)
    if ballPosition[0] <= 0:
        right_score += 1
        reset_ball(direction=-1)  # Ahora va hacia el perdedor (izquierda)
        if win_sound:
            channel = pygame.mixer.Channel(2)
            channel.set_volume(0.0, 1.0)
            channel.play(win_sound)
    
    # Reinicio cuando la pelota sale por derecha (punto para izquierda)
    elif ballPosition[0] >= WIDTH:
        left_score += 1
        reset_ball(direction=1)  # Ahora va hacia el perdedor (derecha)
        if win_sound:
            channel = pygame.mixer.Channel(3)
            channel.set_volume(1.0, 0.0)
            channel.play(win_sound)

def main():
    initialize_game()
    
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
        hand_data, left_detected, right_detected = process_hands(results)
        
        draw_shadow(frame)
        draw_ball(frame)
        draw_middle_line(frame)
        draw_score(frame)
        draw_paddles(frame)
        
        # Solo actualizar posición si no estamos en pausa
        paused = draw_stop(frame, hand_data, left_detected, right_detected)
        if not paused:
            update_ball_position(hand_data)
        
        cv2.imshow("Pong AR - Turn-Based", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            scape()
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
