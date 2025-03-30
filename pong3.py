import cv2
import numpy as np
import pygame
import os
import time
from collections import deque

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

# Posición inicial de la bola
ballPosition = [int(WIDTH // 2), int(HEIGHT // 2)]
ballSpeedX = INITIAL_BALL_SPEED
ballSpeedY = INITIAL_BALL_SPEED

# Puntuación
left_score = 0
right_score = 0
last_touched = None  # 1: izquierda, 2: derecha

# Variables para el seguimiento de rectángulos
left_paddle = [(50, HEIGHT//2 - 50), (100, HEIGHT//2 + 50)]
right_paddle = [(WIDTH-100, HEIGHT//2 - 50), (WIDTH-50, HEIGHT//2 + 50)]
paddle_speed = 5

def draw_ball(frame):
    center = (int(round(ballPosition[0])), int(round(ballPosition[1])))
    cv2.circle(frame, center, BALL_SIZE, (255, 255, 255), -1)

def draw_middle_line(frame):
    for y in range(0, HEIGHT, 20):
        if (y // 10) % 2 == 0:
            cv2.line(frame, (WIDTH // 2, y), (WIDTH // 2, y + 10), (255, 255, 255), 2)

def draw_paddles(frame):
    # Dibujar paletas amarillas
    cv2.rectangle(frame, left_paddle[0], left_paddle[1], (0, 255, 255), -1)
    cv2.rectangle(frame, right_paddle[0], right_paddle[1], (0, 255, 255), -1)

def update_ball_position():
    global ballPosition, ballSpeedX, ballSpeedY, left_score, right_score, last_touched

    next_x = float(ballPosition[0]) + ballSpeedX
    next_y = float(ballPosition[1]) + ballSpeedY

    # Rebotar en bordes superior/inferior
    if next_y <= 0 or next_y >= HEIGHT:
        ballSpeedY = -ballSpeedY

    # Verificar colisiones con paletas
    collision = False

    # Colisión con paleta izquierda
    # Colisión con paleta izquierda
    if (left_paddle[1][0] >= next_x >= left_paddle[0][0] and
        left_paddle[0][1] <= next_y <= left_paddle[1][1]):
        ballSpeedX = -ballSpeedX
        # Añadir efecto de ángulo basado en dónde golpea la paleta
        relative_intersect_y = (left_paddle[0][1] + (left_paddle[1][1] - left_paddle[0][1])/2) - next_y
        normalized_relative_intersect_y = relative_intersect_y/((left_paddle[1][1] - left_paddle[0][1])/2)
        bounce_angle = normalized_relative_intersect_y * (5*np.pi/12)  # 75 grados máximo

        ballSpeedY = -INITIAL_BALL_SPEED * np.sin(bounce_angle)
        ballSpeedX = INITIAL_BALL_SPEED * np.cos(bounce_angle)

        last_touched = 1
        collision = True
        if pong_sound:
            pong_sound.play()

    # Colisión con paleta derecha
    elif (right_paddle[0][0] <= next_x <= right_paddle[1][0] and
          right_paddle[0][1] <= next_y <= right_paddle[1][1]):
        ballSpeedX = -ballSpeedX
        # Añadir efecto de ángulo basado en dónde golpea la paleta
        relative_intersect_y = (right_paddle[0][1]) + (right_paddle[1][1] - right_paddle[0][1])/2 - next_y
        normalized_relative_intersect_y = relative_intersect_y/((right_paddle[1][1] - right_paddle[0][1])/2)
        bounce_angle = normalized_relative_intersect_y * (5*np.pi/12)  # 75 grados máximo

        ballSpeedY = -INITIAL_BALL_SPEED * np.sin(bounce_angle)
        ballSpeedX = -INITIAL_BALL_SPEED * np.cos(bounce_angle)

        last_touched = 2
        collision = True
        if pong_sound:
            pong_sound.play()

    # Actualizar posición
    ballPosition[0] = int(round(next_x))
    ballPosition[1] = int(round(next_y))

    # Asignar puntos según el borde alcanzado
    if ballPosition[0] <= 0:
        right_score += 1  # Punto para la derecha
        last_touched = None
        if win_sound:
            win_sound.play()
        reset_ball()
    elif ballPosition[0] >= WIDTH:
        left_score += 1  # Punto para la izquierda
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

def draw_debug_info(frame):
    # Mostrar información de debug en la esquina inferior izquierda
    debug_texts = [
        f"Ball Position: ({ballPosition[0]}, {ballPosition[1]})",
        f"Ball Speed: ({ballSpeedX:.1f}, {ballSpeedY:.1f})",
        f"Last Touched: {'Left' if last_touched == 1 else 'Right' if last_touched == 2 else 'None'}"
    ]

    for i, text in enumerate(debug_texts):
        cv2.putText(frame, text, (10, HEIGHT - 30 - (i * 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo capturar el frame")
            break

        frame = cv2.flip(frame, 1)

        # Control de paletas con teclado
        key = cv2.waitKey(1)
        if key == ord('w') and left_paddle[0][1] > 0:
            left_paddle[0] = (left_paddle[0][0], left_paddle[0][1] - paddle_speed)
            left_paddle[1] = (left_paddle[1][0], left_paddle[1][1] - paddle_speed)
        elif key == ord('s') and left_paddle[1][1] < HEIGHT:
            left_paddle[0] = (left_paddle[0][0], left_paddle[0][1] + paddle_speed)
            left_paddle[1] = (left_paddle[1][0], left_paddle[1][1] + paddle_speed)
        elif key == ord('i') and right_paddle[0][1] > 0:
            right_paddle[0] = (right_paddle[0][0], right_paddle[0][1] - paddle_speed)
            right_paddle[1] = (right_paddle[1][0], right_paddle[1][1] - paddle_speed)
        elif key == ord('k') and right_paddle[1][1] < HEIGHT:
            right_paddle[0] = (right_paddle[0][0], right_paddle[0][1] + paddle_speed)
            right_paddle[1] = (right_paddle[1][0], right_paddle[1][1] + paddle_speed)

        # Dibujar elementos del juego
        draw_ball(frame)
        draw_middle_line(frame)
        draw_paddles(frame)
        draw_score(frame)
        update_ball_position()
        draw_debug_info(frame)

        cv2.imshow("Pong AR - Rectangles", frame)

        if key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
